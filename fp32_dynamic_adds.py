"""Selective FP32 arithmetic for dynamic-width FP16 adds (LINKS-1995).

Rewrites each FP16 `add` whose output's last dimension is dynamic as

    a32 = cast(a, "fp32")
    b32 = cast(b, "fp32")
    y32 = add(a32, b32)
    y16 = cast(y32, "fp16")        # original output name/type preserved

Every selected add still emits the original FP16-rounded value at its boundary —
including between adjacent selected adds. For two FP16 operands the FP32 sum is
exact and singly rounded on narrowing, so the result is bit-identical to a
correctly-rounded FP16 add.

Operates on the serialized mlprogram spec; weight blobs are untouched.
"""
from coremltools.proto import MIL_pb2

FP32 = MIL_pb2.DataType.Value("FLOAT32")
FP16 = MIL_pb2.DataType.Value("FLOAT16")
STRING = MIL_pb2.DataType.Value("STRING")


def _dtype_const_op(name, dtype_str):
    op = MIL_pb2.Operation()
    op.type = "const"
    out = op.outputs.add()
    out.name = name
    out.type.tensorType.dataType = STRING
    val = op.attributes["val"]
    val.type.tensorType.dataType = STRING
    val.immediateValue.tensor.strings.values.append(dtype_str)
    _string_attr(op, "name", f"{name}_op")
    return op


def _string_attr(op, key, s):
    a = op.attributes[key]
    a.type.tensorType.dataType = STRING
    a.immediateValue.tensor.strings.values.append(s)


def _single_input_name(op, key):
    args = op.inputs[key].arguments
    return args[0].name if len(args) == 1 else None


def _cast_op(name, src_name, src_type, dtype_str, dtype_const, op_name):
    op = MIL_pb2.Operation()
    op.type = "cast"
    op.inputs["x"].arguments.add().name = src_name
    op.inputs["dtype"].arguments.add().name = dtype_const
    out = op.outputs.add()
    out.name = name
    out.type.CopyFrom(src_type)
    out.type.tensorType.dataType = FP32 if dtype_str == "fp32" else FP16
    _string_attr(op, "name", op_name)
    return op


def _is_dynamic_last_dim(tensor_type):
    tt = tensor_type.tensorType
    return tt.rank >= 1 and tt.dimensions[-1].WhichOneof("dimension") != "constant"


def fp32_dynamic_adds(spec, select=None):
    """Rewrite selected FP16 adds in-place on `spec`. Returns a manifest dict."""
    fn = spec.mlProgram.functions["main"]
    manifest = {"selected": [], "skipped_fp16_adds": [], "skipped_non_fp16_adds": []}

    for block in fn.block_specializations.values():
        var_types = {i.name: i.type for i in fn.inputs}
        for inp in block.inputs:
            var_types[inp.name] = inp.type
        for op in block.operations:
            for out in op.outputs:
                var_types[out.name] = out.type

        new_ops = []
        for op in block.operations:
            out = op.outputs[0]
            selected = (
                op.type == "add"
                and out.type.tensorType.dataType == FP16
                and _is_dynamic_last_dim(out.type)
                and (select is None or select(op, out.type))
            )
            if not selected:
                if op.type == "add":
                    key = "skipped_fp16_adds" if out.type.tensorType.dataType == FP16 else "skipped_non_fp16_adds"
                    manifest[key].append(out.name)
                new_ops.append(op)
                continue

            x_name = _single_input_name(op, "x")
            y_name = _single_input_name(op, "y")
            if x_name not in var_types or y_name not in var_types:
                manifest["skipped_fp16_adds"].append(f"{out.name} (unresolvable input)")
                new_ops.append(op)
                continue

            base = f"fp32add_{out.name}"
            x32 = _cast_op(f"{base}_x32", x_name, var_types[x_name], "fp32",
                           "fp32add_dtype_fp32", f"{base}_x32_op")
            y32 = _cast_op(f"{base}_y32", y_name, var_types[y_name], "fp32",
                           "fp32add_dtype_fp32", f"{base}_y32_op")
            add32 = MIL_pb2.Operation()
            add32.type = "add"
            add32.inputs["x"].arguments.add().name = f"{base}_x32"
            add32.inputs["y"].arguments.add().name = f"{base}_y32"
            a32_out = add32.outputs.add()
            a32_out.name = f"{base}_f32"
            a32_out.type.CopyFrom(out.type)
            a32_out.type.tensorType.dataType = FP32
            _string_attr(add32, "name", f"{base}_add32_op")
            y16 = _cast_op(out.name, f"{base}_f32", out.type, "fp16",
                           "fp32add_dtype_fp16", f"{base}_y16_op")
            new_ops += [x32, y32, add32, y16]
            manifest["selected"].append(out.name)

        del block.operations[:]
        if manifest["selected"]:
            block.operations.append(_dtype_const_op("fp32add_dtype_fp32", "fp32"))
            block.operations.append(_dtype_const_op("fp32add_dtype_fp16", "fp16"))
        block.operations.extend(new_ops)
    return manifest


def apply_fp32_dynamic_adds(mlpackage_path, select=None, mlmodel_cls=None):
    """Apply the rewrite to a saved .mlpackage in place. Returns the manifest."""
    import os
    import shutil
    import tempfile

    import coremltools as ct

    src = ct.models.MLModel(str(mlpackage_path))
    spec = src.get_spec()
    manifest = fp32_dynamic_adds(spec, select=select)
    if not manifest["selected"]:
        return manifest
    weights_dir = os.path.join(str(mlpackage_path), "Data", "com.apple.CoreML", "weights")
    rewritten = ct.models.MLModel(spec, weights_dir=weights_dir, skip_model_load=True)
    tmp = tempfile.mkdtemp(suffix=".mlpackage")
    rewritten.save(tmp)
    shutil.rmtree(str(mlpackage_path))
    shutil.move(tmp, str(mlpackage_path))
    return manifest
