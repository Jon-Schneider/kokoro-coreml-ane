import Accelerate
import CoreML

/// `[Float]`/`[Int32]` ↔ `MLMultiArray` conversions for the pipeline's feature dictionaries. fp32↔fp16
/// conversion goes through vImage so the per-stage glue stays off the synthesis hot path's profile.
///
/// Two safety properties matter here because stage outputs come back from whatever backend Core ML chose:
/// the `[Float]`/`[Int32]` → `MLMultiArray` packers refuse element counts that don't match the requested
/// shape (an out-of-bounds bulk write otherwise), and the `MLMultiArray` → `[Float]` reader honors
/// `MLMultiArray.strides` — ANE-produced arrays can be padded, so assuming packed layout misreads them.
enum MLMultiArrayConversions {

    static func floatArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        try validate(valueCount: values.count, shape: shape)
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        values.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else {
                return
            }
            array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
                .initialize(from: base, count: values.count)
        }
        return array
    }

    static func float16Array(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        try validate(valueCount: values.count, shape: shape)
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        var source = values
        source.withUnsafeMutableBufferPointer { sourceBuffer in
            guard let sourceBase = sourceBuffer.baseAddress else {
                return
            }
            var vImageSource = vImage_Buffer(
                data: UnsafeMutableRawPointer(sourceBase),
                height: 1,
                width: vImagePixelCount(values.count),
                rowBytes: values.count * MemoryLayout<Float>.size
            )
            var vImageDestination = vImage_Buffer(
                data: array.dataPointer,
                height: 1,
                width: vImagePixelCount(values.count),
                rowBytes: values.count * MemoryLayout<UInt16>.size
            )
            vImageConvert_PlanarFtoPlanar16F(&vImageSource, &vImageDestination, 0)
        }
        return array
    }

    static func int32Array(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        try validate(valueCount: values.count, shape: shape)
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        values.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else {
                return
            }
            array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
                .initialize(from: base, count: values.count)
        }
        return array
    }

    static func floats(from array: MLMultiArray) -> [Float] {
        let count = array.count
        switch array.dataType {
        case .float32:
            if isPacked(array) {
                let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
                return Array(UnsafeBufferPointer(start: pointer, count: count))
            }
            var result = [Float](repeating: 0, count: count)
            result.withUnsafeMutableBufferPointer { destination in
                guard let base = destination.baseAddress else {
                    return
                }
                gather(array, elementType: Float.self, into: base)
            }
            return result

        case .float16:
            // Gather the (possibly strided) half-precision payload into a packed buffer first, then make
            // one vImage pass to widen it.
            var packed = [UInt16](repeating: 0, count: count)
            packed.withUnsafeMutableBufferPointer { destination in
                guard let base = destination.baseAddress else {
                    return
                }
                if isPacked(array) {
                    base.update(
                        from: array.dataPointer.bindMemory(to: UInt16.self, capacity: count),
                        count: count
                    )
                } else {
                    gather(array, elementType: UInt16.self, into: base)
                }
            }
            var result = [Float](repeating: 0, count: count)
            packed.withUnsafeMutableBufferPointer { sourceBuffer in
                result.withUnsafeMutableBufferPointer { destinationBuffer in
                    guard
                        let sourceBase = sourceBuffer.baseAddress,
                        let destinationBase = destinationBuffer.baseAddress
                    else {
                        return
                    }
                    var vImageSource = vImage_Buffer(
                        data: UnsafeMutableRawPointer(sourceBase),
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<UInt16>.size
                    )
                    var vImageDestination = vImage_Buffer(
                        data: UnsafeMutableRawPointer(destinationBase),
                        height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<Float>.size
                    )
                    vImageConvert_Planar16FtoPlanarF(&vImageSource, &vImageDestination, 0)
                }
            }
            return result

        default:
            // The NSNumber subscript resolves strides itself, so this path is layout-safe (and only hit
            // for types the pipeline never reads back in bulk).
            var result = [Float](repeating: 0, count: count)
            for index in 0 ..< count {
                result[index] = array[index].floatValue
            }
            return result
        }
    }

    /// Shape/strides/type/count in one string, for error messages that must make a failed run on a remote
    /// device diagnosable without reproducing it.
    static func describe(_ array: MLMultiArray) -> String {
        "shape=\(array.shape) strides=\(array.strides) dataType=\(array.dataType.rawValue) count=\(array.count)"
    }

    // MARK: Private

    private static func validate(valueCount: Int, shape: [Int]) throws {
        let expected = shape.reduce(1, *)
        guard valueCount == expected, expected > 0 else {
            throw KokoroANEError.conversionCountMismatch(valueCount: valueCount, shape: shape)
        }
    }

    /// Whether the array's strides describe a dense row-major layout, i.e. `dataPointer` can be read as
    /// `count` contiguous elements.
    private static func isPacked(_ array: MLMultiArray) -> Bool {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var expected = 1
        for dimension in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[dimension] != expected {
                return false
            }
            expected *= shape[dimension]
        }
        return true
    }

    /// Copies a strided array's elements into `destination` in logical (row-major) order. Rows — runs
    /// along the last dimension — are bulk-copied when that dimension is contiguous.
    private static func gather<Element>(
        _ array: MLMultiArray,
        elementType _: Element.Type,
        into destination: UnsafeMutablePointer<Element>
    ) {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let rank = shape.count
        guard rank > 0 else {
            return
        }

        // Capacity reflects the strided extent of the buffer, not the logical element count.
        let bufferExtent = zip(shape, strides).reduce(1) { extent, dimension in
            extent + (dimension.0 - 1) * dimension.1
        }
        let source = array.dataPointer.bindMemory(to: Element.self, capacity: bufferExtent)
        let rowLength = shape[rank - 1]
        let rowStride = strides[rank - 1]
        var leadingIndices = [Int](repeating: 0, count: rank - 1)
        var written = 0
        while true {
            var offset = 0
            for dimension in 0 ..< rank - 1 {
                offset += leadingIndices[dimension] * strides[dimension]
            }
            if rowStride == 1 {
                destination.advanced(by: written).update(from: source.advanced(by: offset), count: rowLength)
                written += rowLength
            } else {
                for element in 0 ..< rowLength {
                    destination[written] = source[offset + element * rowStride]
                    written += 1
                }
            }

            // Odometer increment over the leading dimensions; rolling over every digit means every row
            // has been copied.
            var dimension = rank - 2
            while dimension >= 0 {
                leadingIndices[dimension] += 1
                if leadingIndices[dimension] < shape[dimension] {
                    break
                }
                leadingIndices[dimension] = 0
                dimension -= 1
            }
            if dimension < 0 {
                break
            }
        }
    }
}
