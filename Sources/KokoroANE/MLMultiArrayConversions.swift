import Accelerate
import CoreML

/// `[Float]`/`[Int32]` ↔ `MLMultiArray` conversions for the pipeline's feature dictionaries. fp32↔fp16
/// conversion goes through vImage so the per-stage glue stays off the synthesis hot path's profile.
enum MLMultiArrayConversions {

    static func floatArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
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
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))

        case .float16:
            var result = [Float](repeating: 0, count: count)
            result.withUnsafeMutableBufferPointer { destinationBuffer in
                guard let destinationBase = destinationBuffer.baseAddress else {
                    return
                }
                var vImageSource = vImage_Buffer(
                    data: array.dataPointer,
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
            return result

        default:
            var result = [Float](repeating: 0, count: count)
            for index in 0 ..< count {
                result[index] = array[index].floatValue
            }
            return result
        }
    }
}
