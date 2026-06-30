// Off-device validation harness: runs the Swift KokoroANE pipeline on the same phoneme string
// convert.py uses for its PyTorch reference and writes raw float32 PCM, which compare_validation.py
// scores against output/ref.wav. Usage:
//
//   swift run kokoro-validate <models-dir> <voice.bin> <out.pcm> [phonemes]
import Foundation
import KokoroANE

let arguments = CommandLine.arguments
guard arguments.count >= 4 else {
    FileHandle.standardError.write(Data("usage: kokoro-validate <models-dir> <voice.bin> <out.pcm> [phonemes]\n".utf8))
    exit(2)
}

let modelsDirectory = URL(fileURLWithPath: arguments[1], isDirectory: true)
let voiceURL = URL(fileURLWithPath: arguments[2])
let outputURL = URL(fileURLWithPath: arguments[3])
let phonemes = arguments.count > 4 ? arguments[4] : "ðə kwɪk bɹaʊn fɑːks dʒʌmps oʊvɚ ðə leɪzi dɑːɡ."

do {
    let loadStart = Date()
    let units: KokoroStageComputeUnits = ProcessInfo.processInfo
        .environment["KOKORO_UNITS"] == "cpuOnly" ? .cpuOnly : .backgroundSafe
    let engine = try KokoroEngine(modelsDirectory: modelsDirectory, computeUnits: units)
    let voiceName = voiceURL.deletingPathExtension().lastPathComponent
    let voice = try KokoroVoicePack(name: voiceName, contentsOf: voiceURL)
    print("load: \(String(format: "%.0f", -loadStart.timeIntervalSinceNow * 1000))ms")

    // Warm once (first prediction pays one-time setup), then time a clean pass.
    _ = try engine.synthesize(phonemes: phonemes, voice: voice)
    let synthesisStart = Date()
    let samples = try engine.synthesize(phonemes: phonemes, voice: voice)
    let elapsedMs = -synthesisStart.timeIntervalSinceNow * 1000
    let audioSeconds = Double(samples.count) / Double(KokoroEngine.sampleRate)
    print("synthesis: \(String(format: "%.0f", elapsedMs))ms for \(String(format: "%.2f", audioSeconds))s audio (\(String(format: "%.1f", audioSeconds * 1000 / elapsedMs))x real-time, backgroundSafe units)")

    // Duration-only pass (stages 1–2): the cheap path that powers listening-time estimates. Warm once, then
    // time a clean pass and cross-check exactness against the rendered audio. Stage 7's tail emits
    // `600·T_a` samples for `T_a` predicted frames, so `samples.count` must equal `frames · 600` exactly.
    _ = try engine.predictedFrameCount(phonemes: phonemes, voice: voice)
    let durationStart = Date()
    let frames = try engine.predictedFrameCount(phonemes: phonemes, voice: voice)
    let durationMs = -durationStart.timeIntervalSinceNow * 1000
    let predictedSeconds = KokoroEngine.seconds(forFrameCount: frames)
    let expectedSamples = frames * 600
    let exact = expectedSamples == samples.count
    print("duration: \(String(format: "%.0f", durationMs))ms for \(frames) frames → \(String(format: "%.2f", predictedSeconds))s predicted (\(String(format: "%.1f", elapsedMs / durationMs))x cheaper than synthesis)")
    print("exactness: predicted \(expectedSamples) samples vs rendered \(samples.count) — \(exact ? "EXACT ✓" : "MISMATCH ✗ (Δ\(samples.count - expectedSamples))")")
    if !exact {
        FileHandle.standardError.write(Data("duration prediction mismatch: predicted \(expectedSamples) samples, rendered \(samples.count)\n".utf8))
        exit(1)
    }

    var payload = Data(count: samples.count * MemoryLayout<Float>.size)
    payload.withUnsafeMutableBytes { buffer in
        samples.withUnsafeBytes { source in
            buffer.copyBytes(from: source)
        }
    }
    try payload.write(to: outputURL)
    print("wrote \(samples.count) samples to \(outputURL.path)")
} catch {
    FileHandle.standardError.write(Data("validation failed: \(error)\n".utf8))
    exit(1)
}
