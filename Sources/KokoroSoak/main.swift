// Throwaway Mac soak harness for the narration crash hunt: drives the KokoroANE engine through a battery
// of utterance lengths (tiny → ALBERT's 510-phoneme cap) under the background-safe compute policy — the
// same policy Stache uses on iPhone, which on the Mac also pins the fp32 tail to the CPU — and optionally
// plays the audio through AVAudioEngine the way the app's streaming player does.
//
// Every step prints a BEGIN line before it runs, so if the process crashes the last line of output names
// the exact stage and utterance length — no re-run needed to localize it.
//
// Usage:
//   swift run -c release kokoro-soak <models-dir> <voice.bin> [--play] [--units backgroundSafe|upstreamDemo|cpuOnly]
//
// Exits 0 and prints SOAK PASSED when every case synthesizes (and plays, with --play) cleanly.
import AVFoundation
import Foundation
import KokoroANE

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    FileHandle.standardError.write(Data("""
    usage: kokoro-soak <models-dir> <voice.bin> [--play] [--units backgroundSafe|upstreamDemo|cpuOnly]
    """.utf8))
    exit(2)
}

let modelsDirectory = URL(fileURLWithPath: arguments[1], isDirectory: true)
let voiceURL = URL(fileURLWithPath: arguments[2])
let shouldPlay = arguments.contains("--play")
let unitsName = arguments.firstIndex(of: "--units").flatMap { index in
    arguments.indices.contains(index + 1) ? arguments[index + 1] : nil
} ?? "backgroundSafe"

let computeUnits: KokoroStageComputeUnits
switch unitsName {
case "backgroundSafe": computeUnits = .backgroundSafe
case "upstreamDemo": computeUnits = .upstreamDemo
case "cpuOnly": computeUnits = .cpuOnly
default:
    FileHandle.standardError.write(Data("unknown units: \(unitsName)\n".utf8))
    exit(2)
}

func log(_ message: String) {
    print("[soak] \(message)")
    // Unbuffered so a crash can't eat the last line.
    fflush(stdout)
}

/// Builds an utterance of roughly `targetPhonemeCount` phonemes from a pangram (the same one
/// kokoro-validate scores against the PyTorch reference).
func utterance(ofPhonemeCount targetPhonemeCount: Int) -> String {
    let pangram = "ðə kwɪk bɹaʊn fɑːks dʒʌmps oʊvɚ ðə leɪzi dɑːɡ."
    var result = ""
    while result.count < targetPhonemeCount {
        result += result.isEmpty ? pangram : " " + pangram
    }
    return String(result.prefix(targetPhonemeCount))
}

/// Streams chunks through AVAudioEngine like the app's player: schedule each chunk's buffer as soon as it
/// is synthesized, then wait for the last one to finish.
final class SoakPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat

    init(sampleRate: Double) throws {
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1) else {
            throw NSError(domain: "soak", code: 1, userInfo: [NSLocalizedDescriptionKey: "no format"])
        }
        self.format = format
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try engine.start()
        player.play()
    }

    func schedule(_ samples: [Float]) throws {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw NSError(domain: "soak", code: 2, userInfo: [NSLocalizedDescriptionKey: "no buffer"])
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress, let channel = buffer.floatChannelData?[0] else {
                return
            }
            channel.update(from: base, count: samples.count)
        }
        let done = DispatchSemaphore(value: 0)
        player.scheduleBuffer(buffer) { done.signal() }
        // Block until this buffer has played so soak output stays interleaved with playback progress.
        done.wait()
    }

    func stop() {
        player.stop()
        engine.stop()
    }
}

do {
    log("loading engine from \(modelsDirectory.path) (units: \(unitsName))")
    let loadStart = Date()
    let engine = try KokoroEngine(modelsDirectory: modelsDirectory, computeUnits: computeUnits)
    let voiceName = voiceURL.deletingPathExtension().lastPathComponent
    let voice = try KokoroVoicePack(name: voiceName, contentsOf: voiceURL)
    log("engine + voice '\(voiceName)' loaded in \(String(format: "%.0f", -loadStart.timeIntervalSinceNow * 1000))ms")

    let player = shouldPlay ? try SoakPlayer(sampleRate: Double(KokoroEngine.sampleRate)) : nil
    if shouldPlay {
        log("playback: ON (AVAudioEngine, streaming per-chunk like the app)")
    }

    // Title-sized through max-length utterances. 509 phonemes is ALBERT's cap (510) minus headroom; the
    // long cases force multiple tail windows — the configuration that crashed on iPhone.
    let phonemeCounts = [12, 47, 95, 190, 300, 380, 469, 509]
    var totalAudioSeconds = 0.0
    for (index, phonemeCount) in phonemeCounts.enumerated() {
        let phonemes = utterance(ofPhonemeCount: phonemeCount)
        log("case \(index + 1)/\(phonemeCounts.count) BEGIN synthesize: \(phonemes.count) phonemes")
        let start = Date()
        let samples = try engine.synthesize(phonemes: phonemes, voice: voice)
        let elapsed = -start.timeIntervalSinceNow
        let seconds = Double(samples.count) / Double(KokoroEngine.sampleRate)
        totalAudioSeconds += seconds
        log(String(format: "case %d OK: %.2fs audio, %d samples, %.0fms (%.1fx real-time)",
                   index + 1, seconds, samples.count, elapsed * 1000, seconds / elapsed))

        guard samples.allSatisfy({ $0.isFinite }) else {
            log("case \(index + 1) FAILED: non-finite samples in output")
            exit(1)
        }

        if let player {
            log("case \(index + 1) BEGIN playback")
            try player.schedule(samples)
            log("case \(index + 1) playback done")
        }
    }

    player?.stop()
    log(String(format: "SOAK PASSED — %d cases, %.1fs total audio, units=%@", phonemeCounts.count, totalAudioSeconds, unitsName))
} catch {
    log("SOAK FAILED: \(error)")
    exit(1)
}
