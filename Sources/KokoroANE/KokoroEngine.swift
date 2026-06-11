import CoreML
import Foundation

/// The seven-stage Kokoro-82M Core ML pipeline (ALBERT → post-ALBERT → alignment → prosody → noise →
/// vocoder → tail), orchestrated in Swift. Synthesis takes a phoneme string (G2P is the caller's
/// responsibility) plus a voice pack and returns 24 kHz mono PCM.
///
/// The models directory must contain the seven **pre-compiled** `.mlmodelc` bundles produced by
/// `convert.py` + `coremlcompiler` (`KokoroAlbert.mlmodelc` … `KokoroTail.mlmodelc`), so no Core ML
/// compilation happens on device — models are usable as soon as they are downloaded. An uncompiled
/// `<name>.mlpackage` next to a missing `.mlmodelc` is compiled once and cached, but that path is a
/// development convenience, not the shipping flow.
///
/// Hardware placement is governed by `KokoroStageComputeUnits`; the `backgroundSafe` policy keeps every
/// stage off the GPU so a backgrounded iOS host can keep synthesizing.
public final class KokoroEngine {

    // MARK: Lifecycle

    public init(
        modelsDirectory: URL,
        computeUnits: KokoroStageComputeUnits = .backgroundSafe
    ) throws {
        albert = try Self.loadModel("KokoroAlbert", in: modelsDirectory, units: computeUnits.albert)
        postAlbert = try Self.loadModel("KokoroPostAlbert", in: modelsDirectory, units: computeUnits.postAlbert)
        alignment = try Self.loadModel("KokoroAlignment", in: modelsDirectory, units: computeUnits.alignment)
        prosody = try Self.loadModel("KokoroProsody", in: modelsDirectory, units: computeUnits.prosody)
        noise = try Self.loadModel("KokoroNoise", in: modelsDirectory, units: computeUnits.noise)
        vocoder = try Self.loadModel("KokoroVocoder", in: modelsDirectory, units: computeUnits.vocoder)
        tail = try Self.loadModel("KokoroTail", in: modelsDirectory, units: computeUnits.tail)
        vocabulary = try KokoroVocabulary()
    }

    // MARK: Public

    /// Kokoro emits 24 kHz mono audio.
    public static let sampleRate = 24000

    /// ALBERT's phoneme window: 510 phonemes plus BOS/EOS.
    public static let maximumTokenCount = 512

    /// The compiled model bundles `modelsDirectory` must contain.
    public static let modelFileNames = [
        "KokoroAlbert.mlmodelc",
        "KokoroPostAlbert.mlmodelc",
        "KokoroAlignment.mlmodelc",
        "KokoroProsody.mlmodelc",
        "KokoroNoise.mlmodelc",
        "KokoroVocoder.mlmodelc",
        "KokoroTail.mlmodelc",
    ]

    public let vocabulary: KokoroVocabulary

    /// Synthesize one utterance. `phonemes` is the Kokoro phoneme alphabet (Misaki/espeak style); callers
    /// chunk long text so the tokenized utterance stays within `maximumTokenCount`.
    public func synthesize(
        phonemes: String,
        voice: KokoroVoicePack,
        speed: Float = 1.0
    ) throws -> [Float] {
        let inputIds = vocabulary.tokens(for: phonemes)
        let tokenCount = inputIds.count
        guard tokenCount <= Self.maximumTokenCount else {
            throw KokoroANEError.utteranceTooLong(tokenCount: tokenCount)
        }
        guard tokenCount > 2 else {
            return []
        }

        // Stock Kokoro selects the style row by the phoneme *character* count, not the token count.
        let styleS = voice.styleS(forPhonemeCount: phonemes.count)
        let styleTimbre = voice.styleTimbre(forPhonemeCount: phonemes.count)
        let mask = [Int32](repeating: 1, count: tokenCount)

        // 1. ALBERT: token ids → contextual phoneme embeddings.
        let albertOutputs = try albert.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": try MLMultiArrayConversions.int32Array(inputIds, shape: [1, tokenCount]),
            "attention_mask": try MLMultiArrayConversions.int32Array(mask, shape: [1, tokenCount]),
        ]))
        let bertDur = try output(albertOutputs, stage: "albert", feature: "bert_dur")

        // 2. Post-ALBERT: per-token durations plus the duration-encoder hidden states and text encoding.
        let postAlbertOutputs = try postAlbert.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "bert_dur": bertDur,
            "input_ids": try MLMultiArrayConversions.int32Array(inputIds, shape: [1, tokenCount]),
            "style_s": try MLMultiArrayConversions.float16Array(styleS, shape: [1, 128]),
            "speed": try MLMultiArrayConversions.float16Array([speed], shape: [1]),
            "attention_mask": try MLMultiArrayConversions.int32Array(mask, shape: [1, tokenCount]),
        ]))
        let duration = try output(postAlbertOutputs, stage: "post_albert", feature: "duration")
        let durationHidden = try output(postAlbertOutputs, stage: "post_albert", feature: "d")
        let textEncoding = try output(postAlbertOutputs, stage: "post_albert", feature: "t_en")

        // 3. Alignment: integer per-token durations → frame-aligned encodings. The rounding and the ≥1
        // floor mirror the upstream Python pipeline.
        let durationFloats = MLMultiArrayConversions.floats(from: duration)
        var predictedDurations = [Int32](repeating: 0, count: tokenCount)
        for index in 0 ..< tokenCount {
            predictedDurations[index] = max(1, Int32(durationFloats[index].rounded()))
        }
        let alignmentOutputs = try alignment.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pred_dur": try MLMultiArrayConversions.int32Array(predictedDurations, shape: [1, tokenCount]),
            "d": durationHidden,
            "t_en": textEncoding,
        ]))
        let alignedEncoding = try output(alignmentOutputs, stage: "alignment", feature: "en")
        let alignedText = try output(alignmentOutputs, stage: "alignment", feature: "asr")

        // 4. Prosody: F0 and noise curves.
        let prosodyOutputs = try prosody.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "en": alignedEncoding,
            "style_s": try MLMultiArrayConversions.float16Array(styleS, shape: [1, 128]),
        ]))
        let f0Curve = try output(prosodyOutputs, stage: "prosody", feature: "F0")
        let noiseCurve = try output(prosodyOutputs, stage: "prosody", feature: "N")

        // 5. Noise (fp32): harmonic/noise source excitation from the F0 curve.
        let f0Floats = MLMultiArrayConversions.floats(from: f0Curve)
        let noiseOutputs = try noise.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "F0_curve": try MLMultiArrayConversions.floatArray(f0Floats, shape: [1, f0Floats.count]),
            "style_timbre": try MLMultiArrayConversions.floatArray(styleTimbre, shape: [1, 128]),
        ]))
        let source0 = try output(noiseOutputs, stage: "noise", feature: "x_source_0")
        let source1 = try output(noiseOutputs, stage: "noise", feature: "x_source_1")

        // 6. Vocoder (fp16, ANE): the bulk of the decoder. The anchor output exists only to keep the
        // graph ANE-resident; only `x_pre` feeds the tail.
        let vocoderOutputs = try vocoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "asr": alignedText,
            "F0_curve": f0Curve,
            "N_pred": noiseCurve,
            "x_source_0": source0,
            "x_source_1": source1,
            "style_timbre": try MLMultiArrayConversions.float16Array(styleTimbre, shape: [1, 128]),
        ]))
        let prePostConv = try output(vocoderOutputs, stage: "vocoder", feature: "x_pre")

        // 7. Tail (fp32): conv_post + exp/sin + iSTFT → PCM. The vocoder declares no static shape for
        // `x_pre` and different Core ML backends report different ranks for it (rank 3 on macOS, but
        // ANE-compiled graphs can add leading singleton dimensions), so the frame count is derived from
        // the element count — a positional `shape[2]` read mis-sizes the tail input on those backends.
        let prePostConvFloats = MLMultiArrayConversions.floats(from: prePostConv)
        guard !prePostConvFloats.isEmpty, prePostConvFloats.count % 128 == 0 else {
            throw KokoroANEError.unexpectedStageOutput(
                stage: "vocoder",
                feature: "x_pre",
                descriptor: MLMultiArrayConversions.describe(prePostConv)
            )
        }
        let frameCount = prePostConvFloats.count / 128
        return try tailAudio(prePostConvFloats: prePostConvFloats, frameCount: frameCount)
    }

    // MARK: Private

    /// The tail's iSTFT hop: every `x_pre` position becomes 5 audio samples (deconv kernel 20, stride 5,
    /// `n_fft/2 = 10` cropped from each end), so a `T`-position input yields exactly `5·T − 5` samples.
    private static let tailHop = 5

    /// Maximum `x_pre` positions per tail invocation. Upstream only ever ran the tail on whole utterances
    /// with `.all` compute units (GPU); under the background-safe CPU+ANE policy on iPhone, large inputs
    /// crash inside the Core ML runtime (EXC_BAD_ACCESS) — ~9.6k positions are proven fine on device while
    /// ~96k crash, and the ANE's 16384 dimension ceiling sits between. Windowing bounds every call to the
    /// proven scale; it is numerically exact because the tail is local (conv_post reads ±3 positions, the
    /// iSTFT deconv spans 20 positions at stride 5 — both far smaller than `tailWindowHalo`).
    private static let tailWindowLength = 9600

    /// Positions of context included on each interior window edge. Samples within the halo are discarded,
    /// so each emitted sample sees exactly the neighborhood it would in a whole-utterance call. The math
    /// needs ≥ 5 (conv_post ±3 plus the deconv's 4-position overlap); 16 leaves margin.
    private static let tailWindowHalo = 16

    /// The converted tail declares a 100-position lower bound on `x_pre`, so a window is never narrower
    /// than this — a short final window is widened backward into already-emitted territory instead (the
    /// extra positions are halo and produce no samples).
    private static let tailWindowMinimumWidth = 128

    private let albert: MLModel
    private let postAlbert: MLModel
    private let alignment: MLModel
    private let prosody: MLModel
    private let noise: MLModel
    private let vocoder: MLModel
    private let tail: MLModel

    /// Load `<name>.mlmodelc` from the models directory. If only a `.mlpackage` is present (local
    /// development), compile it once and cache the result alongside it.
    private static func loadModel(_ name: String, in directory: URL, units: MLComputeUnits) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units

        let compiledURL = directory.appendingPathComponent("\(name).mlmodelc")
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }

        let packageURL = directory.appendingPathComponent("\(name).mlpackage")
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw KokoroANEError.modelMissing(name)
        }

        let temporaryCompiledURL = try MLModel.compileModel(at: packageURL)
        try? FileManager.default.moveItem(at: temporaryCompiledURL, to: compiledURL)
        let cachedURL = FileManager.default.fileExists(atPath: compiledURL.path) ? compiledURL : temporaryCompiledURL
        return try MLModel(contentsOf: cachedURL, configuration: configuration)
    }

    private func output(
        _ provider: MLFeatureProvider,
        stage: String,
        feature: String
    ) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: feature)?.multiArrayValue else {
            throw KokoroANEError.missingOutput(stage: stage, feature: feature)
        }
        return value
    }

    /// Runs the tail over `x_pre` (packed `[1, 128, frameCount]`), windowing the position axis so no
    /// single Core ML invocation exceeds `tailWindowLength` positions. Output samples are concatenated
    /// from each window's central span, which the halo makes identical to a whole-utterance call.
    private func tailAudio(prePostConvFloats: [Float], frameCount: Int) throws -> [Float] {
        if frameCount <= Self.tailWindowLength {
            return try tailWindowSamples(
                prePostConvFloats,
                frameCount: frameCount,
                input: 0 ..< frameCount,
                emitting: 0 ..< frameCount
            )
        }

        var audio = [Float]()
        audio.reserveCapacity(Self.tailHop * frameCount - Self.tailHop)
        let step = Self.tailWindowLength - 2 * Self.tailWindowHalo
        var start = 0
        while start < frameCount {
            let end = min(start + step, frameCount)
            var inputStart = max(0, start - Self.tailWindowHalo)
            let inputEnd = min(frameCount, end + Self.tailWindowHalo)
            if inputEnd - inputStart < Self.tailWindowMinimumWidth {
                inputStart = max(0, inputEnd - Self.tailWindowMinimumWidth)
            }
            let samples = try tailWindowSamples(
                prePostConvFloats,
                frameCount: frameCount,
                input: inputStart ..< inputEnd,
                emitting: start ..< end
            )
            audio.append(contentsOf: samples)
            start = end
        }
        return audio
    }

    /// One tail invocation over `input` positions of `x_pre`, returning only the samples belonging to the
    /// `emitting` positions. `input` must contain `emitting` plus enough halo for exactness (or touch the
    /// true utterance boundary, where the model's own zero-padding is the correct global behavior).
    private func tailWindowSamples(
        _ prePostConvFloats: [Float],
        frameCount: Int,
        input: Range<Int>,
        emitting: Range<Int>
    ) throws -> [Float] {
        let width = input.count
        let windowValues: [Float]
        if width == frameCount {
            windowValues = prePostConvFloats
        } else {
            var sliced = [Float](repeating: 0, count: 128 * width)
            for channel in 0 ..< 128 {
                let sourceStart = channel * frameCount + input.lowerBound
                let destinationStart = channel * width
                sliced.replaceSubrange(
                    destinationStart ..< destinationStart + width,
                    with: prePostConvFloats[sourceStart ..< sourceStart + width]
                )
            }
            windowValues = sliced
        }

        let tailOutputs = try tail.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "x_pre": try MLMultiArrayConversions.floatArray(windowValues, shape: [1, 128, width]),
        ]))
        let audio = try output(tailOutputs, stage: "tail", feature: "audio")
        let samples = MLMultiArrayConversions.floats(from: audio)
        let expectedCount = Self.tailHop * width - Self.tailHop
        guard samples.count == expectedCount else {
            throw KokoroANEError.unexpectedStageOutput(
                stage: "tail",
                feature: "audio",
                descriptor: MLMultiArrayConversions.describe(audio)
                    + " (expected \(expectedCount) samples for \(width) x_pre positions)"
            )
        }

        // A window's sample for global position g sits at local index `hop·g − hop·input.lowerBound`. The
        // final positions of the utterance produce `hop` fewer samples (the model crops `n_fft/2` from
        // each end), hence the clamp.
        let globalSampleStart = Self.tailHop * emitting.lowerBound
        let globalSampleEnd = min(
            Self.tailHop * emitting.upperBound,
            Self.tailHop * frameCount - Self.tailHop
        )
        let localStart = globalSampleStart - Self.tailHop * input.lowerBound
        let localEnd = globalSampleEnd - Self.tailHop * input.lowerBound
        return Array(samples[localStart ..< localEnd])
    }
}
