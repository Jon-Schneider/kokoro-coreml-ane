import CoreML

/// Per-stage Core ML compute-unit assignments for the seven Kokoro pipeline stages.
///
/// The pipeline's stages have very different hardware affinities: the fp16 convolution-heavy stages (ALBERT,
/// vocoder) belong on the Neural Engine, while the small fp32 stages (noise, tail) fall back to whatever
/// non-ANE unit the policy allows. The policy exists so hosts can guarantee that no stage ever schedules
/// Metal work: iOS kills apps that touch the GPU while backgrounded, so an audio app narrating in the
/// background must keep every stage on the CPU and Neural Engine.
public struct KokoroStageComputeUnits: Sendable {

    // MARK: Lifecycle

    public init(
        albert: MLComputeUnits,
        postAlbert: MLComputeUnits,
        alignment: MLComputeUnits,
        prosody: MLComputeUnits,
        noise: MLComputeUnits,
        vocoder: MLComputeUnits,
        tail: MLComputeUnits
    ) {
        self.albert = albert
        self.postAlbert = postAlbert
        self.alignment = alignment
        self.prosody = prosody
        self.noise = noise
        self.vocoder = vocoder
        self.tail = tail
    }

    // MARK: Public

    /// Every stage restricted to CPU + Neural Engine. No stage can submit GPU work, so synthesis keeps
    /// running when the host app is backgrounded. The fp32 stages run on the CPU under this policy.
    public static let backgroundSafe = KokoroStageComputeUnits(uniform: .cpuAndNeuralEngine)

    /// The upstream iOS demo's assignment: ANE-friendly stages pinned to CPU+ANE, the fp32 stages
    /// (prosody/noise/tail) left on `.all` so Core ML may schedule them on the GPU. Fastest in the
    /// foreground, but not safe while backgrounded on iOS.
    public static let upstreamDemo = KokoroStageComputeUnits(
        albert: .cpuAndNeuralEngine,
        postAlbert: .cpuAndNeuralEngine,
        alignment: .cpuAndNeuralEngine,
        prosody: .all,
        noise: .all,
        vocoder: .cpuAndNeuralEngine,
        tail: .all
    )

    /// Every stage restricted to the CPU. A debugging escape hatch for isolating ANE-specific issues.
    public static let cpuOnly = KokoroStageComputeUnits(uniform: .cpuOnly)

    public let albert: MLComputeUnits
    public let postAlbert: MLComputeUnits
    public let alignment: MLComputeUnits
    public let prosody: MLComputeUnits
    public let noise: MLComputeUnits
    public let vocoder: MLComputeUnits
    public let tail: MLComputeUnits

    // MARK: Private

    private init(uniform: MLComputeUnits) {
        self.init(
            albert: uniform,
            postAlbert: uniform,
            alignment: uniform,
            prosody: uniform,
            noise: uniform,
            vocoder: uniform,
            tail: uniform
        )
    }
}
