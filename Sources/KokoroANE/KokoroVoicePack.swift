import Foundation

/// One Kokoro voice: the raw `[510, 256]` style table exported from the upstream voice `.pt`/`.npz` packs
/// as little-endian float32 (`<name>.bin`). Row `N-1` (clamped) is the style for an utterance of `N`
/// phoneme characters — stock Kokoro's `voice[len(ps) - 1]` indexing. The first 128 columns condition
/// timbre (decoder/vocoder), the second 128 condition prosody (duration/F0).
public struct KokoroVoicePack: Sendable {

    // MARK: Lifecycle

    public init(name: String, contentsOf url: URL) throws {
        let bytes = try Data(contentsOf: url)
        guard bytes.count == Self.rowCount * Self.rowStride * MemoryLayout<Float>.size else {
            throw KokoroANEError.voicePackCorrupt(name)
        }

        var floats = [Float](repeating: 0, count: bytes.count / MemoryLayout<Float>.size)
        _ = floats.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
        self.name = name
        data = floats
    }

    // MARK: Public

    public let name: String

    /// Style `s` embedding `[128]` (second half of the voice row for an utterance of `phonemeCount`
    /// phoneme characters).
    public func styleS(forPhonemeCount phonemeCount: Int) -> [Float] {
        let start = rowStart(forPhonemeCount: phonemeCount) + Self.styleDimension
        return Array(data[start ..< start + Self.styleDimension])
    }

    /// Timbre embedding `[128]` (first half of the voice row for an utterance of `phonemeCount` phoneme
    /// characters).
    public func styleTimbre(forPhonemeCount phonemeCount: Int) -> [Float] {
        let start = rowStart(forPhonemeCount: phonemeCount)
        return Array(data[start ..< start + Self.styleDimension])
    }

    // MARK: Private

    private static let rowCount = 510
    private static let styleDimension = 128
    private static let rowStride = 256

    private let data: [Float]

    private func rowStart(forPhonemeCount phonemeCount: Int) -> Int {
        let row = min(max(phonemeCount - 1, 0), Self.rowCount - 1)
        return row * Self.rowStride
    }
}
