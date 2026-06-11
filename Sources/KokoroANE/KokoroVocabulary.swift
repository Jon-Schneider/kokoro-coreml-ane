import Foundation

/// The Kokoro phoneme → token-id table (`vocab.json`, bundled as a package resource). Token id 0 is the
/// shared BOS/EOS marker; characters absent from the table (unsupported phonemes, stray symbols) are
/// silently dropped, matching the upstream Python pipeline's `vocab.get` filtering.
public struct KokoroVocabulary: Sendable {

    // MARK: Lifecycle

    public init() throws {
        guard let url = Bundle.module.url(forResource: "vocab", withExtension: "json") else {
            throw KokoroANEError.vocabularyResourceMissing
        }
        let data = try Data(contentsOf: url)
        guard let map = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw KokoroANEError.vocabularyResourceMissing
        }
        self.map = map
    }

    // MARK: Public

    /// Token ids for a phoneme string, wrapped with BOS/EOS (0).
    public func tokens(for phonemes: String) -> [Int32] {
        var ids: [Int32] = [0]
        for character in phonemes {
            if let id = map[String(character)] {
                ids.append(Int32(id))
            }
        }
        ids.append(0)
        return ids
    }

    // MARK: Private

    private let map: [String: Int]
}
