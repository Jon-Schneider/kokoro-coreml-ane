import Foundation

public enum KokoroANEError: Error {
    /// Neither `<name>.mlmodelc` nor `<name>.mlpackage` was found in the models directory.
    case modelMissing(String)
    /// The bundled `vocab.json` resource could not be loaded.
    case vocabularyResourceMissing
    /// A voice `.bin` file had the wrong size for a `[510, 256]` float32 table.
    case voicePackCorrupt(String)
    /// A stage produced no value for an expected output feature.
    case missingOutput(stage: String, feature: String)
    /// The tokenized utterance exceeds ALBERT's 510-phoneme window (512 with BOS/EOS).
    case utteranceTooLong(tokenCount: Int)
}
