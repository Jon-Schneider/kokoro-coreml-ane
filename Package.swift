// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "KokoroANE",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .macCatalyst(.v16),
    ],
    products: [
        .library(name: "KokoroANE", targets: ["KokoroANE"]),
        .executable(name: "kokoro-validate", targets: ["KokoroValidate"]),
    ],
    targets: [
        .target(
            name: "KokoroANE",
            path: "Sources/KokoroANE",
            resources: [
                .process("Resources/vocab.json"),
            ]
        ),
        .executableTarget(
            name: "KokoroValidate",
            dependencies: ["KokoroANE"],
            path: "Sources/KokoroValidate"
        ),
    ]
)
