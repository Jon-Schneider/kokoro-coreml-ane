"""Export Kokoro voice packs as raw float32 .bin files for the Swift pipeline.

Each voice is the upstream [510, 1, 256] style table from hexgrad/Kokoro-82M, squeezed to [510, 256] and
written little-endian float32, matching `KokoroVoicePack` in Sources/KokoroANE. Run:

    uv run python export_voices.py [--out output/voices]
"""

import argparse
from pathlib import Path

VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/voices"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from kokoro import KModel
    from kokoro.pipeline import KPipeline

    model = KModel()
    model.eval()
    # Voices are accent-split across pipelines: 'a' = American English, 'b' = British English.
    pipes = {
        "a": KPipeline(lang_code="a", model=model),
        "b": KPipeline(lang_code="b", model=model),
    }

    for name in VOICES:
        pack = pipes[name[0]].load_voice(name)  # [510, 1, 256]
        table = pack.squeeze(1).contiguous().float().numpy()
        assert table.shape == (510, 256), f"{name}: unexpected shape {table.shape}"
        path = args.out / f"{name}.bin"
        path.write_bytes(table.astype("<f4").tobytes())
        print(f"{name}: {table.shape} -> {path}")


if __name__ == "__main__":
    main()
