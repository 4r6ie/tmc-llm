from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


class GgufCheckError(RuntimeError):
    pass


def inspect_gguf(path: Path) -> dict:
    if not path.exists():
        raise GgufCheckError(f"GGUF file does not exist: {path}")

    if path.stat().st_size < 16:
        raise GgufCheckError(f"GGUF file is too small: {path}")

    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"GGUF":
            raise GgufCheckError(f"Invalid GGUF magic bytes: expected GGUF, got {magic!r}")

        version_bytes = handle.read(4)
        version = struct.unpack("<I", version_bytes)[0]

    return {
        "path": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "magic": "GGUF",
        "version": version,
        "valid": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a GGUF file header.")
    parser.add_argument("--path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        info = inspect_gguf(args.path)
    except GgufCheckError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

