from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VERSIONS_FILE = "models/versions.json"


def load_versions(versions_path: Path) -> dict[str, Any]:
    if versions_path.exists():
        return json.loads(versions_path.read_text(encoding="utf-8"))
    return {"current": None, "versions": {}}


def save_versions(versions_path: Path, data: dict[str, Any]) -> None:
    versions_path.parent.mkdir(parents=True, exist_ok=True)
    versions_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def register_version(
    version: str,
    adapter_dir: Path,
    description: str = "",
    base_version: str | None = None,
    source_files: list[str] | None = None,
    versions_path: Path | None = None,
) -> dict[str, Any]:
    """Register a new model version."""
    vp = versions_path or Path(VERSIONS_FILE)
    data = load_versions(vp)

    if version in data["versions"]:
        raise ValueError(f"Version {version} already exists. Use a new version number.")

    entry: dict[str, Any] = {
        "version": version,
        "registered_at": datetime.now(UTC).isoformat(),
        "adapter_dir": str(adapter_dir),
        "description": description,
        "base_version": base_version,
        "source_files": source_files or [],
        "status": "registered",
    }

    data["versions"][version] = entry
    data["current"] = version
    save_versions(vp, data)
    return entry


def set_current_version(version: str, versions_path: Path | None = None) -> None:
    """Switch the active model version."""
    vp = versions_path or Path(VERSIONS_FILE)
    data = load_versions(vp)
    if version not in data["versions"]:
        raise ValueError(f"Version {version} not found. Available: {list(data['versions'])}")
    data["current"] = version
    save_versions(vp, data)


def get_current_version(versions_path: Path | None = None) -> str | None:
    vp = versions_path or Path(VERSIONS_FILE)
    data = load_versions(vp)
    return data.get("current")


def list_versions(versions_path: Path | None = None) -> dict[str, Any]:
    vp = versions_path or Path(VERSIONS_FILE)
    return load_versions(vp)


def get_version_info(version: str, versions_path: Path | None = None) -> dict[str, Any] | None:
    vp = versions_path or Path(VERSIONS_FILE)
    data = load_versions(vp)
    return data["versions"].get(version)


def update_version_artifacts(
    version: str,
    versions_path: Path | None = None,
    *,
    merged_dir: Path | None = None,
    gguf_dir: Path | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    """Update artifact paths on an already-registered version.

    Returns the updated entry, or None if the version is not registered.
    """
    vp = versions_path or Path(VERSIONS_FILE)
    data = load_versions(vp)
    entry = data["versions"].get(version)
    if entry is None:
        return None

    if merged_dir is not None:
        entry["merged_dir"] = str(merged_dir)
    if gguf_dir is not None:
        entry["gguf_dir"] = str(gguf_dir)
    if status is not None:
        entry["status"] = status
    save_versions(vp, data)
    return entry


def get_current_gguf(versions_path: Path | None = None) -> Path | None:
    """Return the GGUF file of the current model version, if one is registered.

    Prefers quantized builds (q4_k_m) over larger ones so the served model is
    the fast one. An unreadable registry or missing files fall back to None so
    callers can use their own candidate list.
    """
    vp = versions_path or Path(VERSIONS_FILE)
    try:
        data = load_versions(vp)
        current = data.get("current")
        if not current:
            return None
        entry = data["versions"].get(current) or {}
        gguf_dir = entry.get("gguf_dir")
        if not gguf_dir:
            return None
        path = Path(gguf_dir)
        if path.is_file():
            return path
        if path.is_dir():
            ggufs = sorted(path.glob("*.gguf"))
            if not ggufs:
                return None
            quantized = [candidate for candidate in ggufs if "q4_k_m" in candidate.name.lower()]
            return (quantized or ggufs)[0]
    except (OSError, ValueError, KeyError):
        return None
    return None


def delete_version(version: str, versions_path: Path | None = None) -> bool:
    """Remove a version from the registry (does not delete files)."""
    vp = versions_path or Path(VERSIONS_FILE)
    data = load_versions(vp)
    if version not in data["versions"]:
        return False
    del data["versions"][version]
    if data["current"] == version:
        remaining = list(data["versions"].keys())
        data["current"] = remaining[-1] if remaining else None
    save_versions(vp, data)
    return True


def promote_version(
    version: str,
    adapter_dir: Path,
    merged_dir: Path | None = None,
    gguf_dir: Path | None = None,
    versions_path: Path | None = None,
) -> dict[str, Any]:
    """Mark a version as the production model and copy artifacts."""
    vp = versions_path or Path(VERSIONS_FILE)
    data = load_versions(vp)

    if version not in data["versions"]:
        raise ValueError(f"Version {version} not found.")

    entry = data["versions"][version]
    entry["status"] = "promoted"
    entry["promoted_at"] = datetime.now(UTC).isoformat()

    if merged_dir and adapter_dir.exists():
        dest = merged_dir / f"tmc-lm-v{version}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(adapter_dir, dest)
        entry["merged_dir"] = str(dest)

    if gguf_dir:
        entry["gguf_dir"] = str(gguf_dir)

    data["current"] = version
    save_versions(vp, data)
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TMC-LM model version manager.")
    sub = parser.add_subparsers(dest="command")

    reg = sub.add_parser("register", help="Register a new version")
    reg.add_argument("--version", required=True)
    reg.add_argument("--adapter-dir", type=Path, required=True)
    reg.add_argument("--description", default="")
    reg.add_argument("--base-version", default=None)

    sub.add_parser("list", help="List all versions")
    sub.add_parser("current", help="Show current version")

    cur = sub.add_parser("set-current", help="Set active version")
    cur.add_argument("--version", required=True)

    prom = sub.add_parser("promote", help="Promote version to production")
    prom.add_argument("--version", required=True)
    prom.add_argument("--adapter-dir", type=Path, required=True)
    prom.add_argument("--merged-dir", type=Path, default=None, help="Directory to copy the merged model into.")
    prom.add_argument("--gguf-dir", type=Path, default=None, help="GGUF file or directory to register for serving.")

    gguf_cmd = sub.add_parser("set-gguf", help="Record the GGUF artifact for an already-registered version")
    gguf_cmd.add_argument("--version", required=True)
    gguf_cmd.add_argument("--gguf-dir", type=Path, required=True)

    merged_cmd = sub.add_parser(
        "set-merged", help="Record the merged-model directory for an already-registered version"
    )
    merged_cmd.add_argument("--version", required=True)
    merged_cmd.add_argument("--merged-dir", type=Path, required=True)

    dele = sub.add_parser("delete", help="Remove a version from registry")
    dele.add_argument("--version", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "register":
        entry = register_version(args.version, args.adapter_dir, args.description, args.base_version)
        print(json.dumps(entry, indent=2))
    elif args.command == "list":
        data = list_versions()
        print(json.dumps(data, indent=2))
    elif args.command == "current":
        v = get_current_version()
        print(f"Current version: {v}")
    elif args.command == "set-current":
        set_current_version(args.version)
        print(f"Switched to version {args.version}")
    elif args.command == "promote":
        entry = promote_version(args.version, args.adapter_dir, merged_dir=args.merged_dir, gguf_dir=args.gguf_dir)
        print(json.dumps(entry, indent=2))
    elif args.command == "set-gguf":
        entry = update_version_artifacts(args.version, gguf_dir=args.gguf_dir, status="promoted")
        if entry is None:
            print(f"Version {args.version} not found. Register it first with 'register'.", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(entry, indent=2))
    elif args.command == "set-merged":
        entry = update_version_artifacts(args.version, merged_dir=args.merged_dir)
        if entry is None:
            print(f"Version {args.version} not found. Register it first with 'register'.", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(entry, indent=2))
    elif args.command == "delete":
        ok = delete_version(args.version)
        print(f"Deleted: {ok}")
    else:
        print("No command specified. Use --help.", file=sys.stderr)


if __name__ == "__main__":
    main()
