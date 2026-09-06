from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
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
        "registered_at": datetime.now(timezone.utc).isoformat(),
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
    entry["promoted_at"] = datetime.now(timezone.utc).isoformat()

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
        entry = promote_version(args.version, args.adapter_dir)
        print(json.dumps(entry, indent=2))
    elif args.command == "delete":
        ok = delete_version(args.version)
        print(f"Deleted: {ok}")
    else:
        print("No command specified. Use --help.", file=sys.stderr)


if __name__ == "__main__":
    main()
