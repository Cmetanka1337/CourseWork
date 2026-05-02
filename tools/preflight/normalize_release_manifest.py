#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

VERSION = "1.0"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_relative(path_value: str, root: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return path.name
    return str(path)


def get_git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def main() -> int:
    root = repo_root()
    manifest_path = root / "models/release_candidate/release_manifest.json"
    if not manifest_path.exists():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path_metadata = manifest.get("path_metadata", {})

    def normalize_field(field: str) -> None:
        value = manifest.get(field)
        if not value:
            return
        repo_relative = resolve_relative(value, root)
        path_metadata[field] = {
            "repo_relative_path": repo_relative,
            "local_absolute_path": value,
        }
        manifest[field] = repo_relative

    for field in [
        "selected_report",
        "selected_report_copy",
        "selected_model_path",
        "feature_passport_path",
    ]:
        normalize_field(field)

    copied = manifest.get("copied_artifacts", [])
    copied_meta = []
    normalized_copied = []
    for item in copied:
        repo_relative = resolve_relative(item, root)
        copied_meta.append(
            {"repo_relative_path": repo_relative, "local_absolute_path": item}
        )
        normalized_copied.append(repo_relative)
    if copied:
        manifest["copied_artifacts"] = normalized_copied
        path_metadata["copied_artifacts"] = copied_meta

    for entry in manifest.get("all_ranked_candidates", []):
        path_value = entry.get("path")
        if not path_value:
            continue
        repo_relative = resolve_relative(path_value, root)
        entry["path"] = repo_relative
        entry["repo_relative_path"] = repo_relative
        entry["local_absolute_path"] = path_value

    manifest["path_metadata"] = path_metadata
    manifest["git_commit"] = get_git_commit(root)
    manifest["repo_root"] = "."
    manifest["generated_by"] = f"tools/preflight/normalize_release_manifest.py@{VERSION}"

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Normalized manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

