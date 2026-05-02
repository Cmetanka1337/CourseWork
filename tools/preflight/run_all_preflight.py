#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_script(script_path: Path) -> None:
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Preflight step failed: {script_path.name}")


def main() -> int:
    root = repo_root()
    scripts = [
        root / "tools/preflight/print_release_candidate_summary.py",
        root / "tools/preflight/export_bundle_for_ios.py",
        root / "tools/preflight/generate_swift_calibrator_reference.py",
    ]

    try:
        for script in scripts:
            run_script(script)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Preflight complete.")
    print("Next steps:")
    print("- Validate CoreML export (Phase 1).")
    print("- Run Swift golden tests against artifacts/ios_bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

