"""Download official UCI SECOM raw files with SHA256 verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

from shap_diff_analysis.secom import (
    DEFAULT_RAW_DIR,
    SECOM_FILE_SPECS,
    SecoMDataError,
    sha256_file,
    source_manifest,
    verify_secom_files,
)


def download_file(url: str, destination: Path) -> None:
    """Download one file from ``url`` to ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, destination)
    except URLError as exc:
        msg = f"Failed to download {url}: {exc}"
        raise SecoMDataError(msg) from exc


def download_secom(raw_dir: Path, force: bool = False) -> dict[str, str]:
    """Download and verify all official SECOM files."""
    digests: dict[str, str] = {}
    for filename, spec in SECOM_FILE_SPECS.items():
        destination = raw_dir / filename
        if destination.is_file() and not force:
            digest = sha256_file(destination)
            if digest != spec["sha256"]:
                msg = (
                    f"Existing file {destination} has unexpected SHA256 {digest}; "
                    f"expected {spec['sha256']}. Re-run with --force."
                )
                raise SecoMDataError(msg)
        else:
            download_file(spec["url"], destination)
            digest = sha256_file(destination)
            if digest != spec["sha256"]:
                msg = f"Downloaded file {destination.name} failed SHA256 verification: {digest}"
                raise SecoMDataError(msg)
        digests[filename] = digest

    verify_secom_files(raw_dir)
    return digests


def write_download_manifest(raw_dir: Path, digests: dict[str, str]) -> Path:
    """Write download provenance metadata next to the raw files."""
    manifest_path = raw_dir / "download_manifest.json"
    payload = {
        "source": source_manifest().to_dict(),
        "raw_dir": str(raw_dir),
        "sha256": digests,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the SECOM download script."""
    parser = argparse.ArgumentParser(description="Download official UCI SECOM raw files.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for secom.data / secom_labels.data / secom.names",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Download SECOM raw files and write a provenance manifest."""
    args = parse_args()
    digests = download_secom(args.raw_dir, force=args.force)
    manifest_path = write_download_manifest(args.raw_dir, digests)
    print(f"Downloaded and verified SECOM files in {args.raw_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
