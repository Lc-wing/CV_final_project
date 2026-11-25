"""Download and SHA-256 verify trained Anti-UAV checkpoints from a JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming chunks
PLACEHOLDER_PREFIXES: tuple[str, ...] = ("tbd", "placeholder", "todo")
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https", "file"})
GDRIVE_FILE_ID_RE = re.compile(
    r"(?:/file/d/|/d/|[?&]id=)([A-Za-z0-9_-]{10,})"
)


def compute_sha256(path: Path | str) -> str:
    """Compute the lowercase hexadecimal SHA-256 digest of ``path`` using streaming reads."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 hashing: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path | str, expected_hex: str) -> bool:
    """Verify ``path`` against ``expected_hex`` (case-insensitive streaming SHA-256 check)."""
    if not isinstance(expected_hex, str):
        return False
    expected_clean = expected_hex.strip().lower()
    if len(expected_clean) != 64 or any(ch not in "0123456789abcdef" for ch in expected_clean):
        return False
    actual = compute_sha256(path)
    return hmac.compare_digest(actual, expected_clean)


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    return any(lowered.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def load_manifest(manifest_path: Path | str) -> list[dict[str, Any]]:
    """Load and validate checkpoint entries from ``checkpoints_manifest.json``."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        entries = raw.get("checkpoints")
    elif isinstance(raw, list):
        entries = raw
    else:
        entries = None

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest '{path.name}' must contain a non-empty 'checkpoints' list")

    validated: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry #{idx} must be a JSON object")
        for key in ("run_name", "filename", "url", "sha256"):
            val = entry.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"Manifest entry #{idx} missing non-empty string field '{key}'"
                )
        run_name = entry["run_name"].strip()
        if run_name in seen_runs:
            raise ValueError(f"Duplicate run_name {run_name!r} in manifest '{path.name}'")
        seen_runs.add(run_name)
        filename = entry["filename"].strip()
        if (
            Path(filename).name != filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
        ):
            raise ValueError(
                f"Manifest entry '{run_name}' filename must be a plain filename without "
                f"directory components, got {filename!r}"
            )
        validated.append(
            {
                "run_name": run_name,
                "filename": filename,
                "url": entry["url"].strip(),
                "sha256": entry["sha256"].strip(),
                "hosting_note": str(entry.get("hosting_note", "")).strip(),
            }
        )
    return validated


def _normalize_download_url(url: str) -> tuple[str, bool]:
    """Return ``(resolved_url, is_gdrive)`` normalizing Google Drive share links."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if "drive.google.com" in host or "drive.usercontent.google.com" in host:
        match = GDRIVE_FILE_ID_RE.search(url)
        if match:
            file_id = match.group(1)
            return (
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
                True,
            )
        return url, True
    return url, False


def _download_to_path(url: str, dest_part: Path) -> None:
    """Stream-download ``url`` to ``dest_part`` (supports HTTP/HTTPS, file://, and Google Drive)."""
    resolved_url, is_gdrive = _normalize_download_url(url)

    if is_gdrive:
        try:
            import gdown  # type: ignore[import-not-found]
        except ImportError:
            gdown = None
        if gdown is not None:
            result = gdown.download(url, str(dest_part), quiet=False, fuzzy=True)
            if result is None or not dest_part.is_file():
                raise RuntimeError(f"gdown failed to download Google Drive URL: {url}")
            return

    req = urllib.request.Request(
        resolved_url,
        headers={"User-Agent": "cs460-anti-uav-checkpoint-downloader/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as response, dest_part.open("wb") as out_fh:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            out_fh.write(chunk)


def download_checkpoints(
    manifest_path: Path | str,
    output_dir: Path | str,
    run_names: Sequence[str] | None = None,
    verify_only: bool = False,
) -> dict[str, Path]:
    """Download and verify checkpoints listed in ``manifest_path`` into ``output_dir``.

    Rejects placeholder URLs or checksums cleanly with an actionable error message.
    Verifies every downloaded (or cached) checkpoint via streaming SHA-256 before
    committing it to ``output_dir``.
    """
    entries = load_manifest(manifest_path)

    if run_names is not None and len(run_names) > 0:
        requested = [r.strip() for r in run_names]
        available = {e["run_name"] for e in entries}
        unknown = [r for r in requested if r not in available]
        if unknown:
            raise ValueError(
                f"Requested unknown run_name(s) {unknown}; manifest defines {sorted(available)}"
            )
        entries = [e for e in entries if e["run_name"] in set(requested)]

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    for entry in entries:
        run_name = entry["run_name"]
        filename = Path(entry["filename"]).name
        url = entry["url"]
        expected_sha = entry["sha256"]
        dest_path = out_root / filename

        if _is_placeholder(expected_sha):
            raise ValueError(
                f"Checkpoint '{run_name}' has placeholder sha256 ({expected_sha!r}). "
                "Populate checkpoints_manifest.json with real URLs and SHA-256 hashes "
                "after Phase 3 GPU training (hosted in Google Drive MyDrive/CV_project/checkpoints/ "
                "or GitHub Releases)."
            )

        if verify_only:
            if not dest_path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint '{run_name}' not found for --verify-only: {dest_path}"
                )
            if not verify_sha256(dest_path, expected_sha):
                actual = compute_sha256(dest_path)
                raise ValueError(
                    f"SHA-256 mismatch for existing checkpoint '{filename}': "
                    f"expected {expected_sha.lower()}, got {actual}"
                )
            results[run_name] = dest_path
            continue

        if _is_placeholder(url):
            raise ValueError(
                f"Checkpoint '{run_name}' has placeholder URL ({url!r}). "
                "Populate checkpoints_manifest.json with real download URLs and SHA-256 hashes "
                "after Phase 3 GPU training (hosted in Google Drive MyDrive/CV_project/checkpoints/ "
                "or GitHub Releases)."
            )

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            raise ValueError(
                f"Unsupported URL scheme {parsed.scheme!r} for checkpoint '{run_name}': {url!r}. "
                f"Expected one of {sorted(ALLOWED_SCHEMES)}."
            )

        if dest_path.is_file() and verify_sha256(dest_path, expected_sha):
            results[run_name] = dest_path
            continue

        part_path = out_root / f"{filename}.part"
        try:
            _download_to_path(url, part_path)
            if not verify_sha256(part_path, expected_sha):
                actual = compute_sha256(part_path)
                raise ValueError(
                    f"SHA-256 mismatch after downloading '{filename}' ({run_name}): "
                    f"expected {expected_sha.lower()}, got {actual}"
                )
            os.replace(part_path, dest_path)
        except Exception:
            part_path.unlink(missing_ok=True)
            raise

        results[run_name] = dest_path

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m scripts.download_checkpoints``."""
    parser = argparse.ArgumentParser(
        description="Download and SHA-256 verify trained Anti-UAV model checkpoints from a manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("checkpoints_manifest.json"),
        help="Path to checkpoints_manifest.json (default: checkpoints_manifest.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Output directory for downloaded .pt checkpoints (default: checkpoints).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional subset of run_names to download (default: all manifest entries).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify SHA-256 digests of already-present checkpoints without downloading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for checkpoint manifest download and SHA-256 verification."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    downloaded = download_checkpoints(
        manifest_path=args.manifest,
        output_dir=args.out_dir,
        run_names=args.runs,
        verify_only=args.verify_only,
    )
    for run_name, ckpt_path in downloaded.items():
        print(f"Verified checkpoint | run={run_name} -> {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
