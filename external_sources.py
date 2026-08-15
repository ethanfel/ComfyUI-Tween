"""Pinned, lazy installers for optional upstream model runtimes.

Tween does not redistribute these projects.  Their official source archives are
downloaded only when a corresponding loader node is executed, verified against
a pinned SHA-256 digest, and kept next to that model's checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile


logger = logging.getLogger("Tween")


UPSTREAM_SOURCES = {
    "speed": {
        "project": "SPEED",
        "commit": "40fadbe85c88cc6e4015062389da464fd7e85ab9",
        "url": (
            "https://codeload.github.com/bbldCVer/SPEED/zip/"
            "40fadbe85c88cc6e4015062389da464fd7e85ab9"
        ),
        "sha256": "9e9cc71bfeaf7a62008950b8f234f5f035df27b65a5fc0464caee2542f47f68c",
        "required": "src/models/model.py",
    },
    "ldf": {
        "project": "LDF-VFI",
        "commit": "61b34d2379df8a313e8e4cb467cc2f74c52b45d7",
        "url": (
            "https://codeload.github.com/xypeng9903/LDF-VFI/zip/"
            "61b34d2379df8a313e8e4cb467cc2f74c52b45d7"
        ),
        "sha256": "3a903aeb5353c7e5eb932f129d975d1750246502937d8f7b283b61a269c23668",
        "required": "training/models/precond.py",
    },
}
_SOURCE_LOCKS = {name: threading.Lock() for name in UPSTREAM_SOURCES}


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Tween"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Could not download optional upstream runtime from {url}. "
            "Check network access and retry the loader node."
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as source_zip:
        members = source_zip.infolist()
        if not members:
            raise RuntimeError(f"Downloaded source archive is empty: {archive}")

        destination_resolved = destination.resolve()
        for member in members:
            member_path = (destination / member.filename).resolve()
            if os.path.commonpath((destination_resolved, member_path)) != str(destination_resolved):
                raise RuntimeError(f"Unsafe path in source archive: {member.filename}")
        source_zip.extractall(destination)

    top_level = {Path(member.filename).parts[0] for member in members if member.filename}
    if len(top_level) != 1:
        raise RuntimeError("Expected one top-level directory in the upstream source archive")
    return destination / top_level.pop()


def _ensure_upstream_source_unlocked(name: str, model_dir: str | os.PathLike[str]) -> str:
    try:
        spec = UPSTREAM_SOURCES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Tween upstream source: {name}") from exc

    model_root = Path(model_dir)
    source_dir = model_root / "_upstream"
    required_file = source_dir / spec["required"]
    if required_file.is_file():
        marker_path = source_dir / ".tween-source.json"
        if marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid source marker: {marker_path}") from exc
            if (
                marker.get("commit") != spec["commit"]
                or marker.get("archive_sha256") != spec["sha256"]
            ):
                raise RuntimeError(
                    f"{spec['project']} runtime at {source_dir} is pinned to a different commit. "
                    "Remove _upstream and run the loader again."
                )
        else:
            logger.warning(
                "Using manually installed %s runtime at %s (no Tween verification marker)",
                spec["project"], source_dir,
            )
        return str(source_dir)

    if source_dir.exists():
        raise RuntimeError(
            f"Incomplete {spec['project']} runtime at {source_dir}. "
            "Remove that _upstream directory and run the loader again."
        )

    model_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading pinned %s runtime (%s) to %s",
        spec["project"], spec["commit"][:12], source_dir,
    )

    with tempfile.TemporaryDirectory(prefix="tween-source-", dir=model_root) as temp_name:
        temp_dir = Path(temp_name)
        archive = temp_dir / "source.zip"
        _download(spec["url"], archive)

        actual_digest = _sha256(archive)
        if actual_digest != spec["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for {spec['project']} source archive: "
                f"expected {spec['sha256']}, got {actual_digest}"
            )

        extracted = _safe_extract(archive, temp_dir / "extract")
        if not (extracted / spec["required"]).is_file():
            raise RuntimeError(
                f"The {spec['project']} archive does not contain {spec['required']}"
            )

        marker = {
            "project": spec["project"],
            "commit": spec["commit"],
            "archive_sha256": spec["sha256"],
            "source_url": spec["url"],
        }
        (extracted / ".tween-source.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(extracted), str(source_dir))

    logger.info("Installed %s runtime at %s", spec["project"], source_dir)
    return str(source_dir)


def ensure_upstream_source(name: str, model_dir: str | os.PathLike[str]) -> str:
    """Return a verified upstream checkout, downloading it once per process."""
    try:
        source_lock = _SOURCE_LOCKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Tween upstream source: {name}") from exc
    with source_lock:
        return _ensure_upstream_source_unlocked(name, model_dir)
