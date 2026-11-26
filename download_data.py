"""Download and verify the prepared CoReXNet data release from this repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
RELEASE_URL = "https://github.com/thienbaov51-bot/CoReXNet/releases/download/data-v1.0.0/CoReXNet_preprocessed_data_v1.zip"
# Immutable public data-v1.0.0 release, verified after anonymous download.
EXPECTED_ARCHIVE_SHA256 = "c5406d886a575bed7258c57137065190b72d20aadaffe333fc6116079cf80d3c"
DATASETS = ("Electricity", "PEMS03", "PEMS04", "PEMS08", "Solar", "AEMO5")
EXPECTED_MEMBERS = {f"{name}.npz" for name in DATASETS} | {
    f"{name}.metadata.json" for name in DATASETS
} | {"DATA_MANIFEST.json", "data_manifest.json", "README.txt", "SHA256SUMS"}
MAX_ARCHIVE_BYTES = 1 << 30
MAX_MEMBER_BYTES = 512 << 20
MAX_EXPANDED_BYTES = 2 << 30
BLOCK_SIZE = 1 << 20


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(name):
    # The fixed release has root-level files only. Reject both POSIX and Windows
    # absolute paths, drive names, alternate data streams, and path traversal.
    p = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or "\x00" in name
            or p.is_absolute() or ntpath.isabs(name) or ntpath.splitdrive(name)[0]
            or len(p.parts) != 1 or any(part in (".", "..") for part in p.parts)
            or name != p.name):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return name


def require_plain_output_path(path):
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink() or getattr(component, "is_junction", lambda: False)():
            raise ValueError(f"Output path contains a link or junction: {component}")
        if component.exists() and not component.is_dir():
            raise ValueError(f"Output parent is not a directory: {component}")
    return path


def download(temporary_path):
    request = urllib.request.Request(RELEASE_URL, headers={"User-Agent": "CoReXNet-data-downloader/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_ARCHIVE_BYTES:
                raise ValueError("Release archive exceeds the download size limit")
            count = 0
            with temporary_path.open("xb") as stream:
                for block in iter(lambda: response.read(BLOCK_SIZE), b""):
                    count += len(block)
                    if count > MAX_ARCHIVE_BYTES:
                        raise ValueError("Release archive exceeds the download size limit")
                    stream.write(block)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise RuntimeError("The data-v1.0.0 release is not available at the published URL (HTTP 404). No prepared data were installed.") from error
        raise RuntimeError(f"Could not download the data release (HTTP {error.code}). No prepared data were installed.") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not access the data release: {error.reason}. No prepared data were installed.") from error


def checksum_manifest(raw):
    if len(raw) > 128 * 1024:
        raise ValueError("SHA256SUMS is unexpectedly large")
    expected = {}
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+\*?(.+)", line)
        if not match:
            raise ValueError("Malformed SHA256SUMS entry")
        digest, name = match.groups()
        safe_filename(name)
        if name in expected:
            raise ValueError(f"Duplicate checksum entry: {name}")
        expected[name] = digest.lower()
    if set(expected) != EXPECTED_MEMBERS - {"SHA256SUMS"}:
        raise ValueError("SHA256SUMS does not cover exactly the 15 data/document members")
    return expected


def verify_and_stage(archive_path, stage):
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries) != len(EXPECTED_MEMBERS):
            raise ValueError("Unexpected number of archive members")
        seen = set()
        expanded_bytes = 0
        for info in entries:
            name = safe_filename(info.filename)
            if name in seen or name not in EXPECTED_MEMBERS:
                raise ValueError(f"Duplicate or unexpected archive member: {name}")
            seen.add(name)
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            kind = stat.S_IFMT(unix_mode)
            if info.is_dir() or kind == stat.S_IFLNK or kind not in (0, stat.S_IFREG):
                raise ValueError(f"Archive member is not a regular file: {name}")
            if info.flag_bits & 1:
                raise ValueError("Encrypted members are not supported")
            if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"Archive member exceeds the expanded-size limit: {name}")
            expanded_bytes += info.file_size
        if seen != EXPECTED_MEMBERS or expanded_bytes > MAX_EXPANDED_BYTES:
            raise ValueError("Archive content or expanded size differs from the fixed release")
        expected = checksum_manifest(archive.read("SHA256SUMS"))
        expected["SHA256SUMS"] = hashlib.sha256(archive.read("SHA256SUMS")).hexdigest()
        # Verify all member bytes before creating the final output directory.
        # Reading every member fully also triggers zipfile's CRC checks.
        hashes, case_aliases = {}, {}
        for info in entries:
            name = info.filename
            digest = hashlib.sha256()
            # DATA_MANIFEST.json and data_manifest.json are identical aliases.
            # A temporary indexed filename supports case-insensitive filesystems.
            temporary_file = stage / str(len(hashes))
            count = 0
            with archive.open(info, "r") as source, temporary_file.open("xb") as target:
                for block in iter(lambda: source.read(BLOCK_SIZE), b""):
                    count += len(block)
                    if count > info.file_size or count > MAX_MEMBER_BYTES:
                        raise ValueError(f"Unexpected expanded member size: {name}")
                    digest.update(block)
                    target.write(block)
            actual = digest.hexdigest()
            if count != info.file_size or actual != expected[name]:
                raise ValueError(f"SHA256 or byte-count mismatch: {name}")
            alias = name.casefold()
            if alias in case_aliases and case_aliases[alias] != actual:
                raise ValueError(f"Case-colliding archive names have different content: {name}")
            case_aliases[alias] = actual
            hashes[name] = {"sha256": actual, "bytes": count, "staged_file": temporary_file}
        return hashes


def install(verified, output):
    # Preflight the whole destination before writing any final file. A rerun
    # skips identical existing files and never replaces a different dataset.
    for name, record in verified.items():
        destination = output / safe_filename(name)
        if destination.is_symlink() or getattr(destination, "is_junction", lambda: False)():
            raise ValueError(f"Existing destination is a link: {destination}")
        if destination.exists():
            if (not destination.is_file() or destination.stat().st_size != record["bytes"]
                    or sha256(destination) != record["sha256"]):
                raise FileExistsError(f"Existing file differs from the release and will not be overwritten: {destination}")
    output.mkdir(parents=True, exist_ok=True)
    installed, skipped = [], []
    for name, record in verified.items():
        destination = output / name
        if destination.exists():
            if sha256(destination) != record["sha256"]:
                raise FileExistsError(f"Destination changed during installation: {destination}")
            skipped.append(name)
            continue
        # Exclusive creation also prevents overwriting a concurrent writer.
        with record["staged_file"].open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=BLOCK_SIZE)
        if sha256(destination) != record["sha256"]:
            raise ValueError(f"Installed-file checksum mismatch: {destination}")
        installed.append(name)
    return installed, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the release ZIP instead of downloading")
    parser.add_argument("--output", type=Path, default=HERE / "data" / "processed")
    parser.add_argument("--sha256", default=EXPECTED_ARCHIVE_SHA256,
                        help="Optional pinned SHA256 for the complete release ZIP")
    args = parser.parse_args()
    if args.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        parser.error("--sha256 must contain exactly 64 hexadecimal characters")
    try:
        output = require_plain_output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".corexnet_data_", dir=output.parent) as folder:
            temporary = Path(folder)
            if args.archive:
                archive_path = args.archive.expanduser().resolve(strict=True)
                if not archive_path.is_file() or archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Local archive is not a regular file within the download size limit")
            else:
                archive_path = temporary / "CoReXNet_preprocessed_data_v1.zip"
                download(archive_path)
            archive_hash = sha256(archive_path)
            if args.sha256 and archive_hash != args.sha256.lower():
                raise ValueError("Complete release archive SHA256 differs from the pinned value")
            staged = temporary / "members"
            staged.mkdir()
            verified = verify_and_stage(archive_path, staged)
            installed, skipped = install(verified, output)
            print(json.dumps({"status": "COMPLETE", "output": str(output),
                              "release_url": RELEASE_URL, "archive_sha256": archive_hash,
                              "verified_archive_members": len(verified),
                              "installed_members": installed, "skipped_identical_members": skipped,
                              "all_member_sha256": {name: row["sha256"] for name, row in verified.items()}}, indent=2))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Prepared-data installation failed: {error}\n")


if __name__ == "__main__":
    main()
