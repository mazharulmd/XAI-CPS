from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request
import zipfile


URL = "https://zenodo.org/records/22953576/files/data4cyber_dataset.zip?download=1"
ARCHIVE_MD5 = "20d81c975e5c37f453103cdadb052c9e"
ARCHIVE_SIZE = 164_148_991
MAX_UNCOMPRESSED_BYTES = 1_000_000_000


def safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    if ":" in path.parts[0] or "\\" in name:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def range_request(start: int, end: int, timeout: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                URL,
                headers={"User-Agent": "XAI-CPS-research/0.1", "Range": f"bytes={start}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected_range = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
                actual_range = response.headers.get("Content-Range")
                if response.status != 206 or actual_range != expected_range:
                    raise ValueError(f"Unexpected range response: status={response.status}, Content-Range={actual_range!r}")
                data = response.read(end - start + 1)
            if len(data) != end - start + 1:
                raise ValueError(f"Short range response for bytes {start}-{end}: got {len(data)} bytes")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    assert last_error is not None
    raise last_error


def download_archive(archive: Path, timeout: int = 120, chunk_size: int = 1_000_000, workers: int = 1) -> None:
    if archive.exists():
        size = archive.stat().st_size
        actual_md5 = md5_file(archive)
        if size == ARCHIVE_SIZE and actual_md5 == ARCHIVE_MD5:
            return
        raise FileExistsError(f"Existing archive is not the verified v3 release; preserving it: {archive}")
    temporary = archive.with_name(archive.name + ".part")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        size = temporary.stat().st_size
        if size > ARCHIVE_SIZE:
            raise ValueError(f"Partial archive is larger than expected; inspect it manually: {temporary}")
        # Check both ends before resuming so an unrelated .part file is not appended to.
        if size:
            first_end = min(63, size - 1)
            with temporary.open("rb") as stream:
                prefix = stream.read(first_end + 1)
            if prefix != range_request(0, first_end, timeout):
                raise ValueError(f"Partial archive prefix does not match Zenodo: {temporary}")
            if size > 64:
                with temporary.open("rb") as stream:
                    stream.seek(size - 64)
                    tail = stream.read(64)
                if tail != range_request(size - 64, size - 1, timeout):
                    raise ValueError(f"Partial archive tail does not match Zenodo: {temporary}")
    else:
        temporary.touch(exist_ok=False)

    offset = temporary.stat().st_size
    ranges = [(start, min(ARCHIVE_SIZE - 1, start + chunk_size - 1))
              for start in range(offset, ARCHIVE_SIZE, chunk_size)]
    next_offset = offset
    completed = 0
    with temporary.open("ab") as output:
        if workers <= 1:
            for start, end in ranges:
                block = range_request(start, end, timeout)
                output.write(block)
                output.flush()
                next_offset += len(block)
                completed += 1
                print(f"Downloaded {next_offset:,}/{ARCHIVE_SIZE:,} bytes ({completed}/{len(ranges)} chunks)", flush=True)
        else:
            pending: dict[int, bytes] = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(range_request, start, end, timeout): start for start, end in ranges}
                for future in as_completed(futures):
                    start = futures[future]
                    pending[start] = future.result()
                    while next_offset in pending:
                        block = pending.pop(next_offset)
                        output.write(block)
                        output.flush()
                        next_offset += len(block)
                        completed += 1
                        if completed % 5 == 0 or next_offset == ARCHIVE_SIZE:
                            print(f"Downloaded {next_offset:,}/{ARCHIVE_SIZE:,} bytes ({completed}/{len(ranges)} chunks)", flush=True)
    size = temporary.stat().st_size
    if size != ARCHIVE_SIZE:
        raise ValueError(f"Unexpected archive size: {size} bytes (expected {ARCHIVE_SIZE})")
    actual_md5 = md5_file(temporary)
    if actual_md5 != ARCHIVE_MD5:
        raise ValueError(f"Zenodo MD5 mismatch: {actual_md5}")
    temporary.replace(archive)


def extract_verified(archive: Path, destination: Path) -> list[Path]:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing extraction: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    extracted: list[Path] = []
    total = 0
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("Archive contains duplicate member paths")
            for info in infos:
                member = safe_member_path(info.filename)
                total += info.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("Archive exceeds the configured extraction size limit")
                target = destination.joinpath(*member.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                # Reject Unix symlinks and other non-regular member types.
                mode = (info.external_attr >> 16) & 0o170000
                if mode not in {0, 0o100000}:
                    raise ValueError(f"Unsupported non-regular archive member: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
                extracted.append(target)

        manifest_path = destination / "data4cyber_dataset" / "SHA256SUMS.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest.items():
            member = safe_member_path(relative)
            target = destination / "data4cyber_dataset" / Path(*member.parts)
            if not target.is_file():
                raise ValueError(f"SHA-256 manifest member is missing: {relative}")
            actual = sha256_file(target)
            if actual.lower() != str(expected).lower():
                raise ValueError(f"SHA-256 mismatch for {relative}: {actual}")
        return extracted
    except Exception:
        # Leave extracted files in place so a failed verification remains inspectable.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify Data4Cyber v3 from Zenodo")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/data4cyber"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "data4cyber_dataset.zip"
    extraction = args.output_dir / "data4cyber_dataset"
    download_archive(archive, timeout=args.timeout, chunk_size=args.chunk_size, workers=args.workers)
    files = extract_verified(archive, extraction)
    print(f"Verified {len(files)} files under {extraction}")
    print("Dataset license: CC BY 4.0; cite Zenodo DOI 10.5281/zenodo.22953576.")


if __name__ == "__main__":
    main()
