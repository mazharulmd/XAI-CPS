"""Download official HAI 21.03 CSV.gz files into a local, git-ignored directory."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

BASE = "https://raw.githubusercontent.com/icsdataset/hai/master/hai-21.03"
FILES = [*(f"train{i}.csv.gz" for i in range(1, 4)), *(f"test{i}.csv.gz" for i in range(1, 6))]

def valid_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        return True
    except (OSError, EOFError):
        return False

def fetch(name: str, root: Path) -> dict:
    target = root / name
    if target.exists() and not valid_gzip(target):
        target.unlink()
    if not target.exists():
        temp = target.with_suffix(target.suffix + ".part")
        subprocess.run(["curl.exe", "--fail", "--location", "--retry", "3", "--output", str(temp), f"{BASE}/{name}"], check=True)
        temp.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": name, "bytes": target.stat().st_size, "sha256": digest.hexdigest(), "source": f"{BASE}/{name}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/hai-21.03")
    args = parser.parse_args()
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda name: fetch(name, root), FILES))
    manifest = {"dataset": "HAI 21.03", "source": "https://github.com/icsdataset/hai/tree/master/hai-21.03", "files": records}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for row in records:
        print(f"{row['file']} bytes={row['bytes']} sha256={row['sha256']}")
    print(f"manifest={root / 'manifest.json'}")

if __name__ == "__main__":
    main()
