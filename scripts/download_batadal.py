"""Fetch the official BATADAL normal and partially labeled training CSVs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE="https://www.batadal.net/data"
FILES=("BATADAL_dataset03.csv","BATADAL_dataset04.csv")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="data/raw/batadal")
    args=parser.parse_args()
    root=Path(args.data_dir);root.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in FILES:
        target=root/name
        if not target.exists():
            temp=target.with_suffix(target.suffix+".part")
            subprocess.run(["curl.exe","--fail","--location","--retry","3","--output",str(temp),f"{BASE}/{name}"],check=True)
            temp.replace(target)
        digest=hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
        records.append({"file":name,"bytes":target.stat().st_size,"sha256":digest.hexdigest(),"source":f"{BASE}/{name}"})
    manifest={"dataset":"BATADAL","dataset_page":"https://www.batadal.net/data.html",
              "citation":"Taormina et al., The Battle of the Attack Detection Algorithms: Disclosing Cyber Attacks on Water Distribution Networks, Journal of Water Resources Planning and Management, 144(8), 2018, doi:10.1061/(ASCE)WR.1943-5452.0000969.",
              "labels":"Dataset03 is normal-only; Dataset04 contains ATT_FLAG=1 approximate attack labels and ATT_FLAG=-999 unknown rows.",
              "files":records}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    for row in records: print(f"{row['file']} bytes={row['bytes']} sha256={row['sha256']}")
    print(f"manifest={root/'manifest.json'}")


if __name__=="__main__": main()
