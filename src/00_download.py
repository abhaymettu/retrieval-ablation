"""Fetch the official BEIR zips. Skips anything already on disk."""
import io
import sys
import zipfile

import requests

from config import BEIR_URL, DATA, DATASETS


def fetch(name):
    target = DATA / name
    if (target / "corpus.jsonl").exists():
        print(f"{name}: already present")
        return
    url = BEIR_URL.format(name=name)
    print(f"{name}: downloading {url}")
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    DATA.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA)
    assert (target / "corpus.jsonl").exists(), f"{name}: unexpected archive layout"
    print(f"{name}: extracted to {target}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or DATASETS:
        fetch(ds)
