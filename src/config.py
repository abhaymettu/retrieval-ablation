"""Single place for every knob. Nothing here is tuned on test data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"
OUTPUT = ROOT / "output"

DATASETS = ["scifact", "nfcorpus", "fiqa"]
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# scifact ships no dev qrels, so its official train split is used as the dev
# split for tuning. nfcorpus and fiqa use their official dev splits.
DEV_SPLIT = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}

# Two bi-encoders of deliberately different size and cost.
EMBED_MODELS = {
    "minilm": {
        "hf": "sentence-transformers/all-MiniLM-L6-v2",
        "params_m": 22.7,
        "dim": 384,
        "query_prefix": "",
        "doc_prefix": "",
        "max_seq": 256,
    },
    "bge": {
        "hf": "BAAI/bge-base-en-v1.5",
        "params_m": 109.5,
        "dim": 768,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "max_seq": 512,
    },
}

RERANKER = {"hf": "cross-encoder/ms-marco-MiniLM-L-6-v2", "params_m": 22.7, "max_seq": 512}

RETRIEVAL_DEPTH = 1000       # how deep every first stage run is stored
RERANK_DEPTH = 100           # cross encoder scores this many, k=10/50 are subsets
RERANK_KS = [10, 50, 100]
RRF_K = 60                   # standard constant from Cormack et al. 2009
BLEND_ALPHAS = [round(i / 20, 2) for i in range(21)]  # 0.00 .. 1.00 step 0.05

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260815
CI = 0.95

# Machine every timing in this repo was measured on. Stated because latency
# numbers are meaningless without it.
MACHINE = "Apple M4 Pro (12 core, 24 GB unified memory), macOS 15.5, torch 2.13 MPS backend"

# Cost model. The measured latency is local, the dollar rate is rented hardware,
# and mapping one onto the other is an assumption, not a measurement.
COST = {
    "instance": "AWS g5.xlarge on demand, us-east-1",
    "usd_per_hour": 1.006,
    "note": "serial serving, one query at a time, no batching, no caching, "
            "100 percent utilisation. Latency assumed to transfer from the "
            "local machine unchanged, which is the weakest link in this number.",
}
