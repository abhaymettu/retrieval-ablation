PY := .venv/bin/python
RUN := PYTHONPATH=src $(PY)

.PHONY: all analysis venv download audit embed retrieve fusion rerank latency \
        evaluate figures errors report test clean distclean

# Everything, from nothing. The expensive steps skip themselves if their cache
# is already on disk.
all: download audit embed retrieve fusion rerank latency analysis

# Every number in the README, offline, from cached embeddings, runs and timings.
analysis: evaluate figures errors report test

venv:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -r requirements.txt

download:  ; $(RUN) src/00_download.py
audit:     ; $(RUN) src/01_audit.py
embed:     ; $(RUN) src/02_embed.py
retrieve:  ; $(RUN) src/03_retrieve.py
fusion:    ; $(RUN) src/04_fusion.py
rerank:    ; $(RUN) src/05_rerank.py
latency:   ; $(RUN) src/06_latency.py
evaluate:  ; $(RUN) src/07_evaluate.py
figures:   ; $(RUN) src/08_figures.py
errors:    ; $(RUN) src/09_error_analysis.py
report:    ; $(RUN) src/10_report.py
test:      ; $(PY) tests.py

# Drop generated results but keep the expensive caches.
clean:
	rm -rf output runs

# Drop everything downloaded and computed. The next make all starts cold.
distclean: clean
	rm -rf data cache
