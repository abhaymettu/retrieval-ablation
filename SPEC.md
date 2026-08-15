# retrieval-ablation

**Build this repo. This file is the whole brief. Read it fully, then start.**

## House rules (apply to every step)

- No em dashes anywhere, in code comments, README, or commit messages.
- Real benchmark data with real relevance labels for the headline result.
- The honest result is the deliverable. If the expensive reranker buys two
  nDCG points for four times the latency, the README leads with that trade.
- Every number in the README reproducible by one command, offline, from cached
  embeddings and runs.
- Python, pinned deps, `make all` regenerates everything.
- No Claude/Co-Authored-By trailers in commits.

## One-line pitch

What each layer of a retrieval stack is actually worth: BM25 versus dense
versus hybrid versus cross encoder reranking on BEIR, scored on quality,
latency and dollars, with the failure cases characterized.

## Why this repo exists

Nearly every RAG repo builds one pipeline and declares it works. Nobody
measures what each component contributes or what it costs. This repo produces
the Pareto frontier an engineer would actually use to choose a stack, plus an
error analysis that says which queries the stack fails on and why.

## Data

- **BEIR** subsets, chosen small enough to run on a laptop but varied in
  character. Recommended: `scifact` (small, scientific claims), `nfcorpus`
  (medical, hard), `fiqa` (financial QA, long tail), and `trec-covid` if
  runtime allows. Use the official BEIR loaders and the official qrels.
- Report corpus size, query count and label density per dataset. Do not mix
  datasets into one aggregate number without also reporting per dataset.

## Systems to ablate

1. **BM25** (`rank_bm25`, or Pyserini/Lucene if you want the faster and more
   standard implementation). This is the baseline that must be beaten.
2. **Dense bi-encoder** retrieval. At least two embedding models of different
   size and cost, for example a small open model and a larger open model. If
   you include a hosted embedding API, cache the vectors.
3. **Hybrid**: score fusion. Implement both reciprocal rank fusion and a
   normalized score blend with a swept weight. Report the weight sensitivity,
   because a hybrid tuned on test is a lie.
4. **Cross encoder reranking** over the top k from the best first stage, with k
   swept over 10, 50, 100. At least one reranker model.
5. **Optional:** query expansion or HyDE style rewriting. Only if the core
   ablation is complete, and only measured, never assumed to help.

## Evaluation protocol

- **Metrics:** nDCG@10 as the headline (BEIR standard), plus Recall@100,
  MRR@10 and Success@1. Recall@100 matters because it caps what a reranker can
  possibly recover, and that framing is the repo's best insight.
- **Statistics:** paired bootstrap over queries with confidence intervals for
  every system comparison. No bare point estimates. State whether a difference
  is distinguishable from noise.
- **Cost and latency:** measured, not estimated. Per query p50 and p95 latency
  for each stage, index build time, index size on disk, and dollars per 1000
  queries with the assumptions written out. Run on a single stated machine.
- **The Pareto plot** of nDCG@10 versus p95 latency, with dollars as marker
  size, is the hero figure.
- **Error analysis.** Take the queries where the best system fails and the ones
  where BM25 beats dense. Categorize them by hand into 4 to 6 failure modes
  (exact term match, acronym, numeric, multi hop, vocabulary mismatch, label
  noise) with examples in the README. This is the section that gets you the
  interview, because almost nobody does it.
- **Tuning hygiene:** any hyperparameter (fusion weight, k, chunk size) is
  selected on a dev split and reported on test. Say explicitly which split each
  number comes from.

## Deliverables

- `README.md` whose first two sentences carry the verdict. Example shape:
  "On scifact, hybrid retrieval plus a cross encoder reaches nDCG@10 of X, up
  N points on BM25, at 4.2x the p95 latency and $Y per 1000 queries. On
  nfcorpus the reranker gains are inside the bootstrap interval. Recall@100
  caps reranker upside at Z."
- The quality versus latency Pareto figure.
- Full results table: dataset by system by metric, with CIs.
- The failure mode taxonomy with real query examples.
- `LIMITATIONS.md`: BEIR label sparsity, single machine timings, model
  versions and dates, no production serving concerns like caching or sharding.
- Tests: the qrels join, the metric implementations against known values from
  a tiny fixture, the fusion function, and a guard that dev and test splits
  never mix.

## Acceptance criteria

- `make analysis` reproduces every number offline from cached artifacts.
- Every system comparison in the README carries a confidence interval.
- The error analysis section exists with real examples, not a placeholder.
- The README states at least one case where the more expensive system is not
  worth it.

## Resume bullet this earns

"Ablated a retrieval stack across N BEIR datasets, measuring what BM25, dense,
hybrid fusion and cross encoder reranking each contribute in nDCG@10 against
p95 latency and cost per 1000 queries, with paired bootstrap intervals and a
hand built taxonomy of the queries the best stack still fails."
