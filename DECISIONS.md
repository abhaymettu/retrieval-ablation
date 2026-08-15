# Decisions

Written as they were made, in order. Every one of these could have gone another
way, and several of them change the headline number.

## Datasets

**scifact, nfcorpus, fiqa.** Small enough to run end to end on a laptop, and
different enough that no single system wins all three: scifact is short
scientific claims against abstracts with about one relevant document per query,
nfcorpus is medical with roughly 38 graded relevant documents per query, fiqa is
long tail financial questions over a corpus more than ten times larger. Left
trec-covid out: it pairs a much larger corpus with only 50 test queries, which
buys a lot of runtime for very little statistical power, and 50 queries makes
the bootstrap intervals wide enough to swamp any conclusion. That corpus was
never downloaded here, so its size is not a number this repo can verify.

**No aggregate score across datasets.** The per dataset numbers disagree with
each other, and that disagreement is the finding. Averaging it away would hide
the only thing worth reporting.

## Splits

**scifact has no dev qrels, so its official train split serves as dev.** Its 809
train queries are disjoint from the 300 test queries, which `tests.py` asserts.
nfcorpus and fiqa use their official dev splits. Every hyperparameter in this
repo is chosen on dev and reported on test, and the report names which split
each number came from.

**Nothing is chosen on test.** That includes the blend weight, which dense model
to use, which first stage feeds the reranker, and which stack the README
recommends. The alpha sweep is also evaluated on test, but only so the README
can show how much a test tuned hybrid would have flattered itself. That number
is reported, never used.

## BM25

**rank_bm25 Okapi with k1 = 0.9, b = 0.4.** Those are the parameters BEIR's own
published Anserini baseline uses, so the baseline is the standard one rather
than one tuned to be beaten. Not tuned here on either split. Tokenisation is
lowercase alphanumeric, no stemming and no stopword list, which costs a little
against a Lucene baseline and keeps the implementation inspectable. This is the
main reason the BM25 numbers here sit slightly below the published BEIR
Anserini figures.

## Dense retrieval

**all-MiniLM-L6-v2 (22.7M, 384 dim) and bge-base-en-v1.5 (109.5M, 768 dim).**
Roughly 5x apart in parameters, which is the point: the question is what the
larger model buys. bge gets its documented query prefix, MiniLM does not,
because that is how each model was trained. Embeddings are L2 normalised so
cosine similarity is a dot product, and search is exact brute force, not ANN.
An ANN index would be faster and would introduce a recall loss this repo does
not want to have to explain.

## Hybrid

**Both fusion methods, because they answer different questions.** Reciprocal
rank fusion has no free parameter and so cannot be overfit. The normalised score
blend has a weight, and the weight is the interesting part: the sweep shows how
sharp the optimum is and therefore how much a hybrid tuned on test is worth
discounting.

**Min max normalisation per query, not global.** BM25 scores are unbounded and
query dependent, so a global normalisation would let long queries dominate.
A document missing from one list scores zero on that side rather than being
dropped, so a document either system likes stays in contention.

## Reranking

**Cross encoder scores are cached for the union of the top 100 across every
first stage.** Scoring the union once instead of per system makes the "what does
a reranker do on top of BM25 alone" comparison nearly free, and that comparison
is what makes the Recall@100 ceiling argument concrete.

**k = 10, 50, 100 come from one cached pass**, since the top 10 is a subset of
the top 100. Below rank k the first stage order is preserved and pushed beneath
every reranked document, so the two segments never interleave.

## Metrics

**trec_eval conventions**: gain 2^rel - 1, discount log2(rank + 1), unjudged
documents count as relevance 0, and the ideal ranking is drawn from the judged
pool. Implemented here rather than pulled from pytrec_eval so the implementation
can be tested against hand computed values, which `tests.py` does.

**Recall@100 is reported for every system, not only the first stages.** It is
the ceiling on what any reranker can recover, and it is the number that decides
whether a reranker is worth buying.

## Statistics

**Paired bootstrap over queries, 10000 resamples, fixed seed.** Paired because
every system sees the same queries, so pairing removes the query difficulty
variance that otherwise dominates. Percentile intervals rather than BCa, which
is a real approximation but not one that changes any verdict here at these
sample sizes. A difference is called distinguishable only when the interval
excludes zero, and the report says so explicitly when it does not.

## Latency and cost

**Timed serially, one query at a time, cold, with 10 warmup queries discarded,
on one stated machine.** No batching and no caching, because that is the
configuration an engineer can reason about before deciding what to optimise.
Stages are timed apart and added per query, so a composed system's p95 is the
p95 of its own totals rather than a sum of stage p95s, which would overstate it.

**Dollars are a model, not a measurement.** Latency is measured locally and then
priced at a stated rented hourly rate. That transfer is the weakest number in
the repo and is flagged as such in LIMITATIONS.md. The ratios between systems
survive the assumption; the absolute dollars do not.

## Error analysis

**Labels are written by hand into `analysis/failure_labels.json` after reading
`output/error_candidates.json`.** The pipeline does not invent categories. If the
labels file is missing, the taxonomy step skips rather than emitting a
placeholder, so an unlabelled repo cannot pretend to have an error analysis.

## Reporting

**README.md is generated from README.template.md.** The prose is written by
hand, every number is a lookup into the generated JSON, and an unresolved
placeholder is a hard error. A number that cannot be reproduced is a number that
cannot be written down.
