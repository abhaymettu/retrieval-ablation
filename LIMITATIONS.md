# Limitations

Read this before quoting any number from this repo.

## Label sparsity is the biggest one

BEIR qrels are pooled, not exhaustive. A document that no participating system
in the original pooling ever surfaced is unjudged, and this repo, like every
BEIR evaluation, scores unjudged as not relevant. That systematically
underrates any system that finds relevant documents outside the original pool,
which is exactly what a modern dense retriever or a cross encoder is likely to
do. The audit numbers make the scale of this concrete: on fiqa about 0.005
percent of the query by document grid is judged at all, and about 2.6 documents
per query are marked relevant in a 57k document corpus.

The practical consequence: a reranker that appears to gain nothing may be
surfacing genuinely relevant unjudged documents, and this evaluation cannot tell
the difference. Some of the queries in the failure taxonomy are labelled as
label noise for exactly this reason, and that category is a floor on the error,
not a measurement of it.

scifact and fiqa are binary graded, so nDCG's graded discount does no work
there. Only nfcorpus has two relevance levels.

## Timings are single machine, single run, serial

Every latency number comes from one machine, stated in the README, running
queries one at a time with no batching, no caching, no warm query distribution,
and no concurrency. Real serving has all four. The ordering between systems is
robust; the absolute milliseconds are not portable, and neither is the ratio
between a CPU stage like BM25 and a GPU stage like the cross encoder, since
that ratio depends entirely on what hardware each lands on.

Timings were taken once, not repeated across runs, so there is no interval on
the latency numbers the way there is on the quality numbers. Percentiles are
over queries within one run, which captures query to query variation and not
run to run variation.

Two measurement artifacts were found and fixed while building this, which is
the reason to distrust any latency number that was not chased at least this
far. Both would have shipped wrong numbers:

1. **GPU warmup.** With a per stage warmup of 20 queries, the query encoder
   timed at 52 ms on the first dataset measured and 11 ms on the last, in
   measurement order rather than in any order the data justified. The models
   are now driven hard before any clock starts.
2. **Timing one model at a time.** Looping model by model handed whichever ran
   first the per dataset cold start, which reported the 22M parameter encoder
   as *slower* than the 110M one. Both models are now timed inside one loop
   over queries, alternating, which recovers the expected ordering.

A third artifact was found and **not** fixed, because it cannot be: on fiqa,
query encoding measures about 2.6x slower when brute force search over the
177 MB embedding matrix is interleaved with it than when it runs alone, since
the two compete for memory bandwidth on unified memory. Both numbers are
recorded, as `encode_*` and `encode_*_isolated`. The composed per system
totals are unaffected, because a real query pays for both stages either way.
Only the stage by stage attribution is affected: on fiqa some of what the
table charges to encoding properly belongs to search.

## Dollars are a model, not a measurement

Nothing in this repo was ever billed. Latency is measured on local hardware and
then priced at a stated on demand hourly rate for a rented instance, assuming
serial serving at 100 percent utilisation. Both halves of that are assumptions:
that the measured latency transfers to that instance unchanged, and that a real
deployment runs one query at a time on a fully utilised box. Neither is true in
production. Treat the dollar column as a relative ranking of how expensive these
stacks are compared to each other, not as a budget.

Embedding cost is charged at index build time here and then amortised over
nothing, because the corpora are static. A corpus that changes daily makes the
dense and hybrid systems considerably more expensive than this repo suggests.

## Model and data versions

Model weights are pulled from Hugging Face by name, not by revision hash, so a
silent upstream reupload would change the numbers without changing the code.
The BEIR archives are the public zips from the UKP mirror, downloaded fresh by
`make download`. Versions of every Python dependency are pinned in
`requirements.txt`. Build date and machine appear in the README.

## Retrieval and indexing shortcuts

Dense search is exact brute force over the full matrix. Any production system
would use an approximate index, which is faster and loses some recall, so the
dense and hybrid latency numbers here are pessimistic and their recall numbers
optimistic relative to what would actually ship.

BM25 is `rank_bm25`, not Lucene: no stemming, no stopword list, pure Python
scoring. That is slower than Anserini and scores a little lower than the
published BEIR BM25 baseline. The baseline is therefore slightly soft, which
makes every gain over it slightly generous.

No chunking. Documents are indexed whole, title plus body, and truncated at the
model's max sequence length, which discards the tail of long documents for the
dense and cross encoder systems but not for BM25. On corpora with long
documents that quietly favours BM25.

## Statistical caveats

Percentile bootstrap intervals, not BCa, and no correction for the number of
comparisons made. With eight or nine systems compared pairwise across four
metrics per dataset, some intervals that exclude zero will do so by chance.
Treat any single marginal comparison with suspicion and the large, consistent
ones as real.

## Not addressed at all

Query latency under load, caching, sharding, index refresh, tail behaviour with
concurrent requests, multilingual retrieval, domain adaptation or fine tuning,
and the retrieval to generation half of a RAG system. This repo measures
retrieval quality and its cost, and stops there.
