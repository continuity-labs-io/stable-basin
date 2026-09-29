Context: Stable Basin repo, src/benchmarks/aging_resilience/tasks/. Fix cohort construction in
all three tasks. Do not change model code. Seed every random source (including random.shuffle)
from config experiment.seed.

1. Worm (worm_task.py): train on EigenWorms_TRAIN.ts. Build eval-young and eval-old from
   EigenWorms_TEST.ts. Today all three sets load the same file.
2. Killifish (killifish_task.py): delete the random-split fallback. If young
   (age/lifespan <= 0.5) or old (> 0.5) is empty, raise ValueError with the counts.
3. Killifish and CATNAP: split by individual (fish_number / mouse_id), never by chunk. Assign
   each individual to train or held-out, 80/20. Train set = young chunks of train individuals
   only. Eval-young / eval-old = young / old chunks of held-out individuals. A held-out
   individual may appear in both eval sets (paired within-animal comparison). No train
   individual may appear in either eval set.
4. Early-stopping validation uses a held-out slice of the train individuals, never eval-young.
5. Write output/benchmarks/aging_resilience/{dataset}_cohorts.json: per set, number of
   individuals, number of chunks, and individual IDs. Assert train IDs are disjoint from eval IDs.
6. CATNAP: print rows per mouse and the median time between consecutive rows. If rows are
   separate recording sessions (median gap > 1 day), stop and report: a 10-row chunk then spans
   weeks to months and must not be integrated as 10 consecutive frames at dt = 0.01.
7. Extend 11_null_control to killifish and CATNAP: split held-out young individuals into halves
   A/B, run the same curvature metric, report the false-positive rate over 200 re-splits.

Gate: PASS if all three tasks build non-empty, individual-disjoint cohorts and each reports a
null false-positive rate. FAIL for any dataset with an empty cohort: stop, and report no
young-vs-old results for it.
