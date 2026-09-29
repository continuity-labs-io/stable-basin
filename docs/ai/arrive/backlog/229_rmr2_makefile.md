Finally, update the `Makefile` to make the new RMR2 simulation accessible via a single command.

1. Open `Makefile`.
2. Add a new target under the Aging Resilience Suite:
```makefile
.PHONY: rmr2-leaderboard
rmr2-leaderboard:
	python -m src.benchmarks.aging_resilience.12_rmr2_leaderboard --config configs/rmr2_simulation.yaml
```

```
.PHONY: run-rmr2
run-rmr2:
	$(MAKE) rmr2-leaderboard
```

Test
- run `make run-rmr2`
- sanity check ranked leaderboard CSV
