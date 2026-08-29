Context Files to Load:

src/data/waddington_dataset.py

src/data/async_event_packer.py

src/metrics/metrics.py

src/models/ssm/async_ssm.py

src/models/ssm/state_unpacker.py

(Create Target) src/harness/test_async_ksm.py

Raw Prompt to Execute:

We must mathematically prove that dropping 90% of our zero-padded data and
processing it asynchronously does not destroy the state-space model's ability to
track thermodynamic stability (KSM).

Create a test script `src/harness/test_async_ksm.py`.

1. Generate a sequence using
   `SyntheticWaddingtonDataset(size=1, seq_len=300, density=0.1)`.
2. Inject a biological crash into the dense `x_raw` tensor at frame 150 (e.g.,
   multiply the `x_raw` data from frame 150 to 300 by an exponentially
   increasing scalar to simulate a variance explosion).
3. Wrap this dataset in `AsyncEventPackerDataset(dt_resolution=1.0)` and collate
   it using `ragged_collate_fn` to extract the sparse `events` and `event_mask`.
4. Instantiate the `AsyncMaskAwareSSM(dim=30, d_state=16)` and pass the packed
   events through it.
5. Unpack the resulting sparse hidden states back to a dense timeline of length
   300 using your `unpack_to_dense` method.
6. Flatten the dense hidden states from `[1, 300, 30, 16]` to `[300, 480]` and
   feed them into
   `ThermodynamicMetrics(alpha=500.0).calculate_ksm(..., window_size=10)`.
7. Print out the KSM trajectory and assert that it remains stable (near 1.0)
   before frame 150 and correctly drops towards 0.0 after the crash.
