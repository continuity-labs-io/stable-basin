Context Files to Load:

src/models/ssm/async_ssm.py

(Create Target) src/models/ssm/state_unpacker.py

Raw Prompt to Execute:
Our asynchronous engine outputs sparse, event-driven hidden states `[Batch, Max_Events, Dim, D_State]`. However, thermodynamic metrics like DMD/KSM strictly require uniform, dense temporal snapshots to compute eigenvalues.

Create `src/models/ssm/state_unpacker.py`.
Write a function `unpack_to_dense(h_sparse, events, event_mask, seq_len, dt_resolution=1.0)`.
1. It must map the sparse hidden states back to a uniform dense timeline of shape `[Batch, seq_len, Dim, D_State]`.
2. `events[:, i, 2]` contains the physical timestamp. The target dense time index is `timestamp / dt_resolution`.
3. If an event updates Sensor 3 at `t=5`, that hidden state must persist (forward-fill / Zero-Order Hold) at `t=6`, `t=7`, etc., until Sensor 3 fires again.
4. Implement this efficiently in PyTorch (e.g., using a sequential loop over `seq_len` that carries forward the last known state, or using `scatter` to place the sparse updates onto the dense grid followed by a cumulative forward-fill).
