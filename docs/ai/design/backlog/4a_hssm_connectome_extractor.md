**Context Files to Load / Create:**

- `src/data/biology/celegans_connectome_extractor.py` (Create)

**Task: Paper 2 - H-SSM Structural Topology Extractor (`owmeta`)** We are
building the structural priors (the "physical wires") for the Micro-Layer of our
Hierarchical State-Space Model. Write a standalone Python script that queries
the local `owmeta` database to build the structural adjacency matrix of the C.
elegans nervous system. This matrix will serve as the physical sparsity mask for
our thermodynamic Dissipative Friction ($\Gamma$) operator.

**Core Objectives:**

**1. The Database Connection:**

- Utilize `owmeta_core.command.OWM` to connect to the local database using the
  context identifier 'http://openworm.org/data'.
- Query all `Neuron` objects (imported from `owmeta.neuron`).

**2. Graph Extraction ($N \times N$ Matrix):**

- Iterate through the neurons to extract their synaptic connections (focusing on
  'pre' / 'post' chemical synapses and gap junctions).
- Construct a static $N \times N$ Adjacency Matrix (where $N \approx 302$ is the
  total number of neurons).
- The matrix values should represent the structural weights (e.g., synapse
  counts) or binary connections (1 for wired, 0 for structurally isolated).
- Generate a corresponding list/array of the 302 string names (e.g., 'AVAL',
  'AVB') matching the matrix indices to preserve the biological mapping.

**3. Static Asset Generation:**

- Save the generated adjacency matrix and the neuron name list to a compressed
  array at `data/processed/celegans_connectome_gamma_mask.npz`. (Create the
  directories if they do not exist).
- This allows the JAX Equinox engine to instantly load the physical topology
  into GPU VRAM during training without ever needing to query the RDF database.
- Disconnect cleanly from the OWM context at the end of the script.

**Constraints:**

- Do NOT import `torch`, `jax`, or `equinox` in this file. This is a pure
  offline data-extraction script.
- Ensure the script can handle potential missing connection data gracefully
  without crashing.
- Use the standard Python `logging` library. Emit calm, precise logs detailing
  the extraction progress (e.g.,
  `logger.info("Extracting connectome topology for 302 neurons.")`). Do not use
  dramatic, capitalized, or emoji-laden print statements.
