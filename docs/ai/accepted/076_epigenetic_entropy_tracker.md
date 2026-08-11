# TASK 2: Build Epigenetic Entropy Dataloader (Measuring Z)

According to Fedichev's theory, the cumulative aging variable "$Z$" is configurational entropy. Standard epigenetic clocks measure biological age linearly by taking the mean. To measure $Z$, we must output the *statistical dispersion* (variance/entropy) of CpG sites across millions of cells.

Please create a new file: `src/pipeline/sim2real/epigenetic_entropy_dataloader.py`

**Requirements:**
1. **Class Name:** `EpigeneticEntropyLoader`
2. **Data Generation:** 
    *   Mock a batch array of 10,000 aging-related CpG methylation sites (values between 0.0 and 1.0) representing an ensemble of 1,000 cells. Shape should be `[Time, Cells, CpGs]`.
    *   Accept a parameter `biological_age`. 
    *   If `biological_age=50`, generate the methylation states using a high-variance distribution (e.g., wide Gaussian centered at 0.5), representing severe epigenetic drift ($Z$ is high).
    *   If `biological_age=45`, push the values toward a tighter, low-variance bimodal distribution clustered near 0.0 and 1.0 ($Z$ is low).
3. **Update `src/metrics/metrics.py`:**
    *   Add a method `calculate_epigenetic_dispersion(self, cpg_tensor)` to the `ThermodynamicMetrics` class. It should compute the statistical variance or Shannon entropy across the cell dimension for the batch.
    *   Update `extract_fedichev_macrostates()` to include a `Z_epigenetic_entropy` key if the cpg_tensor is provided, using this new method.
