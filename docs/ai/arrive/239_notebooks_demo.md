Load the following files into your context:
- `src/models/vessel/vessel_baseline.py`
- `src/data/toy/muller_brown.py`
- `src/data/ephys/maxwell_dataset.py`
- `Makefile`

Please execute the following tasks to extract our physical simulations into dedicated, highly visual entry points for new researchers. Maintain a subdued, professional tone in all generated code, documentation, and logging.

1. **The Observer Zero Demo:**
- Create a new directory named `examples/`.
- Extract the `run_simulation()` function and the `if __name__ == "__main__":` block from the bottom of `src/models/vessel/vessel_baseline.py` and move them into a new file: `examples/01_simulate_observer_zero.py`.
- In this new file, implement `argparse` to expose `--size`, `--sigma-ext`, `--v-wall-threshold`, `--save` (to output an mp4/gif), `--seed`, and `--steps`.
- Implement headless execution logic: if `--save` is passed, do not call `plt.show()`; instead, save the animation to disk directly to prevent crashes on headless servers.
- Update the docstrings and print headers in `01_simulate_observer_zero.py` to accurately describe the simulation as a "FitzHugh-Nagumo reaction-diffusion system with an insulated boundary." Remove any overly dramatic language (e.g., "Observer Zero", "Utopia", or claiming it is strict active inference).

2. **Müller-Brown Toy Physics Notebook:**
- Create a new directory named `notebooks/`.
- Generate a Jupyter Notebook at `notebooks/01_muller_brown_langevin.ipynb`.
- Instantiate the `MullerBrownDataset` from `src/data/toy/muller_brown.py`.
- Generate a matplotlib 2D contour plot of the Müller-Brown potential and overlay a few generated Langevin trajectories.
- **Critical:** Clip the spatial evaluation and contour levels to strictly `[-1.5, 1.2]` for the X-axis and `[-0.5, 2.0]` for the Y-axis to prevent the potential from mathematically blowing up and ruining the visualization.

3. **MaxWell HD-MEA Telemetry Notebook:**
- Generate a Jupyter Notebook at `notebooks/02_maxwell_telemetry.ipynb`.
- Provide a clear, heavily commented example of how to instantiate the `MaxWellHDMEADataset`. 
- Since the user might not have the `.raw.h5` file yet, wrap the instantiation in a `try/except` block that provides a helpful instruction on where to download the data if a `FileNotFoundError` occurs.
- Write matplotlib code to visualize a slice of the telemetry (e.g., plotting the first 10 channels over 5000 frames as a stacked line plot or raster plot).

4. **Makefile Targets:**
In the `Makefile`:
- Add a target `demo-observer:` that executes `python -m examples.01_simulate_observer_zero --save output/observer_demo.mp4`.
- Add a target `notebooks:` that executes `jupyter notebook notebooks/`.
