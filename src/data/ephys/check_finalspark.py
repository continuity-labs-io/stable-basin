"""Inspect a FinalSpark whole-life package HDF5 (stimulation protocol summary).

Prints the store keys, organoid metadata, stimulation-parameter variation,
which electrodes were recorded but never stimulated, and inter-stimulus
interval statistics.

Usage:
    python src/data/ephys/check_finalspark.py
    python src/data/ephys/check_finalspark.py --data-dir data/ephys/finalspark/fs369data
"""

import argparse
import glob
import os

import pandas as pd

DEFAULT_DATA_DIR = "data/ephys/finalspark/fs369data"
PACKAGE_SUFFIX = "_package.hdf5"


def find_package(data_dir: str) -> tuple[str, str]:
    """Locate the single `<fs_id>_package.hdf5` file in a directory.

    Args:
        data_dir: Directory containing the FinalSpark export.

    Returns:
        Tuple of (path to the package HDF5, culture ID prefix such as "fs369").

    Raises:
        FileNotFoundError: If no package file exists in `data_dir`.
        ValueError: If more than one package file exists in `data_dir`.
    """
    matches = sorted(glob.glob(os.path.join(data_dir, f"*{PACKAGE_SUFFIX}")))
    if not matches:
        raise FileNotFoundError(f"No *{PACKAGE_SUFFIX} file found in {data_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple package files in {data_dir}: {matches}")
    pkg = matches[0]
    fs_id = os.path.basename(pkg)[: -len(PACKAGE_SUFFIX)]
    return pkg, fs_id


def check_package(pkg: str, fs_id: str) -> dict:
    """Print a stimulation-protocol summary for a FinalSpark package HDF5.

    Args:
        pkg: Path to the `<fs_id>_package.hdf5` file.
        fs_id: Culture ID prefix used in the store keys (e.g. "fs369").

    Returns:
        Dict with keys "stim_electrodes", "event_electrodes",
        "never_stimulated" (sorted lists of ints) and "n_stim" (int).
    """
    with pd.HDFStore(pkg, mode="r") as store:
        print("KEYS:", store.keys())
        stim = store.select(f"{fs_id}_wholelife_stimulations")
        ev_el = store.select(f"{fs_id}_wholelife_events", columns=["electrode"])
        meta = store.select(f"{fs_id}_wholelife_metadata")
        for extra in ("export_state", "export_log"):
            k = f"{fs_id}_wholelife_{extra}"
            print(f"\n--- {extra} ---")
            if f"/{k}" in store.keys():
                print(store.select(k).head(20))
            else:
                print(f"{extra}: not present in store")

    print("\n=== METADATA ===")
    for c in meta.columns:
        print(f"{c:20s} {meta.iloc[0][c]}")

    print("\n=== STIM TABLE ===")
    print("rows:", len(stim))
    print("columns:", list(stim.columns))
    print("time range:", stim["time_of_stim"].min(), "->", stim["time_of_stim"].max())

    # Requirement 8: do the parameters vary?
    print("\n=== PARAMETER VARIATION ===")
    for c in stim.columns:
        if c in ("time_of_stim",):
            continue
        u = stim[c].unique()
        shown = sorted(u.tolist())[:8] if len(u) <= 20 else "(many)"
        print(f"{c:28s} n_unique={len(u):5d}  {shown}")

    # Requirement 4: is any electrode stimulated while others are not?
    stim_el = {int(e) for e in stim["electrode"].unique()}
    all_el = {int(e) for e in ev_el["electrode"].unique()}
    never = sorted(all_el - stim_el)
    print("\n=== SHAM CHECK ===")
    print("electrodes stimulated   :", sorted(stim_el))
    print("electrodes with events  :", sorted(all_el))
    print("recorded but NEVER stim :", never)

    # Is it a protocol or ad hoc?
    t = pd.to_datetime(stim["time_of_stim"]).sort_values()
    gaps = t.diff().dt.total_seconds().dropna()
    print("\n=== INTER-STIM INTERVALS (s) ===")
    print(gaps.describe())
    print("most common gaps:", gaps.round(3).value_counts().head(10).to_dict())

    return {
        "stim_electrodes": sorted(stim_el),
        "event_electrodes": sorted(all_el),
        "never_stimulated": never,
        "n_stim": len(stim),
    }


def main() -> None:
    """Parse CLI arguments and run the package check.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing <fs_id>{PACKAGE_SUFFIX} (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    pkg, fs_id = find_package(args.data_dir)
    print(f"Package: {pkg}  (culture: {fs_id})")
    check_package(pkg, fs_id)


if __name__ == "__main__":
    main()
