"""Header probe and inventory utilities for recovery census benchmarks.

This module inspects recording metadata and file headers without loading raw
voltage arrays into memory. It reports channel counts, sample rates, spatial
coordinate availability, and event tables.

Governing rule: a value that should be read from the file is never replaced by
a default. When a required value is absent, the probe sets ``probe_error`` and
returns that field as ``None``.
"""

from dataclasses import asdict, dataclass, field
import fnmatch
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Coordinate column-name pairs searched for in FinalSpark tables.
XY_NAME_PAIRS: tuple[tuple[str, str], ...] = (
    ("x", "y"),
    ("electrode_x", "electrode_y"),
    ("pos_x", "pos_y"),
)

# MaxWell sample-rate lookup order. Each entry is (kind, location).
# kind "sig_attr": attribute name on the located signal dataset.
# kind "dataset": absolute HDF5 dataset path holding the rate.
MAXWELL_SAMPLE_RATE_LOOKUPS: tuple[tuple[str, str], ...] = (
    ("sig_attr", "sampling_rate"),
    ("sig_attr", "sampling"),
    ("sig_attr", "samplingRate"),
    ("sig_attr", "fs"),
    ("dataset", "/settings/sampling"),
    ("dataset", "/settings/sampling_rate"),
    ("dataset", "/sampling_rate"),
    ("dataset", "/recordings/rec0000/well000/settings/sampling"),
    ("dataset", "/data_store/data0000/settings/sampling"),
)

THREEBRAIN_SAMPLE_RATE_KEY = "3BRecInfo/3BRecVars/SamplingRate"
THREEBRAIN_NROWS_KEY = "3BRecInfo/3BMeaChip/NRows"
THREEBRAIN_NCOLS_KEY = "3BRecInfo/3BMeaChip/NCols"
THREEBRAIN_PITCH_KEY = "3BRecInfo/3BMeaChip/Pitch"


@dataclass
class FileEntry:
    """Represents a discovered raw electrophysiology recording file.

    Args:
        path: Path to the recording file.
        source: Dataset source identifier ('finalspark', 'maxwell', 'threebrain',
            'organoid_pharma').
        bytes: Size of the file in bytes.
        mtime: Last modification time of the file.
    """

    path: str
    source: str
    bytes: int
    mtime: float


@dataclass
class RecordingProbe:
    """Header and structural metadata extracted from a recording without reading raw voltages.

    Args:
        recording_id: Deterministic 12-char hex SHA-256 of f"{source}:{path}", or of
            f"{source}:{path}:{session_index}" for session-segmented sources.
        source: Dataset source identifier.
        path: Path to the recording file.
        group_token: Inferred grouping identifier (e.g. 'fs437' or 'Drug_2953').
        recording_timestamp: Timestamp string if available in header, else None.
        n_channels: Number of distinct recording channels.
        n_rows: Number of grid rows if applicable, else None.
        n_cols: Number of grid columns if applicable, else None.
        duration_s: Duration of the recording (or session) in seconds, None if it
            cannot be computed from values read from the file.
        sample_rate_hz: Sampling frequency in Hertz read from the file, else None.
        has_xy: Whether any electrode coordinates (physical or grid index) are available.
        xy_units: Units of the coordinates: 'um', 'grid_index', 'unknown' (coordinates
            found but units not stated in the file), or 'absent'.
        xy_provenance: Origin of spatial coordinates (e.g. 'mapping_dataset',
            'grid_times_pitch', 'derived_from_grid', 'absent', 'absent_verified').
        has_spike_table: Whether the file contains a pre-detected spike/event table.
        probe_error: Error message if probing failed or constraints were violated, else None.
        session_index: Index of the session within the file for session-segmented
            sources (FinalSpark); None for single-session sources.
        maxwell_layout: 'v1' (/sig at root), 'v2' (/recordings or /data_store), or
            'unknown' for MaxWell-format files; None for other sources.
        has_stim_table: Whether the file contains a stimulation event table.
        n_stim_events: Number of rows in the stimulation table, else None.
        stim_time_min: Earliest stimulation time (ISO string), else None.
        stim_time_max: Latest stimulation time (ISO string), else None.
        has_env_channels: Whether the file contains environmental covariate tables.
        env_channel_names: HDF5 keys of the environmental covariate tables.
    """

    recording_id: str
    source: str
    path: str
    group_token: str | None
    recording_timestamp: str | None
    n_channels: int | None
    n_rows: int | None
    n_cols: int | None
    duration_s: float | None
    sample_rate_hz: float | None
    has_xy: bool
    xy_units: str
    xy_provenance: str
    has_spike_table: bool
    probe_error: str | None
    session_index: int | None = None
    maxwell_layout: str | None = None
    has_stim_table: bool = False
    n_stim_events: int | None = None
    stim_time_min: str | None = None
    stim_time_max: str | None = None
    has_env_channels: bool = False
    env_channel_names: list[str] = field(default_factory=list)


def _recording_id(source: str, path: str, session_index: int | None = None) -> str:
    """Computes the deterministic 12-char recording identifier.

    Args:
        source: Dataset source identifier.
        path: Path to the recording file.
        session_index: Session index for session-segmented sources, else None.

    Returns:
        First 12 hex characters of the SHA-256 digest.
    """
    key = f"{source}:{path}" if session_index is None else f"{source}:{path}:{session_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _append_error(existing: str | None, message: str) -> str:
    """Appends an error message to an existing probe_error string.

    Args:
        existing: Current probe_error value (may be None).
        message: Message to append.

    Returns:
        Combined error string separated by '; '.
    """
    return message if existing is None else f"{existing}; {message}"


def _decode_string_scalar(val: Any) -> str:
    """Extracts and decodes a string from Python, NumPy, or HDF5 scalar/bytes values.

    Args:
        val: Scalar, bytes, or length-1 array value.

    Returns:
        Decoded string.
    """
    if hasattr(val, "__len__") and not isinstance(val, (str, bytes, np.bytes_)) and len(val) > 0:
        val = val[0]
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _decode_names(values: Any) -> list[str]:
    """Decodes an iterable of bytes/str names to a list of str.

    Args:
        values: Iterable of bytes or str values.

    Returns:
        List of decoded strings.
    """
    return [
        v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)
        for v in values
    ]


def _find_xy_pair(names: list[str] | tuple[str, ...] | set[str]) -> tuple[str, str] | None:
    """Returns the first coordinate name pair fully present in ``names``.

    Args:
        names: Candidate column/field/child names.

    Returns:
        The matching (x_name, y_name) pair, or None if no pair is present.
    """
    name_set = set(names)
    for x_name, y_name in XY_NAME_PAIRS:
        if x_name in name_set and y_name in name_set:
            return (x_name, y_name)
    return None


def infer_group_token(path: str) -> str | None:
    """Infers an experimental group token from the filename or parent directory.

    Args:
        path: Path to the recording file.

    Returns:
        Inferred group token string, or None if no token can be identified.
    """
    p = Path(path)
    stem = p.name
    for suffix in [".raw.h5", ".package.hdf5", ".hdf5", ".h5", ".brw"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    # Look for known patterns like Drug_2953 or fs437
    m = re.match(r"(Drug_\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"(fs\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1)

    parent = p.parent.name
    m = re.match(r"(Drug_\d+)", parent, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"(fs\d+)", parent, re.IGNORECASE)
    if m:
        return m.group(1)

    # Strip condition / component suffixes (e.g. Drug_2953_control -> Drug_2953)
    cleaned = re.sub(
        r"_(control|\d+uM|\d+uM_washout|package|raw)$", "", stem, flags=re.IGNORECASE
    )
    if cleaned:
        return cleaned

    return stem if stem else None


def inventory(roots: dict[str, str], patterns: dict[str, list[str]]) -> list[FileEntry]:
    """Resolves each source's glob patterns under that source's own root directory.

    Args:
        roots: Mapping of source name to root directory path.
        patterns: Mapping of source name to a list of glob patterns, resolved
            relative to that source's root (e.g. ``{"threebrain": ["**/*.brw"]}``).

    Returns:
        Deterministically sorted list of discovered FileEntry objects.

    Raises:
        ValueError: If a configured root has no patterns, or if two sources resolve
            to the same file.
    """
    missing = sorted(s for s in roots if not patterns.get(s))
    if missing:
        raise ValueError(f"No glob patterns configured for sources: {missing}")

    entries: list[FileEntry] = []
    owner_by_realpath: dict[str, str] = {}

    for source, root in roots.items():
        root_path = Path(root)
        if not root_path.exists():
            logger.info("Root directory does not exist for source '%s': %s", source, root)
            continue

        seen_in_source: set[str] = set()
        for pattern in patterns[source]:
            for match in root_path.glob(pattern):
                if not match.is_file() or match.name.startswith("."):
                    continue
                real = os.path.realpath(match)
                if real in seen_in_source:
                    continue
                if real in owner_by_realpath and owner_by_realpath[real] != source:
                    raise ValueError(
                        f"File {real} resolved by both sources "
                        f"'{owner_by_realpath[real]}' and '{source}'"
                    )
                seen_in_source.add(real)
                owner_by_realpath[real] = source
                full_path = os.path.normpath(str(match))
                entries.append(
                    FileEntry(
                        path=full_path,
                        source=source,
                        bytes=os.path.getsize(full_path),
                        mtime=os.path.getmtime(full_path),
                    )
                )

    entries.sort(key=lambda e: (e.source, e.path))
    return entries


def _visited_keys(f: h5py.File) -> list[str]:
    """Lists every object name in an HDF5 file via Group.visit().

    Args:
        f: Open HDF5 file.

    Returns:
        List of all visited object names in visit order.
    """
    names: list[str] = []
    f.visit(names.append)
    return names


def _maxwell_layout(f: h5py.File) -> str:
    """Classifies a MaxWell file as layout v1 or v2.

    Args:
        f: Open HDF5 file.

    Returns:
        'v1' if /sig is at the root, 'v2' if /recordings or /data_store exists,
        else 'unknown'.
    """
    if "sig" in f:
        return "v1"
    if "recordings" in f or "data_store" in f:
        return "v2"
    return "unknown"


def probe_maxwell(entry: FileEntry) -> tuple[RecordingProbe, dict[str, Any]]:
    """Probes a MaxWell Biosystems .raw.h5 file headers and mapping datasets.

    Args:
        entry: FileEntry describing the MaxWell file.

    Returns:
        A tuple of (RecordingProbe, assumptions_dict). Missing signal datasets or
        sample rates are reported via ``probe_error`` with the field set to None.
    """
    assumptions: dict[str, Any] = {}
    probe_error: str | None = None
    n_channels: int | None = None
    n_frames: int | None = None
    sample_rate_hz: float | None = None
    duration_s: float | None = None
    has_xy = False
    xy_units = "absent"
    xy_provenance = "absent"
    recording_timestamp: str | None = None

    with h5py.File(entry.path, "r") as f:
        layout = _maxwell_layout(f)
        assumptions["maxwell_layout"] = layout

        # 1. Locate voltage dataset
        signal_candidates = ["/sig", "/routing/lsb", "/mapping/sig"]
        sig_dataset = None
        for key in signal_candidates:
            if key in f and isinstance(f[key], h5py.Dataset):
                sig_dataset = f[key]
                assumptions["signal_key"] = key
                break

        if sig_dataset is None:
            probe_error = _append_error(
                probe_error, f"signal dataset not found in {signal_candidates}"
            )
        else:
            n_channels, n_frames = int(sig_dataset.shape[0]), int(sig_dataset.shape[1])
            assumptions["sig_shape"] = [n_channels, n_frames]

        # 2. Sampling rate, tried in a fixed order; the hit is recorded.
        tried: list[str] = []
        for kind, location in MAXWELL_SAMPLE_RATE_LOOKUPS:
            if kind == "sig_attr":
                label = f"sig.attrs['{location}']"
                tried.append(label)
                if sig_dataset is not None and location in sig_dataset.attrs:
                    sample_rate_hz = float(np.squeeze(sig_dataset.attrs[location]))
                    assumptions["sample_rate_source"] = label
                    break
            else:
                tried.append(location)
                if location in f and isinstance(f[location], h5py.Dataset):
                    sample_rate_hz = float(np.squeeze(f[location][()]))
                    assumptions["sample_rate_source"] = location
                    break

        if sample_rate_hz is None:
            assumptions["sample_rate_source"] = None
            assumptions["sample_rate_keys_tried"] = tried
            assumptions["top_level_keys"] = list(f.keys())
            assumptions["key_tree"] = _visited_keys(f)
            if "version" in f and isinstance(f["version"], h5py.Dataset):
                assumptions["version"] = _decode_string_scalar(f["version"][()])
            logger.warning(
                "sample_rate not found in %s for %s; key tree: %s",
                entry.path,
                tried,
                assumptions["key_tree"],
            )
            probe_error = _append_error(probe_error, f"sample_rate not found in {tried}")
        elif sample_rate_hz <= 0:
            probe_error = _append_error(
                probe_error, f"sample_rate {sample_rate_hz} read from file is not positive"
            )
            sample_rate_hz = None
        elif n_frames is not None:
            duration_s = float(n_frames / sample_rate_hz)

        # 3. Mapping / coordinates
        mapping_obj = f.get("/mapping")
        if mapping_obj is not None:
            if isinstance(mapping_obj, h5py.Dataset) and mapping_obj.dtype.names is not None:
                names = list(mapping_obj.dtype.names)
                assumptions["mapping_fields"] = names
                if "x" in names and "y" in names:
                    has_xy = True
            elif isinstance(mapping_obj, h5py.Group):
                assumptions["mapping_subkeys"] = list(mapping_obj.keys())
                if "x" in mapping_obj and "y" in mapping_obj:
                    has_xy = True
            if has_xy:
                xy_units = "um"
                xy_provenance = "mapping_dataset"
                assumptions["xy_units_basis"] = (
                    "MaxWell mapping x/y are electrode positions in um (vendor format "
                    "convention; units are not stored in the file)"
                )
        else:
            assumptions["mapping"] = "not_found"

        # 4. Timestamp
        if "time" in f:
            try:
                raw_time_str = _decode_string_scalar(f["time"][()])
                m = re.search(
                    r"start:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})",
                    raw_time_str,
                )
                recording_timestamp = m.group(1) if m else (raw_time_str.strip() or None)
            except (KeyError, OSError, TypeError, ValueError, UnicodeDecodeError):
                recording_timestamp = None

    probe_result = RecordingProbe(
        recording_id=_recording_id(entry.source, entry.path),
        source=entry.source,
        path=entry.path,
        group_token=infer_group_token(entry.path),
        recording_timestamp=recording_timestamp,
        n_channels=n_channels,
        n_rows=None,
        n_cols=None,
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        has_xy=has_xy,
        xy_units=xy_units,
        xy_provenance=xy_provenance,
        has_spike_table=False,
        probe_error=probe_error,
        maxwell_layout=layout,
    )
    return probe_result, assumptions


def probe_threebrain(entry: FileEntry) -> tuple[RecordingProbe, dict[str, Any]]:
    """Probes a 3Brain .brw file header and chip structure without reading raw voltage.

    Args:
        entry: FileEntry describing the 3Brain file.

    Returns:
        A tuple of (RecordingProbe, assumptions_dict). Missing grid dimensions or
        sample rate are reported via ``probe_error`` with the field set to None.

    Raises:
        KeyError: If the '3BData/Raw' dataset is missing.
    """
    assumptions: dict[str, Any] = {}
    probe_error: str | None = None

    with h5py.File(entry.path, "r") as f:
        if "3BData/Raw" not in f:
            raise KeyError(f"'3BData/Raw' dataset missing in {entry.path}")

        n_elements = int(f["3BData/Raw"].shape[0])
        assumptions["n_raw_elements"] = n_elements

        # Grid rows and cols
        n_rows: int | None = None
        n_cols: int | None = None
        if THREEBRAIN_NROWS_KEY in f and THREEBRAIN_NCOLS_KEY in f:
            n_rows = int(np.squeeze(f[THREEBRAIN_NROWS_KEY][()]))
            n_cols = int(np.squeeze(f[THREEBRAIN_NCOLS_KEY][()]))
            assumptions["grid_source"] = "3BRecInfo/3BMeaChip"
        else:
            assumptions["grid_source"] = None
            probe_error = _append_error(
                probe_error,
                f"grid dimensions not found in {[THREEBRAIN_NROWS_KEY, THREEBRAIN_NCOLS_KEY]}",
            )

        n_channels = (n_rows * n_cols) if (n_rows is not None and n_cols is not None) else None

        # Sampling rate
        sample_rate_hz: float | None = None
        if THREEBRAIN_SAMPLE_RATE_KEY in f:
            sample_rate_hz = float(np.squeeze(f[THREEBRAIN_SAMPLE_RATE_KEY][()]))
            assumptions["sample_rate_source"] = THREEBRAIN_SAMPLE_RATE_KEY
        else:
            assumptions["sample_rate_source"] = None
            probe_error = _append_error(
                probe_error, f"sample_rate not found in {[THREEBRAIN_SAMPLE_RATE_KEY]}"
            )

        duration_s: float | None = None
        if n_channels and n_channels > 0 and sample_rate_hz is not None and sample_rate_hz > 0:
            duration_s = float((n_elements / n_channels) / sample_rate_hz)

        # Coordinates: grid indices, converted to um only if Pitch is in the file.
        has_grid = n_rows is not None and n_cols is not None and n_rows * n_cols > 0
        has_pitch = THREEBRAIN_PITCH_KEY in f
        if has_pitch:
            assumptions["pitch_um"] = float(np.squeeze(f[THREEBRAIN_PITCH_KEY][()]))
        else:
            assumptions["pitch"] = "not_found"

        if has_grid and has_pitch:
            has_xy, xy_units, xy_provenance = True, "um", "grid_times_pitch"
        elif has_grid:
            has_xy, xy_units, xy_provenance = True, "grid_index", "derived_from_grid"
        else:
            has_xy, xy_units, xy_provenance = False, "absent", "absent"

        recording_timestamp = None
        if "3BUserInfo/ExpDate" in f:
            try:
                recording_timestamp = _decode_string_scalar(f["3BUserInfo/ExpDate"][()])
            except (KeyError, OSError, TypeError, ValueError, UnicodeDecodeError):
                recording_timestamp = None

    probe_result = RecordingProbe(
        recording_id=_recording_id(entry.source, entry.path),
        source=entry.source,
        path=entry.path,
        group_token=infer_group_token(entry.path),
        recording_timestamp=recording_timestamp,
        n_channels=n_channels,
        n_rows=n_rows,
        n_cols=n_cols,
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        has_xy=has_xy,
        xy_units=xy_units,
        xy_provenance=xy_provenance,
        has_spike_table=False,
        probe_error=probe_error,
    )
    return probe_result, assumptions


def segment_sessions(sorted_times_ns: np.ndarray, gap_s: float) -> list[tuple[int, int]]:
    """Splits sorted event times into sessions wherever a gap exceeds ``gap_s``.

    A gap exactly equal to ``gap_s`` does not split.

    Args:
        sorted_times_ns: 1-D int64 array of event times in nanoseconds, ascending.
        gap_s: Gap threshold in seconds; must be positive.

    Returns:
        List of half-open (start, stop) index ranges into ``sorted_times_ns``, one
        per session, in time order. Empty input returns an empty list.

    Raises:
        ValueError: If ``gap_s`` is not positive, the input is not 1-D, or the input
            is not sorted ascending.
    """
    if gap_s <= 0:
        raise ValueError(f"gap_s must be positive, got {gap_s}")
    times = np.asarray(sorted_times_ns, dtype=np.int64)
    if times.ndim != 1:
        raise ValueError(f"sorted_times_ns must be 1-D, got shape {times.shape}")
    if times.size == 0:
        return []
    diffs = np.diff(times)
    if np.any(diffs < 0):
        raise ValueError("sorted_times_ns must be sorted ascending")
    gap_ns = int(round(gap_s * 1e9))
    split_after = np.flatnonzero(diffs > gap_ns) + 1
    starts = np.concatenate(([0], split_after))
    stops = np.concatenate((split_after, [times.size]))
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def _h5_time_to_ns(raw: np.ndarray, dtype_attr: Any, column: str) -> np.ndarray:
    """Converts a PyTables-stored datetime column to int64 nanoseconds.

    Args:
        raw: Raw integer values read from the table field.
        dtype_attr: The table's '<column>_dtype' attribute (e.g. b'datetime64[ns]').
        column: Column name, used in error messages.

    Returns:
        int64 array of nanoseconds since the epoch.

    Raises:
        ValueError: If the stored dtype attribute is absent or not datetime64.
    """
    if dtype_attr is None:
        raise ValueError(f"'{column}_dtype' attribute not found; time units unknown")
    dtype_str = _decode_string_scalar(dtype_attr)
    if not dtype_str.startswith("datetime64"):
        raise ValueError(f"column '{column}' stored as {dtype_str}, not datetime64")
    as_dt = np.asarray(raw).view(np.dtype(dtype_str))
    return as_dt.astype("datetime64[ns]").view(np.int64)


def _read_table_columns(
    f: h5py.File,
    path: str,
    key: str,
    columns: list[str],
    time_columns: set[str],
) -> tuple[dict[str, np.ndarray], str]:
    """Reads only the named columns of a pandas-written HDF5 table.

    Uses h5py field reads on '<key>/table' when every column is a stored field;
    otherwise uses ``pd.read_hdf(path, key, columns=...)``.

    Args:
        f: Open HDF5 file (read-only).
        path: Path of the same file, for the pandas route.
        key: Top-level key of the table.
        columns: Column names to read.
        time_columns: Subset of ``columns`` to return as int64 nanoseconds.

    Returns:
        Tuple of (mapping column -> array, read method label).

    Raises:
        KeyError: If a requested column is not present.
    """
    table_path = f"{key}/table"
    if table_path in f and isinstance(f[table_path], h5py.Dataset):
        ds = f[table_path]
        names = ds.dtype.names or ()
        if all(c in names for c in columns):
            out: dict[str, np.ndarray] = {}
            for c in columns:
                raw = ds.fields(c)[()]
                if c in time_columns:
                    out[c] = _h5_time_to_ns(raw, ds.attrs.get(f"{c}_dtype"), c)
                else:
                    out[c] = np.asarray(raw)
            return out, "h5py_fields"

    df = pd.read_hdf(path, key=key, columns=list(columns))
    out = {}
    for c in columns:
        if c not in df.columns:
            raise KeyError(f"column '{c}' not found in table '{key}'")
        if c in time_columns:
            t = pd.to_datetime(df[c], utc=True).dt.tz_convert(None).dt.as_unit("ns")
            out[c] = t.to_numpy().view(np.int64)
        else:
            out[c] = df[c].to_numpy()
    return out, "pandas_read_hdf_columns"


def _pandas_column_names(path: str, key: str) -> list[str]:
    """Reads the column names of a pandas-written HDF5 object without reading rows.

    Args:
        path: Path to the HDF5 file.
        key: Top-level key of the pandas object.

    Returns:
        List of column names (empty for non-DataFrame objects).
    """
    obj = pd.read_hdf(path, key=key, start=0, stop=0)
    if isinstance(obj, pd.DataFrame):
        return [str(c) for c in obj.columns]
    return []


def _object_field_names(f: h5py.File, path: str, key: str) -> list[str]:
    """Returns column / field names for an HDF5 object without reading rows.

    Args:
        f: Open HDF5 file.
        path: Path of the same file, for the pandas route.
        key: Object key.

    Returns:
        Field names from compound dtype, pandas column list, or child names.
    """
    obj = f[key]
    if isinstance(obj, h5py.Dataset):
        return list(obj.dtype.names) if obj.dtype.names is not None else []
    if "pandas_type" in obj.attrs:
        return _pandas_column_names(path, key)
    return list(obj.keys())


def _ns_to_iso(value_ns: int) -> str:
    """Formats nanoseconds since the epoch as an ISO 8601 string.

    Args:
        value_ns: Nanoseconds since the epoch.

    Returns:
        ISO 8601 timestamp string.
    """
    return pd.Timestamp(int(value_ns), unit="ns").isoformat()


def _finalspark_xy_search(
    f: h5py.File,
    path: str,
    events_key: str,
    keys: list[str],
    assumptions: dict[str, Any],
) -> tuple[bool, str, str]:
    """Searches a FinalSpark package for electrode coordinates.

    Search order: events table columns, every '*metadata*' key, then the full
    HDF5 tree via Group.visit().

    Args:
        f: Open HDF5 file.
        path: Path of the same file.
        events_key: Key of the events table.
        keys: Top-level keys of the file.
        assumptions: Dict updated in place with the searched names.

    Returns:
        Tuple of (has_xy, xy_units, xy_provenance).
    """
    # 1. Events table column names.
    events_columns = _object_field_names(f, path, events_key)
    assumptions["events_columns"] = events_columns
    pair = _find_xy_pair(events_columns)
    if pair is not None:
        assumptions["xy_source_key"] = events_key
        assumptions["xy_columns"] = list(pair)
        return True, "unknown", "events_table_columns"

    # 2. Every key matching *metadata*.
    metadata_fields: dict[str, list[str]] = {}
    for k in keys:
        if fnmatch.fnmatch(k, "*metadata*"):
            metadata_fields[k] = _object_field_names(f, path, k)
    assumptions["metadata_fields"] = metadata_fields
    for k, names in metadata_fields.items():
        pair = _find_xy_pair(names)
        if pair is not None:
            assumptions["xy_source_key"] = k
            assumptions["xy_columns"] = list(pair)
            return True, "unknown", "metadata_table"

    # 3. Full HDF5 tree.
    visited = _visited_keys(f)
    for name in visited:
        obj = f[name]
        candidates: list[str] = []
        if isinstance(obj, h5py.Dataset):
            if obj.dtype.names is not None:
                candidates = list(obj.dtype.names)
            elif name.rsplit("/", 1)[-1] == "axis0" or name.endswith("_items"):
                if obj.dtype.kind in "SOU" and obj.ndim == 1:
                    candidates = _decode_names(obj[()])
        else:
            candidates = list(obj.keys())
            if "pandas_type" in obj.attrs and "/" not in name:
                candidates = candidates + _pandas_column_names(path, name)
        pair = _find_xy_pair(candidates)
        if pair is not None:
            assumptions["xy_source_key"] = name
            assumptions["xy_columns"] = list(pair)
            return True, "unknown", "mapping_dataset"

    assumptions["visited_keys"] = visited
    return False, "absent", "absent_verified"


def probe_finalspark(
    entry: FileEntry, session_gap_s: float
) -> tuple[list[RecordingProbe], dict[str, Any]]:
    """Probes a FinalSpark package HDF5 file and emits one probe per recording session.

    Only the 'time_of_event' and 'electrode' columns of the events table are read.
    Sessions are split wherever consecutive events are more than ``session_gap_s``
    apart.

    Args:
        entry: FileEntry describing the FinalSpark file.
        session_gap_s: Gap threshold in seconds that separates sessions.

    Returns:
        A tuple of (list of RecordingProbe, one per session; assumptions_dict).

    Raises:
        KeyError: If no events table or required events column is found.
    """
    assumptions: dict[str, Any] = {"session_gap_s": float(session_gap_s)}
    group_token = infer_group_token(entry.path)

    with h5py.File(entry.path, "r") as f:
        keys = list(f.keys())
        assumptions["hdf5_keys"] = keys

        events_key = next((k for k in keys if "events" in k), None)
        if events_key is None:
            raise KeyError(f"No events table found among keys {keys} in {entry.path}")
        assumptions["events_key"] = events_key

        has_xy, xy_units, xy_provenance = _finalspark_xy_search(
            f, entry.path, events_key, keys, assumptions
        )

        # Stimulation table: row count and time range only.
        stim_key = next((k for k in keys if "stimulation" in k), None)
        has_stim_table = stim_key is not None
        n_stim_events: int | None = None
        stim_time_min: str | None = None
        stim_time_max: str | None = None
        if stim_key is not None:
            assumptions["stim_key"] = stim_key
            stim_cols, stim_method = _read_table_columns(
                f, entry.path, stim_key, ["time_of_stim"], {"time_of_stim"}
            )
            stim_times = stim_cols["time_of_stim"]
            assumptions["stim_read_method"] = stim_method
            n_stim_events = int(stim_times.size)
            if stim_times.size > 0:
                stim_time_min = _ns_to_iso(int(stim_times.min()))
                stim_time_max = _ns_to_iso(int(stim_times.max()))

        # Environmental covariate tables: names only.
        env_channel_names = [k for k in keys if "incubator" in k]
        assumptions["env_keys"] = env_channel_names

        # Events: only time_of_event and electrode columns.
        ev_cols, ev_method = _read_table_columns(
            f, entry.path, events_key, ["time_of_event", "electrode"], {"time_of_event"}
        )
        assumptions["events_read_method"] = ev_method

    times_ns = ev_cols["time_of_event"]
    electrodes = ev_cols["electrode"]
    order = np.argsort(times_ns, kind="stable")
    times_ns = times_ns[order]
    electrodes = electrodes[order]
    sessions = segment_sessions(times_ns, session_gap_s)
    assumptions["n_events"] = int(times_ns.size)
    assumptions["n_sessions"] = len(sessions)

    shared = dict(
        source=entry.source,
        path=entry.path,
        group_token=group_token,
        n_rows=None,
        n_cols=None,
        sample_rate_hz=None,
        has_xy=has_xy,
        xy_units=xy_units,
        xy_provenance=xy_provenance,
        has_spike_table=True,
        has_stim_table=has_stim_table,
        n_stim_events=n_stim_events,
        stim_time_min=stim_time_min,
        stim_time_max=stim_time_max,
        has_env_channels=len(env_channel_names) > 0,
        env_channel_names=env_channel_names,
    )

    if not sessions:
        empty = RecordingProbe(
            recording_id=_recording_id(entry.source, entry.path),
            recording_timestamp=None,
            n_channels=None,
            duration_s=None,
            probe_error=f"events table '{events_key}' is empty",
            session_index=None,
            **shared,
        )
        return [empty], assumptions

    probes: list[RecordingProbe] = []
    for idx, (start, stop) in enumerate(sessions):
        t_first = int(times_ns[start])
        t_last = int(times_ns[stop - 1])
        probes.append(
            RecordingProbe(
                recording_id=_recording_id(entry.source, entry.path, idx),
                recording_timestamp=_ns_to_iso(t_first),
                n_channels=int(np.unique(electrodes[start:stop]).size),
                duration_s=float((t_last - t_first) / 1e9),
                probe_error=None,
                session_index=idx,
                **shared,
            )
        )
    return probes, assumptions


def probe(
    entry: FileEntry,
    census_skip_s: float,
    census_window_len_s: float,
    finalspark_session_gap_s: float,
) -> tuple[list[RecordingProbe], dict[str, Any]]:
    """Inspects file headers and metadata to extract recording properties without
    reading raw voltage arrays.

    Args:
        entry: FileEntry describing the file to probe.
        census_skip_s: Seconds skipped at the start of each recording.
        census_window_len_s: Length in seconds of one census window. A probe whose
            duration is <= census_skip_s + census_window_len_s gets a probe_error.
        finalspark_session_gap_s: Gap in seconds that separates FinalSpark sessions.

    Returns:
        A tuple of (list of RecordingProbe for this file, assumptions_dict). The list
        has one element for single-session sources and one per session for FinalSpark.
    """
    try:
        if entry.source in ("maxwell", "organoid_pharma"):
            single, assumptions = probe_maxwell(entry)
            probes = [single]
        elif entry.source == "threebrain":
            single, assumptions = probe_threebrain(entry)
            probes = [single]
        elif entry.source == "finalspark":
            probes, assumptions = probe_finalspark(entry, finalspark_session_gap_s)
        else:
            raise ValueError(f"Unknown source identifier: {entry.source}")

        min_duration_s = census_skip_s + census_window_len_s
        for p in probes:
            if p.duration_s is not None and p.duration_s <= min_duration_s:
                p.probe_error = _append_error(
                    p.probe_error,
                    f"duration_s ({p.duration_s:.1f}s) <= census_skip_s + "
                    f"census_window_len_s ({min_duration_s:.1f}s)",
                )
            if p.probe_error is not None:
                logger.warning("Probe error for %s (%s): %s", p.path, p.source, p.probe_error)

        return probes, assumptions

    except Exception as e:
        logger.warning("Probe failed for %s (%s): %r", entry.path, entry.source, e)
        fallback_probe = RecordingProbe(
            recording_id=_recording_id(entry.source, entry.path),
            source=entry.source,
            path=entry.path,
            group_token=infer_group_token(entry.path),
            recording_timestamp=None,
            n_channels=None,
            n_rows=None,
            n_cols=None,
            duration_s=None,
            sample_rate_hz=None,
            has_xy=False,
            xy_units="absent",
            xy_provenance="absent",
            has_spike_table=False,
            probe_error=repr(e),
        )
        return [fallback_probe], {"error": repr(e)}


PROBE_TABLE_COLUMNS: list[str] = [
    "recording_id",
    "source",
    "path",
    "session_index",
    "group_token",
    "recording_timestamp",
    "n_channels",
    "n_rows",
    "n_cols",
    "duration_s",
    "sample_rate_hz",
    "has_xy",
    "xy_units",
    "xy_provenance",
    "maxwell_layout",
    "has_spike_table",
    "has_stim_table",
    "n_stim_events",
    "stim_time_min",
    "stim_time_max",
    "has_env_channels",
    "env_channel_names",
    "probe_error",
    "bytes",
    "mtime",
]


def run_census_probe(
    config_path: str = "configs/recovery_census.yaml",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Runs inventory and probe pipeline from configuration and writes outputs.

    Required config keys: roots, patterns, output_dir, census_skip_s,
    census_window_len_s, finalspark_session_gap_s.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        A tuple of (probe_dataframe with one row per probe, assumptions_summary_dict).

    Raises:
        KeyError: If a required config key is missing.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    roots: dict[str, str] = cfg["roots"]
    patterns: dict[str, list[str]] = cfg["patterns"]
    output_dir: str = cfg["output_dir"]
    census_skip_s = float(cfg["census_skip_s"])
    census_window_len_s = float(cfg["census_window_len_s"])
    finalspark_session_gap_s = float(cfg["finalspark_session_gap_s"])

    entries = inventory(roots, patterns)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        "Discovered %d recording files across %d configured roots.", len(entries), len(roots)
    )

    probed_results: list[tuple[list[RecordingProbe], FileEntry, dict[str, Any]]] = []
    for entry in entries:
        probes, assumptions = probe(
            entry,
            census_skip_s=census_skip_s,
            census_window_len_s=census_window_len_s,
            finalspark_session_gap_s=finalspark_session_gap_s,
        )
        probed_results.append((probes, entry, assumptions))

    rows: list[dict[str, Any]] = []
    for probes, entry, _ in probed_results:
        for rec_probe in probes:
            row = asdict(rec_probe)
            row["env_channel_names"] = ";".join(rec_probe.env_channel_names)
            row["bytes"] = entry.bytes
            row["mtime"] = entry.mtime
            rows.append(row)

    df = (
        pd.DataFrame(rows, columns=PROBE_TABLE_COLUMNS)
        if rows
        else pd.DataFrame(columns=PROBE_TABLE_COLUMNS)
    )
    for col in ["session_index", "n_channels", "n_rows", "n_cols", "n_stim_events", "bytes"]:
        df[col] = df[col].astype("Int64")
    csv_path = os.path.join(output_dir, "probe_table.csv")
    df.to_csv(csv_path, index=False)

    # Build assumptions summary
    sources_summary: dict[str, Any] = {}
    for source in sorted(roots.keys()):
        matching = [(ps, e, a) for (ps, e, a) in probed_results if e.source == source]
        all_probes = [p for ps, _, _ in matching for p in ps]
        sources_summary[source] = {
            "file_count": len(matching),
            "probe_count": len(all_probes),
            "total_bytes": sum(e.bytes for _, e, _ in matching),
            "has_xy_count": sum(1 for p in all_probes if p.has_xy),
            "probe_error_count": sum(1 for p in all_probes if p.probe_error is not None),
            "assumptions": [{"path": e.path, **a} for _, e, a in matching],
        }

    summary = {
        "total_files": len(probed_results),
        "total_probes": len(rows),
        "total_bytes": sum(e.bytes for _, e, _ in probed_results),
        "census_skip_s": census_skip_s,
        "census_window_len_s": census_window_len_s,
        "finalspark_session_gap_s": finalspark_session_gap_s,
        "sources": sources_summary,
    }

    json_path = os.path.join(output_dir, "probe_assumptions.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Probed %d files (%d probes) across %d sources.",
        len(probed_results),
        len(rows),
        len(roots),
    )
    for source, stats in sources_summary.items():
        logger.info(
            "Source '%s': %d files, %d probes, %d bytes, %d with coordinates, %d probe errors.",
            source,
            stats["file_count"],
            stats["probe_count"],
            stats["total_bytes"],
            stats["has_xy_count"],
            stats["probe_error_count"],
        )

    return df, summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Probe recording files for recovery census.")
    parser.add_argument(
        "--config", type=str, default="configs/recovery_census.yaml", help="Path to config YAML."
    )
    args = parser.parse_args()

    run_census_probe(args.config)
