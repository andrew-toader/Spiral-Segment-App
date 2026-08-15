"""
Convert a MATLAB pXXX_spiral.mat file (saved by the original
spiral_preprocess.m script) into the pXXX_spiral.pkl format the Python
GUI (gui_app.py) expects, so previously-completed MATLAB work can be
loaded, reviewed, and continued without redoing it.

Usage:
    python mat_to_pkl.py /path/to/pXXX_spiral.mat
    python mat_to_pkl.py /path/to/pXXX_spiral.mat -o /path/to/output.pkl

What it does:
  - Loads the .mat file (tries scipy.io.loadmat for legacy v5/v7 files
    first, falls back to h5py for v7.3/HDF5-based files).
  - Expects these top-level cell-array variables, matching what the
    original MATLAB script saved: spiral_ims, spirals, import_order,
    crossing_points, template, center_template, center_spiral.
  - Converts MATLAB's 1-indexed cell arrays into Python dicts keyed
    1..N, matching gui_app.py's convention.
  - Converts center points from MATLAB's [x, y] (col, row) convention
    to this tool's (row, col) convention.
  - Prints exactly what it found before converting, so you can sanity
    check it's reading the right thing.

If your .mat file's variable names or structure don't match (e.g. if it
was saved with `save(path, '-struct', 'final_state')`, wrapping
everything inside one struct instead of flat top-level variables), this
prints what it actually found -- share that output and the converter can
be adjusted.
"""

import argparse
import os
import pickle
import sys

import numpy as np


EXPECTED_KEYS = [
    "spiral_ims", "spirals", "import_order",
    "crossing_points", "template", "center_template", "center_spiral",
]


def _load_mat(path):
    """Try scipy.io.loadmat (v5/v7) first, fall back to h5py (v7.3)."""
    try:
        import scipy.io as sio
        return sio.loadmat(path, squeeze_me=False, struct_as_record=False), "scipy"
    except NotImplementedError:
        pass
    except Exception as e:
        print(f"scipy.io.loadmat failed ({e}); trying h5py (v7.3 format)...")

    import h5py
    f = h5py.File(path, "r")
    return f, "h5py"


def _cell_to_list(cell_array):
    """Normalize a MATLAB 1xN or Nx1 cell array (loaded via scipy) into a
    flat Python list, in MATLAB's original 1..N order."""
    arr = np.asarray(cell_array)
    arr = arr.squeeze()
    if arr.ndim == 0:
        return [arr.item()]
    return list(arr.ravel())


def _unwrap_scalar_cell(item):
    """A scipy-loaded cell entry is often itself a nested 1x1/NxM array
    (e.g. a string wrapped as array(['preop'], dtype='<U6')). Unwrap down
    to the actual value."""
    while isinstance(item, np.ndarray) and item.size == 1 and item.dtype != object:
        item = item.item()
    while isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
        item = item.ravel()[0]
    return item


def convert_scipy(mat_dict, verbose=True):
    missing = [k for k in EXPECTED_KEYS if k not in mat_dict]
    if missing:
        print("Could not find these expected variables in the .mat file:", missing)
        print("Variables actually present:",
              [k for k in mat_dict.keys() if not k.startswith("__")])
        raise KeyError(
            "Missing expected variable(s) -- see printed list above. "
            "This .mat file's structure doesn't match what this converter "
            "expects; share the printed variable list and it can be adjusted."
        )

    import_order_raw = _cell_to_list(mat_dict["import_order"])
    import_order = [str(_unwrap_scalar_cell(x)).strip() for x in import_order_raw]
    n = len(import_order)

    if verbose:
        print(f"Found {n} images: {import_order}")

    def cell_dict(key, transform=None):
        items = _cell_to_list(mat_dict[key])
        if len(items) != n:
            print(f"WARNING: '{key}' has {len(items)} entries, expected {n} "
                  f"(matching import_order) -- results may be misaligned.")
        out = {}
        for i, item in enumerate(items):
            val = _unwrap_scalar_cell(item)
            if transform is not None:
                val = transform(val)
            out[i + 1] = val  # 1-indexed, matching gui_app.py's convention
        return out

    def to_center_tuple(val):
        arr = np.asarray(val).ravel().astype(float)
        if arr.size < 2:
            return None
        x, y = arr[0], arr[1]  # MATLAB pts.Position convention: [x, y] = [col, row]
        return (int(round(y)), int(round(x)))  # -> (row, col)

    def to_float_array(val):
        return np.asarray(val, dtype=np.float64)

    def to_bool_array(val):
        return np.asarray(val).astype(bool)

    state = {
        "import_order": import_order,
        "spiral_ims": cell_dict("spiral_ims", to_float_array),
        "spirals": cell_dict("spirals", to_float_array),
        "crossing_points": cell_dict("crossing_points", to_bool_array),
        "template": cell_dict("template", to_float_array),
        "center_template": cell_dict("center_template", to_center_tuple),
        "center_spiral": cell_dict("center_spiral", to_center_tuple),
    }
    return state


def convert(mat_path, verbose=True):
    loaded, backend = _load_mat(mat_path)
    if backend == "scipy":
        return convert_scipy(loaded, verbose=verbose)
    else:
        raise NotImplementedError(
            "This .mat file is in v7.3 (HDF5) format. The converter's h5py "
            "path isn't implemented yet -- if scipy.io.loadmat failed on "
            "your file, let me know and I'll add it."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", help="Path to the pXXX_spiral.mat file")
    parser.add_argument("-o", "--output", default=None,
                         help="Output .pkl path (default: same name, .pkl extension)")
    args = parser.parse_args()

    if args.output is None:
        base, _ = os.path.splitext(args.mat_path)
        args.output = base + ".pkl"

    print(f"Converting {args.mat_path} -> {args.output}")
    state = convert(args.mat_path)

    with open(args.output, "wb") as f:
        pickle.dump(state, f)

    print(f"Done. Wrote {len(state['import_order'])} images to {args.output}")


if __name__ == "__main__":
    main()
