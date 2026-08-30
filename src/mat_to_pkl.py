"""
Convert a MATLAB pXXX_spiral.mat file (saved by the original
spiral_preprocess.m script) into the pXXX_spiral.pkl format the Python
GUI (gui_app.py) expects, so previously-completed MATLAB work can be
loaded, reviewed, and continued without redoing it.

Usage:
    python mat_to_pkl.py /path/to/pXXX_spiral.mat
    python mat_to_pkl.py /path/to/pXXX_spiral.mat -o /path/to/output.pkl

    Batch mode -- convert every patient in a data folder at once:
    python mat_to_pkl.py /path/to/data_folder
    (looks inside each immediate subfolder of data_folder for a
    "*_spiral.mat" file; converts every one found, writing each
    resulting .pkl alongside its source .mat. If a patient's subfolder
    has its own "spiral" subfolder of .png images, it's used
    automatically to resolve real filenames for that patient -- same as
    passing --images-dir in single-file mode. One patient failing
    doesn't stop the rest.)

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
  - Re-skeletonizes the spiral/template arrays to genuine 1px-wide
    binary by default (MATLAB's saved values are a continuous-valued,
    antialiased imresize output that was never re-binarized -- lines
    several pixels wide/faded rather than a hard single-pixel edge;
    pass --no-reskeletonize to keep MATLAB's exact original values
    instead).
  - Prints exactly what it found before converting, so you can sanity
    check it's reading the right thing.

If your .mat file's variable names or structure don't match (e.g. if it
was saved with `save(path, '-struct', 'final_state')`, wrapping
everything inside one struct instead of flat top-level variables), this
prints what it actually found -- share that output and the converter can
be adjusted.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

import core


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


def resolve_basename_by_index(import_order, images_dir, delimiter="_", verbose=True):
    """Match each import_order entry (a short 'index name' from the .mat
    file, e.g. 'preop') against real .png files in images_dir, using the
    EXACT SAME resolver the live GUI uses at runtime
    (core.resolve_real_basename): prefers a "<name>_spiral.png" match,
    then any "<name>_*spiral*" variant (so a same-prefix sibling file for
    a different task, like "<name>_line.png", doesn't get picked by
    mistake), before falling back to the generic delimiter-splitting
    rule. Sharing this one function means the converter and the GUI can
    never disagree about how to resolve a name. Returns
    {index_name: real_basename}; anything that can't be resolved keeps
    an identity mapping and is reported rather than guessed."""
    basename_by_index = {}
    problems = []
    for name in import_order:
        resolved = core.resolve_real_basename(
            images_dir, name, fallback_basename=name, delimiter=delimiter
        )
        basename_by_index[name] = resolved
        if resolved == name and not os.path.exists(os.path.join(images_dir, name + ".png")):
            problems.append(name)

    if verbose:
        changed = sum(1 for k, v in basename_by_index.items() if k != v)
        print(f"Matched against real files in {images_dir} "
              f"(delimiter={delimiter!r}): {changed} resolved to a "
              f"different real filename, {len(import_order) - changed} "
              f"unchanged.")
    if problems:
        print(f"WARNING: couldn't find a matching .png for: {problems} "
              f"in {images_dir} -- left as identity mapping, double check these.")
    return basename_by_index


def reskeletonize_array(arr, threshold_frac=0.15, dilate_radius=2):
    """Convert a continuous-valued array (e.g. MATLAB's antialiased
    imresize output, which was never re-binarized/re-skeletonized after
    resizing) into a genuine 1px-wide binary skeleton, matching the
    format the live GUI produces (core.resize_binary_curve).

    Thresholding a continuous antialiased line directly at 50% of its
    max value and skeletonizing is fragile: wherever the antialiasing
    blur pushed a pixel below the cutoff, the thresholded mask breaks
    into scattered fragments, and skeletonizing those separately just
    produces a field of disconnected dashes/dots instead of a
    continuous curve (confirmed: a fully-connected test curve came out
    as 83+ fragments this way).

    Two things fix this, and empirically work better together than
    either alone: (1) a LOWER threshold (default 15% of the array's own
    max, not a fixed 0.5) keeps more of the antialiased tail as
    foreground, since much of the "missing" signal at a fragment gap is
    faint-but-present rather than truly absent; (2) a small dilation
    before thinning bridges whatever gap remains. Tested against both a
    tightly-wound and loosely-wound synthetic spiral: threshold-only
    (no dilation) fixed the tight case but never fully reconnected the
    loose one even at very low thresholds, while this combination
    achieved full connectivity on both, using less dilation (and
    therefore less risk of fusing separate windings together) than
    dilation alone needed.
    """
    from skimage.morphology import dilation, disk
    arr = np.asarray(arr)
    max_val = arr.max()
    threshold = threshold_frac * max_val if max_val > 0 else 0.5
    bw = arr > threshold
    if dilate_radius > 0:
        bw = dilation(bw, disk(dilate_radius))
    skel = core.bwskel(bw)
    skel = core.bwmorph_clean(skel)
    return skel.astype(np.float64)


def convert(mat_path, verbose=True, images_dir=None, delimiter="_", reskeletonize=True,
            reskel_dilate_radius=2, reskel_threshold_frac=0.15):
    loaded, backend = _load_mat(mat_path)
    if backend == "scipy":
        state = convert_scipy(loaded, verbose=verbose)
    else:
        raise NotImplementedError(
            "This .mat file is in v7.3 (HDF5) format. The converter's h5py "
            "path isn't implemented yet -- if scipy.io.loadmat failed on "
            "your file, let me know and I'll add it."
        )

    if images_dir is not None:
        state["basename_by_index"] = resolve_basename_by_index(
            state["import_order"], images_dir, delimiter=delimiter, verbose=verbose
        )
    else:
        # No real folder to check against -- identity mapping. If the real
        # files turn out to differ (e.g. 'preop' vs 'preop_spiral.png'),
        # rerun with --images-dir to resolve it properly, or the GUI's
        # own loader will still error clearly rather than silently fail.
        state["basename_by_index"] = {name: name for name in state["import_order"]}

    if reskeletonize:
        from skimage.measure import label as cc_label
        if verbose:
            print("Re-skeletonizing spiral/template arrays to genuine "
                  "1px-wide binary (matching the live GUI's output format)...")
        for key in ("spirals", "template"):
            for ii, arr in state[key].items():
                result = reskeletonize_array(
                    arr, threshold_frac=reskel_threshold_frac,
                    dilate_radius=reskel_dilate_radius,
                )
                state[key][ii] = result
                if verbose:
                    n_fragments = cc_label(result > 0.5, connectivity=2).max()
                    name = state["import_order"][ii - 1]
                    if n_fragments > 1:
                        print(f"  WARNING: {key}[{name}] came out as "
                              f"{n_fragments} disconnected pieces, not 1 -- "
                              f"the curve may have real gaps too close "
                              f"together to safely auto-reconnect. Check "
                              f"this one visually; try a lower "
                              f"--reskel-threshold or a larger "
                              f"--reskel-dilate-radius if it's clearly "
                              f"still broken, but watch for nearby windings "
                              f"getting incorrectly fused together instead.")

    # Permanent, position-keyed record of the exact real .png each image
    # came from -- same field the live GUI populates when saving, built
    # here from the same resolved mapping so it's consistent no matter
    # which tool wrote the .pkl.
    state["source_filenames"] = {
        i + 1: state["basename_by_index"][name] + ".png"
        for i, name in enumerate(state["import_order"])
    }
    return state


def batch_convert(data_folder, verbose=True, delimiter="_", reskeletonize=True,
                   reskel_dilate_radius=2, reskel_threshold_frac=0.15):
    """Convert every '*_spiral.mat' file found in data_folder's immediate
    subfolders (e.g. data_folder/pXXX/pXXX_spiral.mat) -- each converted
    .pkl is written alongside its source .mat, in that same subfolder.

    Assumes the same pXXX/spiral/*.png layout used throughout this
    project: if a patient's subfolder has its own 'spiral' subfolder,
    it's used automatically as --images-dir for that one patient (so
    real filenames still get resolved without needing to pass anything
    per-patient); if it doesn't, that patient still converts, just
    without image-name resolution.

    One patient failing doesn't stop the rest -- each is attempted
    independently and a summary is printed at the end.
    """
    try:
        subfolders = sorted(
            d for d in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, d))
        )
    except OSError as e:
        print(f"Couldn't read {data_folder}: {e}")
        return

    found = []
    for name in subfolders:
        sub_path = os.path.join(data_folder, name)
        matches = sorted(glob.glob(os.path.join(sub_path, "*_spiral.mat")))
        if matches:
            found.append((name, matches[0]))

    if not found:
        print(f"No '*_spiral.mat' files found in any subfolder of {data_folder}.")
        return

    print(f"Found {len(found)} patient(s) with a *_spiral.mat file: {[n for n, _ in found]}")

    succeeded, failed = [], []
    for name, mat_path in found:
        sub_path = os.path.dirname(mat_path)
        images_dir = os.path.join(sub_path, "spiral")
        if not os.path.isdir(images_dir):
            images_dir = None
            if verbose:
                print(f"  ({name}: no 'spiral' subfolder found -- converting "
                      f"without image-name resolution)")

        output_path = os.path.splitext(mat_path)[0] + ".pkl"
        print(f"\n[{name}] Converting {mat_path} -> {output_path}")
        try:
            state = convert(
                mat_path, verbose=verbose, images_dir=images_dir, delimiter=delimiter,
                reskeletonize=reskeletonize, reskel_dilate_radius=reskel_dilate_radius,
                reskel_threshold_frac=reskel_threshold_frac,
            )
            with open(output_path, "wb") as f:
                pickle.dump(state, f)
            print(f"[{name}] Done -- wrote {len(state['import_order'])} images to {output_path}")
            succeeded.append(name)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            failed.append((name, str(e)))

    print(f"\n=== Batch conversion complete: {len(succeeded)} succeeded, {len(failed)} failed ===")
    if succeeded:
        print("Succeeded:", succeeded)
    if failed:
        print("Failed:")
        for name, err in failed:
            print(f"  {name}: {err}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mat_path",
        help="Path to a single pXXX_spiral.mat file, OR a data folder "
             "containing patient subfolders (each with their own "
             "*_spiral.mat) -- passing a folder automatically converts "
             "every one found in it."
    )
    parser.add_argument("-o", "--output", default=None,
                         help="Output .pkl path (single-file mode only -- "
                              "ignored in batch/folder mode, where each "
                              "output is written alongside its source .mat)")
    parser.add_argument(
        "--images-dir", default=None,
        help="Path to the folder of actual .png images (e.g. pXXX/spiral). "
             "If given, import_order entries (short index names like "
             "'preop') are matched against real on-disk filenames (like "
             "'preop_spiral.png') so the GUI can find them. Recommended. "
             "Single-file mode only -- in batch/folder mode, each "
             "patient's own 'spiral' subfolder is used automatically."
    )
    parser.add_argument(
        "--delimiter", default="_",
        help="Character that separates the short index name from the rest "
             "of the real filename (default: '_', e.g. "
             "'preop_spiral.png' -> index name 'preop'). Only matters "
             "together with --images-dir."
    )
    parser.add_argument(
        "--no-reskeletonize", action="store_true",
        help="Skip re-skeletonizing the spiral/template arrays, keeping "
             "them exactly as MATLAB saved them (a continuous-valued, "
             "antialiased imresize output, never re-binarized -- lines "
             "several pixels wide/faded rather than a hard 1px edge). "
             "By default (i.e. without this flag) the converter always "
             "re-skeletonizes to genuine 1px-wide binary, matching the "
             "live GUI's output format -- only pass this if you "
             "specifically need pixel-for-pixel parity with the original "
             ".mat values instead."
    )
    parser.add_argument(
        "--reskel-threshold", type=float, default=0.15,
        help="Threshold, as a fraction of each array's own max value, "
             "used before re-thinning during re-skeletonization "
             "(default: 0.15, i.e. 15%%). A lower threshold keeps more "
             "of the antialiased tail as foreground, which reconnects "
             "more of a fragmented curve on its own than a naive 50%% "
             "cutoff would."
    )
    parser.add_argument(
        "--reskel-dilate-radius", type=int, default=2,
        help="How much to dilate (in pixels) before re-thinning during "
             "re-skeletonization (default: 2), on top of the threshold "
             "above -- the two together reconnect gaps a naive 50%% "
             "threshold alone would leave fragmented. Too small and "
             "fragments won't fully reconnect (the converter warns you "
             "per-image if this happens); too large and CLOSE-BUT-"
             "SEPARATE windings of a tightly-wound spiral can get fused "
             "together instead, which is worse because it's not visually "
             "obvious. If you see a fragmentation warning, check that "
             "image visually before raising this."
    )
    args = parser.parse_args()

    if os.path.isdir(args.mat_path):
        batch_convert(
            args.mat_path, delimiter=args.delimiter,
            reskeletonize=not args.no_reskeletonize,
            reskel_dilate_radius=args.reskel_dilate_radius,
            reskel_threshold_frac=args.reskel_threshold,
        )
        return

    if args.output is None:
        base, _ = os.path.splitext(args.mat_path)
        args.output = base + ".pkl"

    print(f"Converting {args.mat_path} -> {args.output}")
    state = convert(args.mat_path, images_dir=args.images_dir, delimiter=args.delimiter,
                     reskeletonize=not args.no_reskeletonize,
                     reskel_dilate_radius=args.reskel_dilate_radius,
                     reskel_threshold_frac=args.reskel_threshold)

    with open(args.output, "wb") as f:
        pickle.dump(state, f)

    print(f"Done. Wrote {len(state['import_order'])} images to {args.output}")


if __name__ == "__main__":
    main()
