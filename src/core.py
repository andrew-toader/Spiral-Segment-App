"""
Pure image-processing core for the spiral segmentation pipeline. No GUI or
matplotlib dependency here on purpose -- this module is shared by the GUI
app and can be unit-tested standalone.

Segment label convention used throughout:
    SPIRAL  = 0  (default / unhighlighted)
    TEMPLATE = 1
    REMOVE  = 2  (stray -- excluded from both reconstructions)
    BOTH    = 3  (counts as part of both spiral and template)
"""

import os

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.filters import threshold_local
from skimage.measure import label as cc_label
from skimage.morphology import dilation, disk, footprint_rectangle, skeletonize
from skimage.transform import resize


SPIRAL, TEMPLATE, REMOVE, BOTH = 0, 1, 2, 3

# See notes from the earlier translation: the original MATLAB code inverted
# the dark-foreground adaptive mask. Testing against real images showed
# INVERT=True was tracing the background, not the ink -- so this defaults
# to False now. Flip it back if your images are polarity-reversed
# (e.g. white pen on dark paper).
INVERT_ADAPTIVE_MASK = False


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def im2double(img):
    """Equivalent of MATLAB's im2double: scale integer images to [0, 1] floats."""
    if np.issubdtype(img.dtype, np.floating):
        return img.astype(np.float64)
    info = np.iinfo(img.dtype)
    return img.astype(np.float64) / info.max


def load_image(fpath, fname):
    """Equivalent of im2double(rgb2gray(imread(...))). Handles RGBA PNGs.
    `fname` should be the actual on-disk basename (no extension) -- callers
    that use a separate short "index name" resolve it to the real basename
    first (see index_name_from_filename / gui_app.py's basename_by_index)."""
    img = imread(os.path.join(fpath, fname + ".png"))
    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]
        return rgb2gray(img).astype(np.float64)
    return im2double(img)


def index_name_from_filename(basename, delimiter="_"):
    """Extract a short 'index name' from a real on-disk filename basename
    by taking everything before the first occurrence of `delimiter`
    (default: '_'). E.g. 'preop_spiral' -> 'preop'. Pass delimiter=None
    (or a delimiter not present in the name) to use the full basename
    unchanged.

    This is the shared naming rule used both when scanning a folder in
    the GUI and when resolving a MATLAB-converted .pkl's short
    "variable names" against real files (mat_to_pkl.py)."""
    if delimiter and delimiter in basename:
        return basename.split(delimiter, 1)[0]
    return basename


def build_index_name_map(basenames, delimiter="_"):
    """Given a list of real on-disk basenames, return
    (index_names, basename_by_index) where index_names is basenames
    transformed via index_name_from_filename, in the same order, and
    basename_by_index maps each index name back to its real basename.

    Collisions (two different basenames reducing to the same index name)
    fall back to keeping the full basename for every entry in that
    colliding group, so nothing is silently lost or overwritten.
    """
    proposed = {}
    for b in basenames:
        name = index_name_from_filename(b, delimiter)
        proposed.setdefault(name, []).append(b)

    basename_by_index = {}
    index_names = []
    for b in basenames:
        name = index_name_from_filename(b, delimiter)
        if len(proposed[name]) > 1:
            name = b  # collision -- use the full basename instead
        basename_by_index[name] = b
        index_names.append(name)
    return index_names, basename_by_index


def resolve_real_basename(folder, index_name, fallback_basename=None, delimiter="_"):
    """Return a basename that actually exists as `<basename>.png` in
    `folder`, self-healing a stale/wrong basename_by_index entry rather
    than failing.

    Tries `fallback_basename` first (the stored mapping). If that file
    doesn't exist -- e.g. a .pkl converted from .mat without knowing the
    real filenames, so the mapping fell back to an identity guess --
    tries, in order:

      1. `<index_name>_spiral.png` exactly -- the original MATLAB
         convention this whole tool is built around.
      2. Any `<index_name>_*spiral*.png` (covers naming variants like a
         "_spiral2" repeat attempt) -- this explicitly prefers spiral
         files over other same-prefix, same-timepoint files for a
         DIFFERENT task (e.g. "<index_name>_line.png"), which the
         generic delimiter-splitting rule alone can't disambiguate: both
         "preop_line" and "preop_spiral" reduce to "preop", so without
         this preference the two would collide and neither could be
         matched.
      3. The generic build_index_name_map rule, as a last resort for
         folders that don't follow the "_spiral" convention at all.

    Falls back to `fallback_basename` (or `index_name`) unchanged if
    nothing better is found, so the caller still gets a sensible error
    rather than this silently swallowing a genuinely missing file.
    """
    candidate = fallback_basename or index_name
    if os.path.exists(os.path.join(folder, candidate + ".png")):
        return candidate

    try:
        real_basenames = [
            os.path.splitext(f)[0] for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ]
    except OSError:
        return candidate

    exact_spiral = f"{index_name}_spiral"
    if exact_spiral in real_basenames:
        return exact_spiral

    prefix = f"{index_name}_"
    spiral_candidates = sorted(
        b for b in real_basenames
        if b.startswith(prefix) and "spiral" in b[len(prefix):].lower()
    )
    if spiral_candidates:
        return spiral_candidates[0]

    _, derived_map = build_index_name_map(real_basenames, delimiter=delimiter)
    if index_name in derived_map:
        return derived_map[index_name]
    return candidate


# --------------------------------------------------------------------------
# Binarization
# --------------------------------------------------------------------------

def adaptive_binarize(img, sensitivity=0.5, foreground="dark", block_size=None):
    """Approximate equivalent of MATLAB's
    imbinarize(img, 'adaptive', 'ForegroundPolarity', fg, 'Sensitivity', s).
    """
    if block_size is None:
        block_size = max(15, (min(img.shape) // 8) | 1)
    elif block_size % 2 == 0:
        block_size += 1

    local_thresh = threshold_local(img, block_size=block_size, method="gaussian")
    bias = (sensitivity - 0.5) * 0.3

    if foreground == "dark":
        bw = img < (local_thresh + bias)
    else:
        bw = img > (local_thresh - bias)
    return bw


# --------------------------------------------------------------------------
# bwmorph operations (single-pass, matching MATLAB's default n=1)
# --------------------------------------------------------------------------

def _neighbors(bw):
    P = np.pad(bw, 1, mode="constant", constant_values=False)
    N = P[0:-2, 1:-1]
    S = P[2:, 1:-1]
    E = P[1:-1, 2:]
    W = P[1:-1, 0:-2]
    NE = P[0:-2, 2:]
    NW = P[0:-2, 0:-2]
    SE = P[2:, 2:]
    SW = P[2:, 0:-2]
    return N, NE, E, SE, S, SW, W, NW


def bwmorph_clean(bw):
    N, NE, E, SE, S, SW, W, NW = _neighbors(bw)
    has_neighbor = N | NE | E | SE | S | SW | W | NW
    return bw & has_neighbor


def bwmorph_fill(bw):
    N, NE, E, SE, S, SW, W, NW = _neighbors(bw)
    all_neighbors_fg = N & NE & E & SE & S & SW & W & NW
    return bw | (~bw & all_neighbors_fg)


def bwmorph_bridge(bw):
    N, NE, E, SE, S, SW, W, NW = _neighbors(bw)
    ns = N & S
    ew = E & W
    ne_sw = NE & SW
    nw_se = NW & SE
    count = N.astype(np.uint8) + NE + E + SE + S + SW + W + NW
    opposite_pair = ns | ew | ne_sw | nw_se
    fire = (~bw) & (count == 2) & opposite_pair
    return bw | fire


def bwmorph_diag(bw):
    N, NE, E, SE, S, SW, W, NW = _neighbors(bw)
    nw_case = N & W & ~NW
    ne_case = N & E & ~NE
    sw_case = S & W & ~SW
    se_case = S & E & ~SE
    fire = (~bw) & (nw_case | ne_case | sw_case | se_case)
    return bw | fire


def bwmorph_hbreak(bw):
    N, NE, E, SE, S, SW, W, NW = _neighbors(bw)
    is_h_bridge = bw & N & NE & NW & S & SE & SW & (~E) & (~W)
    return bw & (~is_h_bridge)


def bwskel(bw):
    return skeletonize(bw)


def branch_points(skel):
    N, NE, E, SE, S, SW, W, NW = (a.astype(np.int8) for a in _neighbors(skel))
    ring = [N, NE, E, SE, S, SW, W, NW]
    cn = np.zeros_like(N)
    for i in range(8):
        cn += np.abs(ring[i] - ring[(i + 1) % 8])
    cn = cn // 2
    return skel & (cn >= 3)


# --------------------------------------------------------------------------
# Segment extraction
# --------------------------------------------------------------------------

def label_segments(bw):
    """List of (N, 2) arrays of (row, col) pixel coords, one per 8-connected
    component. Stands in for MATLAB's bwboundaries."""
    labels = cc_label(bw, connectivity=2)
    n = labels.max()
    return [np.argwhere(labels == k) for k in range(1, n + 1)]


def nearest_segment_index(segments, row, col):
    if not segments:
        return None
    best_idx, best_dist = None, np.inf
    for idx, coords in enumerate(segments):
        d = np.min((coords[:, 0] - row) ** 2 + (coords[:, 1] - col) ** 2)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def nearest_point_on_mask(mask, row, col):
    """(row, col) of the True pixel in `mask` closest to the given point.
    Used to snap a clicked center point onto the actual spiral/template
    curve, rather than wherever the click happened to land. Returns None
    if the mask has no True pixels."""
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    dists = (pts[:, 0] - row) ** 2 + (pts[:, 1] - col) ** 2
    idx = np.argmin(dists)
    return int(pts[idx, 0]), int(pts[idx, 1])


def segments_in_box(segments, row_min, row_max, col_min, col_max):
    """Indices of segments that have at least one point inside the box."""
    out = []
    for idx, coords in enumerate(segments):
        in_box = (
            (coords[:, 0] >= row_min)
            & (coords[:, 0] <= row_max)
            & (coords[:, 1] >= col_min)
            & (coords[:, 1] <= col_max)
        )
        if in_box.any():
            out.append(idx)
    return out


def segments_near_path(segments, path_points, threshold):
    """Indices of segments with at least one point within `threshold`
    pixels of the freehand path (an (M, 2) array of (row, col) points
    sampled along the drawn stroke). Used by the freehand/lasso "marker"
    selection tool -- this is proximity to the drawn line itself, not
    containment inside a closed loop."""
    from scipy.spatial import cKDTree

    if len(path_points) == 0:
        return []
    tree = cKDTree(path_points)
    out = []
    for idx, coords in enumerate(segments):
        dists, _ = tree.query(coords, k=1)
        if np.min(dists) <= threshold:
            out.append(idx)
    return out


def split_segment(coords, split_row, split_col, radius=2):
    """Manually split one segment's pixels into separate connected
    sub-segments by removing a small neighborhood around (split_row,
    split_col) and re-labeling what's left.

    This exists for cases automatic branch-point detection can't catch --
    most notably a *tangential* crossing (two curves just touching/kissing
    rather than crossing at a clear X). At a tangential contact the
    skeleton often has no distinguishing local pixel pattern at all (it
    just looks like one smooth line passing through), so no amount of
    tuning the crossing-number branch detector can find it reliably --
    the fix has to be "let the person who can see it's a crossing mark
    it themselves."

    Returns a list of (N, 2) coordinate arrays -- normally 2 pieces, but
    could be 1 (nothing actually separated -- e.g. clicked right at an
    endpoint) or more (if the removed neighborhood happened to sit at a
    point where more than 2 branches met).
    """
    rows, cols = coords[:, 0], coords[:, 1]
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    h, w = int(r1 - r0) + 1, int(c1 - c0) + 1

    local = np.zeros((h, w), dtype=bool)
    local[rows - r0, cols - c0] = True

    sr, sc = split_row - r0, split_col - c0
    rr, cc_grid = np.ogrid[:h, :w]
    remove_mask = (rr - sr) ** 2 + (cc_grid - sc) ** 2 <= radius ** 2
    local[remove_mask] = False

    sub_labels = cc_label(local, connectivity=2)
    n = sub_labels.max()
    pieces = []
    for k in range(1, n + 1):
        local_coords = np.argwhere(sub_labels == k)
        if len(local_coords) == 0:
            continue
        global_coords = local_coords + np.array([r0, c0])
        pieces.append(global_coords)
    return pieces if pieces else [coords]  # fall back to unsplit if nothing survived


def prep_mask_and_segments(img):
    bw = adaptive_binarize(img, sensitivity=0.5, foreground="dark")
    mask = ~bw if INVERT_ADAPTIVE_MASK else bw

    mask = bwmorph_bridge(mask)
    mask = bwmorph_diag(mask)
    mask = bwmorph_fill(mask)
    mask = bwmorph_hbreak(mask)
    mask = bwmorph_clean(mask)
    mask_skel = bwskel(mask)

    crossing_points = branch_points(mask_skel)
    crossing_points_thick = dilation(crossing_points, footprint_rectangle((3, 3)))

    segment_source = mask_skel & ~crossing_points_thick
    segments = label_segments(segment_source)

    return mask_skel, crossing_points, crossing_points_thick, segments


# --------------------------------------------------------------------------
# Geometric auto-classification (circle / line fitting)
# --------------------------------------------------------------------------

def fit_circle(points):
    """Algebraic (Kasa) circle fit. `points` is (N, 2) of (row, col).
    Returns (center_row, center_col, radius, rms_residual) or None."""
    a = points[:, 0].astype(np.float64)
    b = points[:, 1].astype(np.float64)
    am, bm = a.mean(), b.mean()
    u, v = a - am, b - bm

    Suu, Svv, Suv = np.sum(u * u), np.sum(v * v), np.sum(u * v)
    Suuu, Svvv = np.sum(u ** 3), np.sum(v ** 3)
    Suvv, Svuu = np.sum(u * v * v), np.sum(v * u * u)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    Bv = np.array([(Suuu + Suvv) / 2.0, (Svvv + Svuu) / 2.0])

    try:
        uc, vc = np.linalg.solve(A, Bv)
    except np.linalg.LinAlgError:
        return None

    center_row, center_col = am + uc, bm + vc
    r = np.sqrt(uc ** 2 + vc ** 2 + (Suu + Svv) / len(points))
    dists = np.hypot(a - center_row, b - center_col)
    resid = np.sqrt(np.mean((dists - r) ** 2))
    return center_row, center_col, r, resid


def fit_line(points):
    """PCA line fit. Returns (point_on_line, direction, perp_rms_residual,
    straightness) where straightness in [0, 1], 1 = perfectly straight."""
    a = points[:, 0].astype(np.float64)
    b = points[:, 1].astype(np.float64)
    am, bm = a.mean(), b.mean()
    u, v = a - am, b - bm
    n = len(points)

    cov = np.array([[np.sum(u * u), np.sum(u * v)],
                     [np.sum(u * v), np.sum(v * v)]]) / max(n, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_min, lam_max = eigvals[0], eigvals[1]
    straightness = 1.0 - (lam_min / lam_max if lam_max > 1e-9 else 1.0)
    direction = eigvecs[:, 1]
    resid = np.sqrt(max(lam_min, 0.0))
    return (am, bm), direction, resid, straightness


def auto_classify_segments(
    segments,
    image_shape,
    circle_rel_thresh=0.03,
    circle_center_tol_frac=0.15,
    line_straightness_thresh=0.995,
    line_center_tol_frac=0.05,
    min_points=5,
):
    """Classify each segment as SPIRAL (default) or TEMPLATE, based on how
    well it fits a circle or line centered near the drawing's centroid.
    Returns an int array of labels, same length as `segments`."""
    labels = np.full(len(segments), SPIRAL, dtype=int)
    if not segments:
        return labels

    all_points = np.concatenate(segments, axis=0)
    centroid_row, centroid_col = all_points[:, 0].mean(), all_points[:, 1].mean()
    diag = np.hypot(*image_shape)

    for i, coords in enumerate(segments):
        if len(coords) < min_points:
            continue

        fit = fit_circle(coords)
        if fit is not None:
            cr, cc, r, resid = fit
            if r > 1e-6 and (resid / r) < circle_rel_thresh:
                center_dist = np.hypot(cr - centroid_row, cc - centroid_col)
                if center_dist < circle_center_tol_frac * r:
                    labels[i] = TEMPLATE
                    continue

        point_on_line, direction, resid, straightness = fit_line(coords)
        seg_extent = np.hypot(
            coords[:, 0].max() - coords[:, 0].min(),
            coords[:, 1].max() - coords[:, 1].min(),
        )
        if straightness > line_straightness_thresh and seg_extent > 0:
            v = np.array([centroid_row, centroid_col]) - np.array(point_on_line)
            perp_dist = abs(v[0] * direction[1] - v[1] * direction[0])
            if perp_dist < line_center_tol_frac * diag and resid < 0.03 * seg_extent:
                labels[i] = TEMPLATE
                continue

    return labels


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------

def paint_segments(shape, segments, indices):
    out = np.zeros(shape, dtype=bool)
    for idx in indices:
        coords = segments[idx]
        out[coords[:, 0], coords[:, 1]] = True
    return out


def reconstruct_images(shape, segments, labels, crossing_points):
    """Build the spiral and template binary images from current per-segment
    labels, adding crossing points back in and cleaning isolated pixels."""
    spiral_idx = [i for i, lab in enumerate(labels) if lab in (SPIRAL, BOTH)]
    template_idx = [i for i, lab in enumerate(labels) if lab in (TEMPLATE, BOTH)]

    spiral = paint_segments(shape, segments, spiral_idx)
    template = paint_segments(shape, segments, template_idx)

    spiral = bwmorph_clean(spiral | crossing_points)
    template = bwmorph_clean(template | crossing_points)

    return spiral.astype(np.float64), template.astype(np.float64)


def resize_result(arr, output_size):
    if output_size == "keep":
        return arr.astype(np.float64)
    return resize(
        arr.astype(np.float64),
        output_size,
        order=3,
        mode="edge",
        anti_aliasing=True,
        preserve_range=True,
    )


def resize_binary_curve(mask, output_size, dilate_radius=2):
    """Resize a thin (often 1px-wide) binary curve without breaking it up.

    Naively resizing a 1px skeleton with anti-aliased interpolation and
    re-thresholding is fragile: interpolation blur pushes some pixels
    below the 0.5 cutoff, shattering the curve into disconnected dashes
    (confirmed empirically: a single connected test curve fragmented into
    136 pieces this way). Dilating the curve first gives the resize step
    enough width margin to survive blur + thresholding intact, then we
    skeletonize back down to a clean 1px curve at the new resolution.

    `mask` should be boolean. Returns a float64 0/1 array.
    """
    thick = dilation(mask, disk(dilate_radius))

    if output_size == "keep":
        resized_bw = thick
    else:
        resized = resize(
            thick.astype(np.float64),
            output_size,
            order=3,
            mode="edge",
            anti_aliasing=True,
            preserve_range=True,
        )
        resized_bw = resized > 0.5

    skel = skeletonize(resized_bw)
    skel = bwmorph_clean(skel)
    return skel.astype(np.float64)


def match_segments_to_saved(segments, final_spiral, final_template, orig_shape,
                             threshold=0.45, dilate_radius=2):
    """Best-effort reconstruction of per-segment labels for the "Edit"
    action: given freshly-recomputed segments (full-resolution pixel
    coordinates) and the previously SAVED, resized spiral/template masks,
    guess each segment's original label by checking how much of it
    overlaps the saved masks once scaled into the same coordinate space.

    This is inherently approximate, not exact recovery -- the saved masks
    went through a dilate -> resize -> skeletonize round trip when they
    were written, so positions aren't pixel-identical to the original
    segments even for a segment that was clearly labeled spiral/template.
    A small dilation tolerance and a fractional (not exact) overlap
    threshold account for that; this is meant to get you close, not
    perfect, so you're adjusting a few segments instead of starting from
    scratch.
    """
    oh, ow = orig_shape
    th, tw = final_spiral.shape

    spiral_mask = dilation(final_spiral > 0.5, disk(dilate_radius))
    template_mask = dilation(final_template > 0.5, disk(dilate_radius))

    labels = np.full(len(segments), REMOVE, dtype=int)
    for i, coords in enumerate(segments):
        if len(coords) == 0:
            continue
        scaled_rows = np.clip((coords[:, 0].astype(np.float64) * th / oh).astype(int), 0, th - 1)
        scaled_cols = np.clip((coords[:, 1].astype(np.float64) * tw / ow).astype(int), 0, tw - 1)

        spiral_frac = spiral_mask[scaled_rows, scaled_cols].mean()
        template_frac = template_mask[scaled_rows, scaled_cols].mean()

        is_spiral = spiral_frac >= threshold
        is_template = template_frac >= threshold

        if is_spiral and is_template:
            labels[i] = BOTH
        elif is_spiral:
            labels[i] = SPIRAL
        elif is_template:
            labels[i] = TEMPLATE
    return labels
