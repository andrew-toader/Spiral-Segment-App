"""
Python translation of spiral_analysis.m, order_pts_center.m, and
uravel_spiral.m -- the per-patient spiral tremor analysis pipeline that
runs on completed segmentations (spirals/template arrays + centers +
crossing points) from a patient's *_spiral.pkl file.

Coordinate convention: everything here operates in (row, col) space
directly -- numpy's natural array-indexing convention, matching
np.argwhere's output and this whole tool's own center-point convention
(established when the segmentation GUI was built). This is mathematically
equivalent to the original MATLAB script's (x, y) = (col, row) convention:
Euclidean distance and Bresenham line-drawing are symmetric under axis
relabeling, so as long as points and centers are used consistently (which
they are here), no MATLAB-style coordinate swapping needs replicating.

Several MATLAB built-ins (pwelch, highpass, medfilt1, imbinarize) don't
have literal equivalents in Python; the approximations used here are
noted on each helper. Since the pipeline's actual output is a set of
RELATIVE percent-change-from-baseline metrics computed the same way for
every trial of a given patient, exact bit-for-bit parity with MATLAB's
internal filter/PSD implementations matters far less than internal
consistency across trials -- which these preserve.
"""

import numpy as np
from scipy import signal

# numpy.trapz was renamed to numpy.trapezoid in NumPy 2.0 and removed in
# later versions -- support either without assuming which one is installed.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ==========================================================================
# Small MATLAB-equivalent helpers
# ==========================================================================

def imbinarize_otsu(img):
    """Equivalent of MATLAB's imbinarize(img) with no extra args (Otsu's
    method). Falls back to an all-False mask for a degenerate
    (single-valued) image, where Otsu isn't defined."""
    from skimage.filters import threshold_otsu
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() == arr.min():
        return np.zeros_like(arr, dtype=bool)
    try:
        t = threshold_otsu(arr)
    except Exception:
        t = 0.5
    return arr > t


def bresenham_line(r0, c0, r1, c1):
    """Standard Bresenham line algorithm. Returns a list of (row, col)
    integer points from (r0, c0) to (r1, c1), inclusive of both ends."""
    r0, c0, r1, c1 = int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1))
    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


def matlab_medfilt1(x, n):
    """Approximate equivalent of MATLAB's medfilt1(x, n). scipy's medfilt
    requires an odd kernel size; MATLAB allows even n with a slightly
    asymmetric centered window. Rounding up to the next odd size is a
    very close approximation for this use (smoothing a once-per-spoke
    distance sequence, not a precision-critical signal)."""
    x = np.asarray(x, dtype=np.float64)
    k = n if n % 2 == 1 else n + 1
    k = min(k, len(x) if len(x) % 2 == 1 else len(x) - 1)
    k = max(k, 1)
    return signal.medfilt(x, kernel_size=k)


def matlab_pwelch(x):
    """Approximate equivalent of MATLAB's pwelch(x) with no extra
    arguments: Welch PSD with 8 segments at 50% overlap and a Hamming
    window (MATLAB's documented default -- nperseg chosen so that
    nperseg*(1 + 7*0.5) == N). For a 2D input, computes one PSD per
    COLUMN (matching MATLAB's per-column behavior for matrix input) and
    returns them stacked as columns of the output, shape (n_freqs, n_cols).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    nperseg = max(8, int(np.floor(n / 4.5)))
    nperseg = min(nperseg, n)
    noverlap = nperseg // 2

    psds = []
    for col in range(x.shape[1]):
        _, psd = signal.welch(
            x[:, col], window="hamming", nperseg=nperseg,
            noverlap=noverlap, detrend="constant", scaling="density",
        )
        psds.append(psd)
    return np.array(psds).T


def matlab_highpass(x, wpass, order=6):
    """Approximate equivalent of MATLAB's highpass(x, Wpass), where Wpass
    is a normalized cutoff frequency (fraction of Nyquist, 0-1). Not a
    bit-exact replica of MATLAB's specific minimum-order/Kaiser-window
    filter design, but serves the same purpose (remove the slow-varying
    trend, keep higher-frequency tremor-like content) via a standard
    Butterworth design with zero-phase filtering. Falls back to simple
    per-column mean removal if the sequence is too short for the
    requested filter order (avoids a hard failure on a short trial)."""
    x = np.asarray(x, dtype=np.float64)
    wpass = min(max(wpass, 1e-6), 0.99)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]

    out = np.empty_like(x)
    for col in range(x.shape[1]):
        col_data = x[:, col]
        try:
            sos = signal.butter(order, wpass, btype="highpass", output="sos")
            out[:, col] = signal.sosfiltfilt(sos, col_data)
        except ValueError:
            out[:, col] = col_data - col_data.mean()
    return out[:, 0] if one_d else out


# ==========================================================================
# Core algorithm: order_pts_center.m + uravel_spiral.m
# ==========================================================================

def order_pts_center(unordered_pts, center):
    """Python translation of order_pts_center.m: greedy nearest-neighbor
    traversal starting from `center`, always jumping to whichever
    remaining point is closest to the last one visited, until every
    point has been visited once.

    `unordered_pts` is an (N, 2) array of (row, col) points; `center` is
    a (row, col) pair. Returns an (N, 2) array of points in visit order.

    This is a GREEDY walk (not an optimal shortest path/TSP solve),
    exactly matching the original -- "the more ideal the spiral, the
    better the performance," per the MATLAB docstring.
    """
    pts = np.asarray(unordered_pts, dtype=np.float64)
    n = len(pts)
    ordered = np.empty((n, 2), dtype=np.float64)
    prev = np.asarray(center, dtype=np.float64)

    remaining = pts.copy()
    for i in range(n):
        d2 = np.sum((remaining - prev) ** 2, axis=1)
        nn = np.argmin(d2)
        ordered[i] = remaining[nn]
        prev = remaining[nn]
        remaining = np.delete(remaining, nn, axis=0)
    return ordered


def uravel_spiral(spiral_skel, pts_template_ordered):
    """Python translation of uravel_spiral.m: walk along the ordered
    template curve (from order_pts_center) as a sequence of radial
    "spokes" from the template's own near-center starting point
    (pts_template_ordered[0]), and at each spoke, find where the spiral
    skeleton crosses it -- giving spiral_dist[i] (the spiral's distance
    from center along that spoke) directly comparable to
    dist_template[i] (the template's own distance from center at that
    same spoke). This comparison, swept across every spoke, is the
    actual "unravel" -- effectively an r(spoke-index) profile for both
    curves that can be compared point-for-point.

    `spiral_skel` is a boolean 2D array. `pts_template_ordered` is an
    (N, 2) array of (row, col) points in walk order (index 0 = the
    template point closest to the true center).

    Each spoke is checked as three parallel Bresenham lines (the direct
    line, plus copies offset by (+1,+1) and (-1,-1)) so a spiral trace
    that doesn't lie exactly on the ideal straight spoke -- hand tremor,
    natural deviation -- still gets caught, matching the original.

    Returns (spiral_dist, dist_template, num_points), all length-N
    arrays; spiral_dist is median-filtered (window 10) as in the
    original.
    """
    pts_template_ordered = np.asarray(pts_template_ordered, dtype=np.float64)
    center_template = pts_template_ordered[0]
    spiral_working = np.asarray(spiral_skel, dtype=bool).copy()
    h, w = spiral_working.shape

    n = len(pts_template_ordered)
    dist_template = np.zeros(n)
    spiral_dist = np.zeros(n)
    num_points = np.zeros(n, dtype=int)

    r0, c0 = center_template

    for ii in range(n):
        r_ii, c_ii = pts_template_ordered[ii]

        lines = [
            bresenham_line(r0, c0, r_ii, c_ii),
            bresenham_line(r0 + 1, c0 + 1, r_ii + 1, c_ii + 1),
            bresenham_line(r0 - 1, c0 - 1, r_ii - 1, c_ii - 1),
        ]

        spiral_pts = []
        for line in lines:
            for (r, c) in line:
                if 0 <= r < h and 0 <= c < w and spiral_working[r, c]:
                    spiral_pts.append((r, c))
                    spiral_working[r, c] = False  # consume -- no double-count

        num_points[ii] = len(spiral_pts)
        dist_template[ii] = np.hypot(r_ii - center_template[0], c_ii - center_template[1])

        if spiral_pts:
            arr = np.asarray(spiral_pts, dtype=np.float64)
            dists = np.hypot(arr[:, 0] - center_template[0], arr[:, 1] - center_template[1])
            spiral_dist[ii] = dists.mean()
        else:
            spiral_dist[ii] = 0.0

    spiral_dist = matlab_medfilt1(spiral_dist, 10)
    return spiral_dist, dist_template, num_points


# ==========================================================================
# Per-trial feature computation (spiral_analysis.m's per-jj loop body)
# ==========================================================================

def analyze_trial(spiral_arr, template_arr, crossing_points_arr,
                   center_spiral, center_template):
    """Compute every per-trial feature from spiral_analysis.m's main loop
    for one trial. Inputs are exactly what's stored per-position in a
    patient's *_spiral.pkl: the (already ~1px, post-reskeletonize)
    spiral/template float arrays, the crossing-points boolean array, and
    the (row, col) spiral/template centers.

    Returns a dict of scalar features plus the intermediate arrays
    needed for plotting (ordered points, distance profiles, PSDs).
    """
    from core import bwskel, bwmorph_clean  # reuse the app's own skeletonizer

    # Re-skeletonize (matches the .m script's own bwskel(imbinarize(...))
    # calls -- idempotent if the array's already a clean 1px skeleton,
    # and a safety net if it isn't).
    spiral_skel = bwskel(imbinarize_otsu(spiral_arr))
    template_skel = bwmorph_clean(bwskel(imbinarize_otsu(template_arr)))

    template_pts = np.argwhere(template_skel)
    spiral_pts = np.argwhere(spiral_skel)

    pts_template_ordered = order_pts_center(template_pts, center_template)
    pts_spiral_ordered = order_pts_center(spiral_pts, center_spiral)

    # Spatial (2D) Welch PSDs of the Gaussian-blurred masks -- computed
    # for parity with the original script; not part of the final
    # headline metric, but saved/available like the original.
    from scipy.ndimage import gaussian_filter
    spiral_filt = gaussian_filter(np.asarray(spiral_arr, dtype=np.float64), sigma=0.5)
    template_filt = gaussian_filter(np.asarray(template_arr, dtype=np.float64), sigma=0.5)
    spirals_welch = matlab_pwelch(spiral_filt).sum(axis=1) + matlab_pwelch(spiral_filt.T).sum(axis=1)
    template_welch = matlab_pwelch(template_filt).sum(axis=1) + matlab_pwelch(template_filt.T).sum(axis=1)
    spiral_welch_auc = _trapz(spirals_welch)
    template_welch_auc = _trapz(template_welch)
    welch_ratio_auc = spiral_welch_auc / template_welch_auc if template_welch_auc else np.nan

    # AUC-of-sorted-detrended-coordinates features (parity only, unused
    # in the final headline metric, same as the original).
    spiral_pts_r = np.sort(spiral_pts[:, 0].astype(np.float64))
    spiral_pts_c = np.sort(spiral_pts[:, 1].astype(np.float64))
    template_pts_r = np.sort(template_pts[:, 0].astype(np.float64))
    template_pts_c = np.sort(template_pts[:, 1].astype(np.float64))
    auc_spiral = (
        _trapz(signal.detrend(spiral_pts_r)) + _trapz(signal.detrend(spiral_pts_c))
    ) / len(spiral_pts_r)
    auc_template = (
        _trapz(signal.detrend(template_pts_r)) + _trapz(signal.detrend(template_pts_c))
    ) / len(template_pts_r)

    # The core "unravel" comparison.
    spiral_dist, dist_template, num_points_per_spoke = uravel_spiral(spiral_skel, pts_template_ordered)
    num_points = float(np.median(num_points_per_spoke))

    len_fraction = 1 + abs(1 - len(pts_spiral_ordered) / (len(pts_template_ordered) * 0.75))
    num_crossings = int(np.asarray(crossing_points_arr).astype(bool).sum())

    # PSD of the highpass-filtered ordered spiral coordinates -- the
    # first of the two features behind the headline metric.
    n_sp = len(pts_spiral_ordered)
    wpass = min(20.0 / n_sp, 0.99) if n_sp > 0 else 0.5
    spiral_coord_hp = matlab_highpass(pts_spiral_ordered, wpass)
    spiral_coord_welch = matlab_pwelch(spiral_coord_hp).sum(axis=1)
    welch_coord_auc = _trapz(spiral_coord_welch)
    welch_coord_auc_comb = welch_coord_auc * len_fraction

    # PSD of the raw (unravel) distance-from-center profile -- the
    # second of the two features behind the headline metric.
    spiral_dist_welch = matlab_pwelch(spiral_dist)[:, 0]
    spiral_dist_welch = spiral_dist_welch.copy()
    spiral_dist_welch[:5] = 0
    welch_dist_auc = _trapz(spiral_dist_welch)
    welch_dist_auc_comb = welch_dist_auc * len_fraction

    return {
        "pts_spiral_ordered": pts_spiral_ordered,
        "pts_template_ordered": pts_template_ordered,
        "spiral_dist": spiral_dist,
        "dist_template": dist_template,
        "spiral_coord_welch": spiral_coord_welch,
        "spiral_dist_welch": spiral_dist_welch,
        "num_points": num_points,
        "num_crossings": num_crossings,
        "len_fraction": len_fraction,
        "auc_spiral": auc_spiral,
        "auc_template": auc_template,
        "welch_ratio_auc": welch_ratio_auc,
        "welch_coord_auc": welch_coord_auc,
        "welch_dist_auc": welch_dist_auc,
        "welch_coord_auc_comb": welch_coord_auc_comb,
        "welch_dist_auc_comb": welch_dist_auc_comb,
    }


def analyze_patient(state, baseline_index=0, trial_indices=None):
    """Run analyze_trial for a set of trials in a loaded *_spiral.pkl
    state dict, then compute the headline improvement_spiral metric for
    each relative to a chosen baseline trial.

    `trial_indices`, if given, is a list of 0-indexed positions into
    state['import_order'] to include (in that order) -- e.g. [0, -1] or
    [0, 4] for a pre/post-only comparison instead of every trial. If
    omitted, every trial is included (the original script's "all trials
    vs one baseline" behavior). `baseline_index` is an index INTO
    trial_indices (i.e. position 0 of the subset you're comparing, not
    necessarily position 0 of the full import_order) -- so a pre/post
    call with trial_indices=[0, -1] and baseline_index=0 compares the
    last trial against the first, matching a natural "pre" vs "post".

    Returns a dict with 'import_order' (names of just the included
    trials, in the given order), 'per_trial' (list of analyze_trial
    results, one per included trial), and 'improvement_spiral' (array of
    percent-change-from-baseline scores, one per included trial,
    matching the original script's min(coord, dist) combination).
    """
    full_import_order = state["import_order"]
    if trial_indices is None:
        trial_indices = list(range(len(full_import_order)))
    # Normalize negative indices (e.g. -1 for "last trial") the same way
    # Python list indexing does, before converting to the .pkl's 1-indexed
    # keys -- otherwise -1 would incorrectly map to key 0 instead of the
    # actual last trial.
    n_full = len(full_import_order)
    trial_indices = [idx % n_full for idx in trial_indices]

    import_order = [full_import_order[i] for i in trial_indices]
    n = len(import_order)

    per_trial = []
    for idx in trial_indices:
        ii = idx + 1  # 1-indexed, matches the .pkl's own convention
        result = analyze_trial(
            state["spirals"][ii], state["template"][ii], state["crossing_points"][ii],
            state["center_spiral"][ii], state["center_template"][ii],
        )
        per_trial.append(result)

    welch_coord_auc_comb = np.array([t["welch_coord_auc_comb"] for t in per_trial])
    welch_dist_auc_comb = np.array([t["welch_dist_auc_comb"] for t in per_trial])

    base_coord = welch_coord_auc_comb[baseline_index]
    base_dist = welch_dist_auc_comb[baseline_index]

    improvement_coord = (welch_coord_auc_comb - base_coord) / base_coord if base_coord else np.full(n, np.nan)
    improvement_dist = (welch_dist_auc_comb - base_dist) / base_dist if base_dist else np.full(n, np.nan)
    improvement_spiral = np.minimum(improvement_coord, improvement_dist)

    return {
        "import_order": import_order,
        "trial_indices": trial_indices,
        "per_trial": per_trial,
        "baseline_index": baseline_index,
        "welch_coord_auc_comb": welch_coord_auc_comb,
        "welch_dist_auc_comb": welch_dist_auc_comb,
        "improvement_welch_coord_combi": improvement_coord,
        "improvement_welch_dist_combi": improvement_dist,
        "improvement_spiral": improvement_spiral,
    }
