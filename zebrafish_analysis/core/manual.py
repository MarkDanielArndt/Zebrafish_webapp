# zebrafish_analysis/core/manual.py
"""
Manual endpoint correction — shared path-computation logic.

compute_manual_length(seg_mask, point1, point2, spacing)
    Routes a smooth centerline path through the fish mask between two
    manually placed head/tail points.  Used by both the Gradio webapp and
    the 3D Slicer extension.
"""

import numpy as np


def compute_manual_length(seg_mask, point1, point2, spacing):
    """
    Compute fish length from manually placed head/tail points.

    Parameters
    ----------
    seg_mask : np.ndarray
        2-D segmentation mask (any dtype; non-zero = fish).
    point1, point2 : tuple
        (row, col) in mask coordinates.
    spacing : tuple
        (dy, dx) — physical units (µm) per pixel in mask space.

    Returns
    -------
    length_um : float
        Arc length along the routed path.
    straight_length_um : float
        Euclidean straight-line distance between the two points.
    path : np.ndarray, shape (N, 2)
        Integer (row, col) waypoints of the routed path.
    straight_line_points : tuple
        ((r0, c0), (r1, c1)) — first and last path points as tuples.
    """
    try:
        seg_mask_bin = seg_mask > 0
        dy, dx = spacing

        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)

        from scipy.ndimage import distance_transform_edt, gaussian_filter
        from skimage.graph import route_through_array

        dist_transform = distance_transform_edt(seg_mask_bin)

        if dist_transform.max() == 0:
            # Empty mask — straight-line fallback
            diff = p2 - p1
            straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
            path = np.array([p1, p2], dtype=int)
            return straight_length, straight_length, path, (
                tuple(p1.astype(int)), tuple(p2.astype(int))
            )

        # Cost map: prefer center (high dist-transform value)
        max_dist = dist_transform.max()
        cost_map = np.where(seg_mask_bin, max_dist - dist_transform + 0.1, 1e10)
        cost_map = gaussian_filter(cost_map, sigma=2.0)

        # Clamp points to mask bounds
        p1_int = np.clip(
            np.round(p1).astype(int), [0, 0],
            [seg_mask_bin.shape[0] - 1, seg_mask_bin.shape[1] - 1],
        )
        p2_int = np.clip(
            np.round(p2).astype(int), [0, 0],
            [seg_mask_bin.shape[0] - 1, seg_mask_bin.shape[1] - 1],
        )

        # Track whether clicked points were outside the mask so we can extend the path later
        p1_outside = not seg_mask_bin[p1_int[0], p1_int[1]]
        p2_outside = not seg_mask_bin[p2_int[0], p2_int[1]]
        p1_anchor = p1_int.copy()  # actual clicked position (clamped to image bounds)
        p2_anchor = p2_int.copy()

        # Snap points outside mask to nearest mask pixel for internal routing
        def _snap(pt_int, pt_float):
            if seg_mask_bin[pt_int[0], pt_int[1]]:
                return pt_int
            mask_coords = np.argwhere(seg_mask_bin)
            if len(mask_coords) == 0:
                return pt_int
            from scipy.spatial.distance import cdist
            dists = cdist([pt_float], mask_coords)[0]
            return mask_coords[np.argmin(dists)]

        p1_int = _snap(p1_int, p1)
        p2_int = _snap(p2_int, p2)

        # Route through cost map
        try:
            indices, _ = route_through_array(
                cost_map,
                start=tuple(p1_int),
                end=tuple(p2_int),
                fully_connected=True,
                geometric=True,
            )
            path = np.array(indices, dtype=int)
        except Exception as exc:
            print(f"compute_manual_length: route_through_array failed ({exc}), using straight line")
            n_points = int(np.ceil(np.linalg.norm(p2_int - p1_int))) + 1
            t = np.linspace(0, 1, n_points)
            path = np.round(
                p1_int[None, :] * (1 - t[:, None]) + p2_int[None, :] * t[:, None]
            ).astype(int)

        if len(path) < 2:
            diff = p2 - p1
            straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
            path = np.array([p1, p2], dtype=int)
            return straight_length, straight_length, path, (
                tuple(p1.astype(int)), tuple(p2.astype(int))
            )

        # Constrained Gaussian smoothing — path stays inside mask
        path = _smooth_path_constrained(path, seg_mask_bin, dist_transform, iterations=10)

        # Remove duplicate consecutive points
        if len(path) >= 2:
            keep = np.concatenate([[True], np.any(np.diff(path, axis=0) != 0, axis=1)])
            path = path[keep]

        # If clicked points were outside the mask, extend path with straight-line segments
        # from the actual clicked position to the mask boundary entry/exit point.
        if p1_outside:
            n_ext = max(2, int(np.ceil(np.linalg.norm(p1_anchor.astype(float) - p1_int.astype(float)))) + 1)
            t = np.linspace(0, 1, n_ext)[:-1]  # exclude p1_int — already path[0]
            ext = np.round(p1_anchor[None, :] * (1 - t[:, None]) + p1_int[None, :] * t[:, None]).astype(int)
            path = np.vstack([ext, path])

        if p2_outside:
            n_ext = max(2, int(np.ceil(np.linalg.norm(p2_anchor.astype(float) - p2_int.astype(float)))) + 1)
            t = np.linspace(0, 1, n_ext)[1:]  # exclude p2_int — already path[-1]
            ext = np.round(p2_int[None, :] * (1 - t[:, None]) + p2_anchor[None, :] * t[:, None]).astype(int)
            path = np.vstack([path, ext])

        # Arc length
        pf = path.astype(float)
        dxy = np.diff(pf, axis=0)
        length = float(np.sqrt((dxy[:, 0] * dy) ** 2 + (dxy[:, 1] * dx) ** 2).sum())

        # Straight-line distance
        diff = p2 - p1
        straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))

        return length, straight_length, path, (tuple(path[0]), tuple(path[-1]))

    except Exception as exc:
        import traceback
        print(f"compute_manual_length error: {exc}")
        traceback.print_exc()
        # Straight-line fallback
        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)
        dy, dx = spacing
        diff = p2 - p1
        straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
        path = np.array([p1, p2], dtype=int)
        return straight_length, straight_length, path, (
            tuple(p1.astype(int)), tuple(p2.astype(int))
        )


def _smooth_path_constrained(path, mask, dist_map, iterations=8):
    """Smooth path with Gaussian while keeping all points inside the mask."""
    if len(path) < 5:
        return path

    path_smooth = path.astype(float).copy()
    n = len(path_smooth)

    for _ in range(iterations):
        prev = path_smooth.copy()

        for i in range(1, n - 1):
            window, half_w = 7, 3
            s = max(0, i - half_w)
            e = min(n, i + half_w + 1)
            idxs = np.arange(s, e)
            weights = np.exp(-0.5 * ((idxs - i) / 2.5) ** 2)
            weights /= weights.sum()
            smoothed = (weights[:, None] * prev[s:e]).sum(axis=0)
            path_smooth[i] = 0.75 * smoothed + 0.25 * prev[i]

        # Project back to mask
        for i in range(1, n - 1):
            pi = np.clip(
                np.round(path_smooth[i]).astype(int),
                [0, 0], [mask.shape[0] - 1, mask.shape[1] - 1],
            )
            if not mask[pi[0], pi[1]]:
                found = False
                for r in range(1, 6):
                    y0, y1 = max(0, pi[0] - r), min(mask.shape[0], pi[0] + r + 1)
                    x0, x1 = max(0, pi[1] - r), min(mask.shape[1], pi[1] + r + 1)
                    local = mask[y0:y1, x0:x1]
                    if local.any():
                        lc = np.argwhere(local)
                        lc[:, 0] += y0
                        lc[:, 1] += x0
                        d = np.sum((lc - path_smooth[i]) ** 2, axis=1)
                        path_smooth[i] = lc[np.argmin(d)].astype(float)
                        found = True
                        break
                if not found:
                    path_smooth[i] = prev[i]

    path_smooth = np.round(path_smooth).astype(int)
    path_smooth[:, 0] = np.clip(path_smooth[:, 0], 0, mask.shape[0] - 1)
    path_smooth[:, 1] = np.clip(path_smooth[:, 1], 0, mask.shape[1] - 1)
    return path_smooth
