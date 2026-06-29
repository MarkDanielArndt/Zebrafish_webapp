import numpy as np
import cv2
import os
from segmentation_models_pytorch import Unet
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import binary_erosion, convolve, distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from skimage.morphology import medial_axis
from skimage.graph import MCP_Geometric
import torch.nn as nn
import timm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import hf_hub_download

def compute_eye_metrics(mask_eye, mask_fish=None, spacing=(1.0, 1.0)):
    """
    Compute eye mask, centroid, physical area, and physical diameter from an eye mask.

    Args:
        mask_eye: 2D eye mask/probability map.
        mask_fish: optional 2D fish mask to constrain eye pixels to fish body.
        spacing: (dy, dx) physical spacing per pixel.

    Returns:
        dict with keys:
            eye_mask: bool array (same shape as input)
            eye_centroid: np.array([row, col]) or None
            eye_area: float (in spacing units squared)
            eye_diameter: float (in spacing units)
            eye_diameter_points: tuple((r1,c1),(r2,c2)) or None
    """
    if mask_eye is None:
        return {
            "eye_mask": None,
            "eye_centroid": None,
            "eye_area": 0.0,
            "eye_diameter": 0.0,
            "eye_diameter_points": None,
        }

    eye_raw = np.asarray(mask_eye)
    if eye_raw.ndim != 2 or eye_raw.size == 0:
        return {
            "eye_mask": None,
            "eye_centroid": None,
            "eye_area": 0.0,
            "eye_diameter": 0.0,
            "eye_diameter_points": None,
        }

    if eye_raw.dtype == bool:
        eye_mask = eye_raw.copy()
    else:
        eye_raw_float = eye_raw.astype(float)
        max_val = float(np.nanmax(eye_raw_float)) if eye_raw_float.size else 0.0
        if max_val <= 0.0:
            return {
                "eye_mask": np.zeros_like(eye_raw, dtype=bool),
                "eye_centroid": None,
                "eye_area": 0.0,
                "eye_diameter": 0.0,
                "eye_diameter_points": None,
            }
        eye_mask = eye_raw_float >= (0.5 * max_val)

    if mask_fish is not None:
        fish_mask = np.asarray(mask_fish).astype(bool)
        if fish_mask.shape == eye_mask.shape:
            eye_mask = eye_mask & fish_mask

    if not eye_mask.any():
        return {
            "eye_mask": eye_mask,
            "eye_centroid": None,
            "eye_area": 0.0,
            "eye_diameter": 0.0,
            "eye_diameter_points": None,
        }

    ecoords = np.argwhere(eye_mask)
    if len(ecoords) == 0:
        return {
            "eye_mask": eye_mask,
            "eye_centroid": None,
            "eye_area": 0.0,
            "eye_diameter": 0.0,
            "eye_diameter_points": None,
        }

    eye_centroid = ecoords.mean(axis=0)
    dy, dx = spacing
    eye_area = float(len(ecoords) * dy * dx)

    eye_boundary = eye_mask & ~binary_erosion(eye_mask)
    dcoords = np.argwhere(eye_boundary)
    if len(dcoords) < 2:
        dcoords = ecoords

    eye_diameter = 0.0
    eye_diameter_points = None
    if len(dcoords) >= 2:
        dcoords_phys = dcoords.astype(float).copy()
        dcoords_phys[:, 0] *= dy
        dcoords_phys[:, 1] *= dx

        mean_phys = dcoords_phys.mean(axis=0)
        centered = dcoords_phys - mean_phys
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        major_vec = evecs[:, int(np.argmax(evals))]
        major_vec /= (np.linalg.norm(major_vec) + 1e-12)

        proj = centered @ major_vec
        i_min = int(np.argmin(proj))
        i_max = int(np.argmax(proj))
        p0_phys = dcoords_phys[i_min]
        p1_phys = dcoords_phys[i_max]

        eye_diameter = float(np.linalg.norm(p1_phys - p0_phys))

        p0_pix = np.array([p0_phys[0] / dy, p0_phys[1] / dx])
        p1_pix = np.array([p1_phys[0] / dy, p1_phys[1] / dx])
        eye_diameter_points = (
            tuple(np.round(p0_pix).astype(int)),
            tuple(np.round(p1_pix).astype(int)),
        )

    return {
        "eye_mask": eye_mask,
        "eye_centroid": eye_centroid,
        "eye_area": eye_area,
        "eye_diameter": eye_diameter,
        "eye_diameter_points": eye_diameter_points,
    }

def compute_eye_diameters(mask_eye, spacing=(1.0, 1.0)):
    """
    Measure the horizontal and vertical diameters of the eye from the binary mask.
    spacing = (dy, dx) — physical units (µm) per pixel.
    Returns {‘eye_width_um’: horizontal diameter, ‘eye_height_um’: vertical diameter}.
    """
    out = {"eye_width_um": 0.0, "eye_height_um": 0.0}
    if mask_eye is None:
        return out
    m = np.asarray(mask_eye)
    if m.ndim == 3:
        m = m[..., 0]
    m = m > 0
    if not m.any():
        return out
    dy, dx = spacing

    try:
        import cv2
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = labels == largest
    except Exception:
        pass

    ys, xs = np.where(m)
    width_px  = int(xs.max() - xs.min() + 1)   
    height_px = int(ys.max() - ys.min() + 1)   
    out["eye_width_um"]  = float(width_px * dx)
    out["eye_height_um"] = float(height_px * dy)
    return out

def compute_tube_metrics(mask, spacing=(1.0, 1.0)):
    """
    Fit a minimum-area rotated rectangle to a tube-shaped binary mask.

    Unlike a simple bounding box, the rectangle follows the tube's actual
    orientation, so the short side is the cross-sectional width regardless of
    how the tube is rotated in the image (the long side is discarded — callers
    that only care about the body length already get that from
    tube_length_border2border).

    spacing: (dy, dx) physical units per pixel.

    Returns dict with keys:
        area: physical area (spacing units squared)
        length: long-axis extent (spacing units) — the "long part"
        width: short-axis extent (spacing units) — the tube width
        length_line: ((r1,c1),(r2,c2)) endpoints of the long-axis midline, or None
        width_line: ((r1,c1),(r2,c2)) endpoints of the width midline, or None
    """
    out = {"area": 0.0, "length": 0.0, "width": 0.0, "length_line": None, "width_line": None}
    if mask is None:
        return out
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8)
    if not m.any():
        return out
    dy, dx = spacing

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8)

    out["area"] = float(int(m.sum()) * dy * dx)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return out
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return out

    box = cv2.boxPoints(cv2.minAreaRect(contour))  # 4 (x, y) pixel points, in order around the rect
    box_rc = box[:, ::-1]  # -> (row, col) to match this codebase's point convention
    A, B, C, D = box_rc

    def _phys_len(p1, p2):
        dr, dc = (p2[0] - p1[0]) * dy, (p2[1] - p1[1]) * dx
        return float(np.sqrt(dr ** 2 + dc ** 2))

    def _mid(p1, p2):
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    side_AB, side_BC = _phys_len(A, B), _phys_len(B, C)

    # The segment joining the midpoints of a pair of parallel sides spans the
    # *other* pair's side length (it cuts straight across the rectangle), so
    # the line with length == side_AB is mid(BC)-mid(DA), and vice versa.
    if side_AB >= side_BC:
        out["length"], out["width"] = side_AB, side_BC
        out["length_line"] = (_mid(B, C), _mid(D, A))
        out["width_line"] = (_mid(A, B), _mid(C, D))
    else:
        out["length"], out["width"] = side_BC, side_AB
        out["length_line"] = (_mid(A, B), _mid(C, D))
        out["width_line"] = (_mid(B, C), _mid(D, A))

    return out


def tube_length_border2border(mask, spacing=(1.0, 1.0), return_path=False, return_skeleton=False, return_straight_line=False, return_extensions=False, mask_eye=None, return_eye_info=False):
    """
    Border-to-border, branch-free centerline length for a tube-like binary mask.

    mask: 2D binary array
    spacing: (dy, dx) in physical units
    return_path: return (N,2) array of [row,col]
    return_skeleton: return a bool image with only the centerline path
    return_straight_line: return the two endpoints of the longest straight line
    return_extensions: return boolean array indicating which path points are extensions
    mask_eye: optional 2D binary eye mask; if provided, path starts at the fish border pixel closest to eye mask
    return_eye_info: return eye-derived diagnostics computed inside this function
    
    Returns:
        length: total path length along centerline
        straight_length: longest straight-line distance between any two border points
        [optional] path: coordinates of the centerline path
        [optional] skel_main: skeleton image
        [optional] straight_line_points: tuple of (point1, point2) for longest line
        [optional] extension_mask: boolean array where True = extension, False = skeleton
        [optional] eye_info: dict with eye_mask, eye_centroid, closest_border_to_eye,
                             eye_diameter, eye_area, eye_diameter_points
    """
    mask = mask.astype(bool)
    eye_mask_used = np.zeros_like(mask, dtype=bool)
    eye_centroid = None
    closest_border_to_eye = None
    eye_diameter = 0.0
    eye_area = 0.0
    eye_diameter_points = None
    bridge_len_override = None
    if mask.sum() == 0:
        out = (0.0, 0.0,)
        if return_path: out += (np.zeros((0, 2), dtype=int),)
        if return_skeleton: out += (np.zeros_like(mask, dtype=bool),)
        if return_eye_info:
            out += ({
                "eye_mask": eye_mask_used,
                "eye_centroid": eye_centroid,
                "closest_border_to_eye": closest_border_to_eye,
                "eye_diameter": eye_diameter,
                "eye_area": eye_area,
                "eye_diameter_points": eye_diameter_points,
            },)
        return out[0] if len(out) == 1 else out

    # --- boundary pixels ---
    boundary = mask & ~binary_erosion(mask)
    bcoords = np.argwhere(boundary)
    if len(bcoords) == 0:
        out = (0.0, 0.0,)
        if return_path: out += (np.zeros((0, 2), dtype=int),)
        if return_skeleton: out += (np.zeros_like(mask, dtype=bool),)
        if return_eye_info:
            out += ({
                "eye_mask": eye_mask_used,
                "eye_centroid": eye_centroid,
                "closest_border_to_eye": closest_border_to_eye,
                "eye_diameter": eye_diameter,
                "eye_area": eye_area,
                "eye_diameter_points": eye_diameter_points,
            },)
        return out[0] if len(out) == 1 else out
    btree = cKDTree(bcoords)

    # --- medial axis + distance (in pixels) ---
    skel, dist_skel = medial_axis(mask, return_distance=True)

    # --- endpoints on skeleton (may be empty for loops) ---
    k = np.ones((3, 3), dtype=np.uint8)
    neigh = convolve(skel.astype(np.uint8), k, mode="constant", cval=0)
    endpoints = np.argwhere(skel & (neigh == 2))
    candidates = endpoints if len(endpoints) >= 2 else np.argwhere(skel)

    if len(candidates) < 2:
        # fallback: use boundary PCA extremes as rough endpoints
        pts = np.argwhere(mask)
        mu = pts.mean(axis=0)
        X = pts - mu
        # principal direction
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        v = vt[0]
        proj = (bcoords - mu) @ v
        p1 = tuple(bcoords[np.argmin(proj)])
        p2 = tuple(bcoords[np.argmax(proj)])
        path = np.array([p1, p2], dtype=int)
        extension_mask = np.zeros(len(path), dtype=bool)
    else:
        # --- diameter path on skeleton (branch-free polyline) ---
        cost_skel = np.where(skel, 1.0, np.inf)
        mcp_skel = MCP_Geometric(cost_skel, fully_connected=True)

        A = tuple(candidates[0])
        costsA, _ = mcp_skel.find_costs([A])
        valsA = np.array([costsA[tuple(p)] for p in candidates])
        B = tuple(candidates[np.nanargmax(valsA)])

        costsB, _ = mcp_skel.find_costs([B])
        valsB = np.array([costsB[tuple(p)] for p in candidates])
        C = tuple(candidates[np.nanargmax(valsB)])

        path_skel = np.array(mcp_skel.traceback(C), dtype=int)  # C -> ... -> B
        if path_skel.size == 0:
            out = (0.0, 0.0,)
            if return_path: out += (np.zeros((0, 2), dtype=int),)
            if return_skeleton: out += (np.zeros_like(mask, dtype=bool),)
            if return_eye_info:
                out += ({
                    "eye_mask": eye_mask_used,
                    "eye_centroid": eye_centroid,
                    "closest_border_to_eye": closest_border_to_eye,
                    "eye_diameter": eye_diameter,
                    "eye_area": eye_area,
                    "eye_diameter_points": eye_diameter_points,
                },)
            return out[0] if len(out) == 1 else out
        
        # --- Smooth the skeleton path to reduce sharp turns, especially at thick ends ---
        def smooth_skeleton_path(path, dist_map, window=15, end_weight=10.0):
            """
            Smooth skeleton path with extra emphasis on straightening at the ends.
            Uses distance transform to weight smoothing - more smoothing where tube is thicker.
            """
            if len(path) < 10:
                return path
            
            path_smooth = path.astype(float).copy()
            n = len(path)
            
            # Compute thickness at each point
            thickness = np.array([dist_map[tuple(p)] for p in path])
            thickness_norm = thickness / (thickness.max() + 1e-6)
            
            # Apply Gaussian-like smoothing with position-dependent weight
            for i in range(n):
                # Distance from ends (normalized)
                dist_from_start = i / n
                dist_from_end = (n - 1 - i) / n
                end_proximity = 1.0 - min(dist_from_start, dist_from_end) * 2  # 1 at ends, 0 at middle
                end_proximity = max(0, end_proximity)
                
                # Smoothing strength: higher at thick parts and at ends
                smooth_weight = 0.3 + 0.5 * thickness_norm[i] + end_weight * end_proximity
                smooth_weight = min(smooth_weight, 1.0)
                
                # Define window around current point
                half_win = window // 2
                start_idx = max(0, i - half_win)
                end_idx = min(n, i + half_win + 1)
                
                if end_idx - start_idx > 2:
                    # Compute weighted average of nearby points
                    local_points = path[start_idx:end_idx].astype(float)
                    weights = np.exp(-0.5 * ((np.arange(len(local_points)) - (i - start_idx)) / (half_win/2)) ** 2)
                    weights /= weights.sum()
                    
                    smoothed_point = (weights[:, None] * local_points).sum(axis=0)
                    path_smooth[i] = smooth_weight * smoothed_point + (1 - smooth_weight) * path[i]
            
            # Round and convert back to int
            path_smooth = np.round(path_smooth).astype(int)
            
            # Ensure all points are within bounds and on the mask
            path_smooth[:, 0] = np.clip(path_smooth[:, 0], 0, mask.shape[0] - 1)
            path_smooth[:, 1] = np.clip(path_smooth[:, 1], 0, mask.shape[1] - 1)
            
            return path_smooth
        
        path_skel = smooth_skeleton_path(path_skel, dist_skel, window=60, end_weight=10.0)

        # --- ray-cast from skeleton endpoints to find boundary in straight line ---
        def raycast_to_boundary(skel_point, dir_vec):
            """
            Ray-cast from skeleton endpoint in the given direction to find boundary point.
            Returns the boundary point where the ray exits the mask.
            """
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-6:
                # fallback: nearest boundary
                _, idx = btree.query(skel_point, k=1)
                return tuple(bcoords[idx])
            
            # Normalize direction
            direction = dir_vec / dir_norm
            
            # Ray-cast from skeleton point outward
            current = np.array(skel_point, dtype=float)
            step_size = 0.5  # sub-pixel steps for accuracy
            max_steps = int(max(mask.shape) * 2)  # safety limit
            
            for _ in range(max_steps):
                current += direction * step_size
                
                # Check if we're out of bounds
                r, c = int(round(current[0])), int(round(current[1]))
                if r < 0 or r >= mask.shape[0] or c < 0 or c >= mask.shape[1]:
                    # Hit image boundary, backtrack slightly
                    current -= direction * step_size
                    break
                
                # Check if we've exited the mask
                if not mask[r, c]:
                    # We've left the mask, backtrack to last valid point
                    current -= direction * step_size
                    break
            
            # Find the nearest boundary point to where our ray ended
            ray_end = current
            _, idx = btree.query(ray_end, k=1)
            return tuple(bcoords[idx])

        n = len(path_skel)
        step = min(10, n - 1)

        # outward direction at start (from inside towards border)
        start = tuple(path_skel[0])
        dir_start = path_skel[0] - path_skel[step]
        b1 = raycast_to_boundary(start, dir_start)

        # outward direction at end
        end = tuple(path_skel[-1])
        dir_end = path_skel[-1] - path_skel[-1 - step]
        b2 = raycast_to_boundary(end, dir_end)

        # --- extend skeleton path directly to boundaries ---
        # Use straight-line extension from skeleton endpoints to boundary points
        def extend_to_boundary(skel_point, boundary_point):
            """Create straight line from skeleton endpoint to boundary point"""
            sp = np.array(skel_point, dtype=float)
            bp = np.array(boundary_point, dtype=float)
            direction = bp - sp
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                return np.array([skel_point], dtype=int)
            
            # Number of points for interpolation - straight line regardless of mask
            n_points = max(2, int(np.ceil(dist)))
            t = np.linspace(0, 1, n_points)
            extension = sp[None, :] + t[:, None] * direction[None, :]
            extension = np.round(extension).astype(int)
            
            # Clip to image bounds only
            extension[:, 0] = np.clip(extension[:, 0], 0, mask.shape[0] - 1)
            extension[:, 1] = np.clip(extension[:, 1], 0, mask.shape[1] - 1)
            
            return extension
        
        # Extend from start
        ext_start = extend_to_boundary(start, b1)
        # Extend from end
        ext_end = extend_to_boundary(end, b2)
        
        # Track which parts are extensions
        len_ext_start = len(ext_start)
        len_skel = len(path_skel)
        len_ext_end = len(ext_end)
        
        # Combine: boundary -> extension -> skeleton -> extension -> boundary
        if len(ext_start) > 1:
            ext_start = ext_start[::-1]  # reverse to go from boundary toward skeleton
        if len(ext_end) > 1:
            ext_end = ext_end[1:]  # skip first point (already in skeleton)
        
        # Build complete path
        path = np.vstack([ext_start, path_skel, ext_end])
        
        # Create extension mask
        extension_mask = np.zeros(len(path), dtype=bool)
        extension_mask[:len(ext_start)] = True
        if len(ext_end) > 1:
            extension_mask[-len(ext_end)+1:] = True
        
        # Remove any duplicate consecutive points
        mask_diff = np.any(np.diff(path, axis=0) != 0, axis=1)
        keep_indices = np.concatenate([[True], mask_diff])
        path = path[keep_indices]
        extension_mask = extension_mask[keep_indices]

    # --- optional: force start point to be the border pixel closest to eye mask ---
    if mask_eye is not None and len(path) >= 2:
        eye_metrics = compute_eye_metrics(mask_eye, mask_fish=mask, spacing=spacing)
        eye_mask = eye_metrics["eye_mask"]
        eye_centroid = eye_metrics["eye_centroid"]
        eye_diameter = float(eye_metrics["eye_diameter"])
        eye_area = float(eye_metrics["eye_area"])
        eye_diameter_points = eye_metrics["eye_diameter_points"]

        if eye_mask is not None and eye_mask.any() and len(bcoords) > 0:
            eye_mask_used = eye_mask.copy()
            eye_boundary = eye_mask & ~binary_erosion(eye_mask)
            ecoords = np.argwhere(eye_boundary)
            if len(ecoords) == 0:
                ecoords = np.argwhere(eye_mask)

            if len(ecoords) > 0:
                eye_centroid = ecoords.mean(axis=0)
                # Closest fish-border pixel to the eye mask
                dist_be = cdist(bcoords.astype(float), ecoords.astype(float))
                closest_border = bcoords[np.argmin(dist_be.min(axis=1))].astype(int)
                closest_border_to_eye = closest_border

                # Orient path so the nearer endpoint is at the start
                d0 = np.linalg.norm(path[0].astype(float) - closest_border.astype(float))
                d1 = np.linalg.norm(path[-1].astype(float) - closest_border.astype(float))
                if d1 < d0:
                    path = path[::-1]
                    extension_mask = extension_mask[::-1]

                # Remove anchor "swerve": trim to nearest existing path point,
                # then reconnect to the eye anchor with a smooth C1 bridge.
                if len(path) >= 2:
                    d_path = np.linalg.norm(path.astype(float) - closest_border.astype(float), axis=1)
                    idx_near = int(np.argmin(d_path))

                    # If nearest point ended up near path tail, flip once and recompute
                    if idx_near > (len(path) // 2):
                        path = path[::-1]
                        extension_mask = extension_mask[::-1]
                        d_path = np.linalg.norm(path.astype(float) - closest_border.astype(float), axis=1)
                        idx_near = int(np.argmin(d_path))

                    # Trim early detour points and start from nearest centerline location
                    if idx_near > 0:
                        path = path[idx_near:]
                        extension_mask = extension_mask[idx_near:]

                    # Further trim an early section, then reconnect with a smooth
                    # cubic Hermite bridge that matches the centerline tangent at join.
                    n_trim_start = min(52, len(path) - 2) if len(path) > 30 else 0
                    if n_trim_start > 0:
                        kept = path[n_trim_start:]
                        kept_ext = extension_mask[n_trim_start:]
                    else:
                        kept = path.copy()
                        kept_ext = extension_mask.copy()

                    anchor = closest_border.astype(float)
                    join = kept[0].astype(float)
                    chord = join - anchor
                    d01 = np.linalg.norm(chord)

                    if d01 > 1e-6:
                        # Start tangent points from anchor toward join.
                        t0 = chord / (d01 + 1e-9)

                        # End tangent follows local centerline direction at join.
                        look_ahead = min(len(kept) - 1, 6)
                        t1_vec = kept[look_ahead].astype(float) - join
                        t1_norm = np.linalg.norm(t1_vec)
                        if t1_norm < 1e-6:
                            t1_vec = chord
                            t1_norm = d01
                        t1 = t1_vec / (t1_norm + 1e-9)

                        tangent_scale = 0.55 * d01
                        m0 = t0 * tangent_scale
                        m1 = t1 * tangent_scale

                        n_bridge = int(np.clip(np.ceil(d01 / 1.2), 8, 30))
                        t = np.linspace(0.0, 1.0, n_bridge + 1)
                        h00 = 2 * t**3 - 3 * t**2 + 1
                        h10 = t**3 - 2 * t**2 + t
                        h01 = -2 * t**3 + 3 * t**2
                        h11 = t**3 - t**2
                        bridge_f = (
                            h00[:, None] * anchor[None, :]
                            + h10[:, None] * m0[None, :]
                            + h01[:, None] * join[None, :]
                            + h11[:, None] * m1[None, :]
                        )

                        # Keep a sub-pixel bridge length for accurate final length
                        if len(bridge_f) >= 2:
                            dy_b, dx_b = spacing
                            db = np.diff(bridge_f, axis=0)
                            bridge_len_override = float(np.sqrt((db[:, 0] * dy_b) ** 2 + (db[:, 1] * dx_b) ** 2).sum())

                        # Gentle local smoothing to suppress discretization corners.
                        if len(bridge_f) > 4:
                            for _ in range(3):
                                prev = bridge_f.copy()
                                bridge_f[1:-1] = 0.25 * prev[:-2] + 0.5 * prev[1:-1] + 0.25 * prev[2:]

                        bridge = np.round(bridge_f).astype(int)
                        bridge[:, 0] = np.clip(bridge[:, 0], 0, mask.shape[0] - 1)
                        bridge[:, 1] = np.clip(bridge[:, 1], 0, mask.shape[1] - 1)

                        path = np.vstack([bridge, kept[1:]]) if len(kept) > 1 else bridge
                        extension_mask = (
                            np.concatenate([
                                np.ones(len(bridge), dtype=bool),
                                kept_ext[1:]
                            ])
                            if len(kept_ext) > 1
                            else np.ones(len(bridge), dtype=bool)
                        )

                    # Enforce first point as exact anchor and smooth early path slightly.
                    path[0] = closest_border
                    extension_mask[0] = True
                    if len(path) >= 6:
                        smooth_end = min(len(path) - 2, 24)
                        if smooth_end >= 2:
                            pf = path.astype(float)
                            for _ in range(2):
                                prev = pf.copy()
                                pf[1:smooth_end + 1] = (
                                    0.2 * prev[0:smooth_end]
                                    + 0.6 * prev[1:smooth_end + 1]
                                    + 0.2 * prev[2:smooth_end + 2]
                                )
                            pf[0] = closest_border.astype(float)
                            path[:smooth_end + 1] = np.round(pf[:smooth_end + 1]).astype(int)
                            path[:, 0] = np.clip(path[:, 0], 0, mask.shape[0] - 1)
                            path[:, 1] = np.clip(path[:, 1], 0, mask.shape[1] - 1)

                    # Remove duplicate consecutive points that can arise after snapping
                    if len(path) >= 2:
                        mask_diff2 = np.any(np.diff(path, axis=0) != 0, axis=1)
                        keep_indices2 = np.concatenate([[True], mask_diff2])
                        path = path[keep_indices2]
                        extension_mask = extension_mask[keep_indices2]

    # --- compute physical length along the (branch-free) path ---
    dy, dx = spacing
    pf = path.astype(float)
    dxy = np.diff(pf, axis=0)
    seg = np.sqrt((dxy[:, 0] * dy) ** 2 + (dxy[:, 1] * dx) ** 2)
    length = float(seg.sum())

    # If a Hermite bridge was used, compensate discretization loss from rounding.
    if bridge_len_override is not None and len(path) >= 2 and len(extension_mask) == len(path):
        non_ext = np.where(~extension_mask)[0]
        bridge_end_idx = int(non_ext[0]) if len(non_ext) > 0 else len(path) - 1
        if bridge_end_idx >= 1:
            pb = path[:bridge_end_idx + 1].astype(float)
            db_pix = np.diff(pb, axis=0)
            bridge_len_discrete = float(np.sqrt((db_pix[:, 0] * dy) ** 2 + (db_pix[:, 1] * dx) ** 2).sum())
            length += (bridge_len_override - bridge_len_discrete)

    # --- compute straight-line distance between start and end points of path ---
    straight_line_points = None
    if len(path) >= 2:
        start_point = path[0].astype(float)
        end_point = path[-1].astype(float)
        diff = end_point - start_point
        straight_length = float(np.sqrt((diff[0] * dy) ** 2 + (diff[1] * dx) ** 2))
        straight_line_points = (tuple(path[0]), tuple(path[-1]))
    else:
        straight_length = 0.0

    skel_main = np.zeros_like(mask, dtype=bool)
    if path.size:
        skel_main[path[:, 0], path[:, 1]] = True

    out = (length, straight_length,)
    if return_path: out += (path,)
    if return_skeleton: out += (skel_main,)
    if return_straight_line: out += (straight_line_points,)
    if return_extensions: out += (extension_mask,)
    if return_eye_info:
        out += ({
            "eye_mask": eye_mask_used,
            "eye_centroid": eye_centroid,
            "closest_border_to_eye": closest_border_to_eye,
            "eye_diameter": eye_diameter,
            "eye_area": eye_area,
            "eye_diameter_points": eye_diameter_points,
        },)
    return out


def normalize_images(data):
        # Check if data contains np.arrays, if yes, directly normalize them
        if isinstance(data[0], np.ndarray):
            return np.array(data, dtype=np.float32)
        else:
            return np.array([np.array(image) for image in data], dtype=np.float32) 
        
def apply_mask(original_image, mask):
    """
    Apply the mask to the original image.
    """
    # Convert the mask to a 3-channel image
    original_image = cv2.resize(original_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Invert the mask so that the fish is white (255) and background is black (0)
    #mask = cv2.bitwise_not(mask)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original_image, mask_3ch)

    return masked_image

def classification_curvature(image, mask, model, use_threshold, threshold):
    
    masked_image = apply_mask(image, mask)

    cropped_image = preprocess_masked_image(masked_image)

    if cropped_image is None:
        # Fully-black masked image (empty segmentation) — return uncertain curvature
        return None, torch.tensor([5]).to(device)

    # Ensure the masked image is in RGB format
    masked_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    # Ensure the image is scaled to [0, 255] before preprocessing
    masked_image_rgb = np.clip(masked_image_rgb, 0, 255).astype(np.uint8)
    
    # Preprocess the image
    processed_image = normalize_images([masked_image_rgb])

    processed_image = T.ToPILImage()(processed_image[0])
    processed_image = T.ToTensor()(processed_image)
    processed_image = processed_image.unsqueeze(0)
    #processed_image = torch.from_numpy(processed_image).permute(0, 3, 1, 2).float()
    processed_image = processed_image.to(device)
    
    outputs = model(processed_image)
    curvature = 1 + torch.argmax(outputs, dim=1)

    probs = F.softmax(outputs, dim=1)
    confs, preds = torch.max(probs, dim=1)

    if use_threshold:
        if confs < threshold:
            curvature = torch.tensor([5]).to(device)

    return cropped_image, curvature

class FishClassifier(nn.Module):
    def __init__(self, num_classes, dense_layer_size, dropout_rate, model_name='resnet101'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.flatten = nn.Flatten()
        # Get backbone output feature size by passing a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            dummy_output = self.backbone(dummy_input)
            backbone_out_features = dummy_output.shape[1] if len(dummy_output.shape) > 1 else dummy_output.shape[0]
        self.fc1 = nn.Linear(backbone_out_features, dense_layer_size)
        #self.fc1 = nn.Linear(self.backbone.num_features, dense_layer_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_layer_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model():
    
    _HF_TOKEN = os.getenv("HF_TOKEN", None)
    model_path = hf_hub_download(
        repo_id="markdanielarndt/Classification",
        filename="best_model_class.pth",
        token=_HF_TOKEN
    )

    fallback = {'dense_layer': 512, 'dropout': 0.2, 'model_name': 'convnext_base'}
    print("Warning: best_params not found. Using fallback params:", fallback)
    best_params = fallback

    # Instantiate and load
    model = FishClassifier(num_classes=4,
                        dense_layer_size=best_params['dense_layer'],
                        dropout_rate=best_params['dropout'],
                        model_name=best_params['model_name'])
    try:
        # # Try loading state dict first
        state = torch.load(model_path, map_location=device)
        #if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        #     model.load_state_dict(state)
        # else:
        #     # If saved the entire model object
        model = state
    except Exception as e:
        # Last resort: try direct load_state_dict on the object
        model.load_state_dict(torch.load(model_path, map_location=device))
    # model = model.to(device)
    # Ensure `model` is an nn.Module on the correct device and in eval mode.

    def _strip_module_prefix(state_dict):
        if any(k.startswith('module.') for k in state_dict.keys()):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
        return state_dict

    if isinstance(model, nn.Module):
        model = model.to(device)
        model.eval()
    else:
        # model is likely a state_dict / OrderedDict -> instantiate and load
        state_dict = model
        if isinstance(state_dict, (dict, OrderedDict)):
            state_dict = _strip_module_prefix(state_dict)
            instantiated = FishClassifier(
                num_classes=4,
                dense_layer_size=best_params['dense_layer'],
                dropout_rate=best_params['dropout'],
                model_name=best_params['model_name']
            )
            # handle common nested keys
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                state_dict = _strip_module_prefix(state_dict)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                state_dict = _strip_module_prefix(state_dict)
            instantiated.load_state_dict(state_dict)
            model = instantiated.to(device)
            model.eval()
        else:
            raise TypeError("Loaded object is neither an nn.Module nor a state dict.")

    return model


def plot_edges_with_curvature(mask, min_contour_length, window_size_ratio):
    # Compute edge properties
    edge_pixels, curvature_values = compute_curvature_profile(mask, min_contour_length, window_size_ratio)

    # Plot the mask
    plt.imshow(mask, cmap='gray')
    # We set the min and max of the colorbar, so that 90% of the curvature values are shown.
    # This is to have a nice visualization. You can change this threshold according to your specific task.
    threshold = np.percentile(np.abs(curvature_values), 90)
    plt.scatter(edge_pixels[:, 1], edge_pixels[:, 0], c=curvature_values, cmap='jet', s=5, vmin=-threshold, vmax=threshold)

    plt.colorbar(label='Curvature')
    plt.title("Curvature of Edge Pixels")
    plt.show()
    return curvature_values

def compute_curvature_profile(mask, min_contour_length, window_size_ratio):
    # Compute the contours of the mask to be able to analyze each part individually
    contours = measure.find_contours(mask, 0.5)

    # Initialize arrays to store the curvature information for each edge pixel
    curvature_values = []
    edge_pixels = []

    # Iterate over each contour
    for contour in contours:
        # Iterate over each point in the contour
        for i, point in enumerate(contour):
            # We set the minimum contour length to 20
            # You can change this minimum-value according to your specific requirements
            if contour.shape[0] > min_contour_length:
                # Compute the curvature for the point
                # We set the window size to 1/5 of the whole contour edge. Adjust this value according to your specific task
                window_size = int(contour.shape[0]/window_size_ratio)
                curvature = compute_curvature(point, i, contour, window_size)
                # We compute, whether a point is convex or concave.
                # If you want to have the 2nd derivative shown you can comment this part
                # if curvature > 0:
                #     curvature = 1
                # if curvature <= 0:
                #     curvature = -1
                # Store curvature information and corresponding edge pixel
                curvature_values.append(curvature)
                edge_pixels.append(point)

    # Convert lists to numpy arrays for further processing
    curvature_values = np.array(curvature_values)
    edge_pixels = np.array(edge_pixels)

    return edge_pixels, curvature_values


def compute_curvature(point, i, contour, window_size):
    # Compute the curvature using polynomial fitting in a local coordinate system

    # Extract neighboring edge points
    start = max(0, i - window_size // 2)
    end = min(len(contour), i + window_size // 2 + 1)
    neighborhood = contour[start:end]

    # Extract x and y coordinates from the neighborhood
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]

    # Compute the tangent direction over the entire neighborhood and rotate the points
    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood), np.gradient(x_neighborhood))
    tangent_direction_original.fill(tangent_direction_original[len(tangent_direction_original)//2])

    # Translate the neighborhood points to the central point
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]


    # Apply rotation to the translated neighborhood points
    # We have to rotate the points to be able to compute the curvature independent of the local orientation of the curve
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    # Fit a polynomial of degree 2 to the rotated coordinates
    coeffs = np.polyfit(rotated_x, rotated_y, 2)


    # You can compute the curvature using the formula: curvature = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
    # dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
    # d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)
    # curvature = np.abs(d2y_dx2) / np.power(1 + np.power(dy_dx, 2), 1.5)
    # We compute the 2nd derivative in order to determine whether the curve at the certain point is convex or concave
    curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)

    # Return the mean curvature for the central point
    return np.mean(curvature)

# Set minimum length of the contours that should be analyzed
min_contour_length = 20
# Set the ratio of the window size (contour length / window_size_ratio) for local polynomial approximation
window_size_ratio = 5

def preprocess_masked_image(image, target_size=(256, 256)):
    """
    Preprocess a single masked image by cropping to the bounding box, 
    padding to a square, and resizing to the target size.

    Args:
        image (numpy array): The input masked image.
        target_size (tuple): The desired output size (width, height).

    Returns:
        numpy array: The processed image.
    """
    # Step 1: Convert to grayscale and find non-black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)

    if coords is None:  # Image is fully black
        print("The image is fully black and cannot be processed.")
        return None

    # Step 2: Crop to bounding box
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y+h, x:x+w]

    # Step 3: Pad to square size
    height, width = cropped_image.shape[:2]
    max_dim = max(height, width)
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left

    padded_image = cv2.copyMakeBorder(
        cropped_image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Step 4: Resize
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_image

