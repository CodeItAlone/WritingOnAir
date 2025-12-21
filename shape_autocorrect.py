import cv2
import numpy as np
import math

# ----------------- Utility -----------------

def _distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _angle(a, b, c):
    ab = a - b
    cb = c - b
    cosang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


# ----------------- Line -----------------

def line_score(points):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 6:
        return None

    # --- Fit line ---
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    diffs = pts - np.array([[x0, y0]])
    dist = np.abs(diffs[:, 0] * vy - diffs[:, 1] * vx)
    mean_err = dist.mean()

    bbox = pts.max(axis=0) - pts.min(axis=0)
    scale = np.hypot(bbox[0], bbox[1]) + 1e-6

    # --- 1. Distance score (existing logic) ---
    dist_score = 1.0 - (mean_err / scale) * 6.0

    if dist_score < 0.65:
        return None

    # --- 2. Curvature rejection (NEW) ---
    directions = pts[1:] - pts[:-1]
    norms = np.linalg.norm(directions, axis=1) + 1e-6
    directions = directions / norms[:, None]

    # angle change between successive segments
    dots = np.sum(directions[1:] * directions[:-1], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angle_changes = np.degrees(np.arccos(dots))

    mean_angle_change = angle_changes.mean()

    # handwriting has frequent direction changes
    if mean_angle_change > 6.0:
        return None

    # --- 3. End-to-end dominance check (NEW) ---
    start_end_dist = np.linalg.norm(pts[-1] - pts[0])
    path_length = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))

    # a real line has near-direct movement
    if start_end_dist / (path_length + 1e-6) < 0.97:
        return None

    # --- Final score ---
    score = max(0.0, dist_score - (mean_angle_change / 20.0))

    if score < 0.7:
        return None

    return {
        "type": "line",
        "score": score,
        "params": (int(pts[0][0]), int(pts[0][1]),
                   int(pts[-1][0]), int(pts[-1][1]))
    }



# ----------------- Circle -----------------

def circle_score(points):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 12:
        return None

    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = np.hypot(bbox[0], bbox[1])
    if diag < 30:
        return None

    (cx, cy), r = cv2.minEnclosingCircle(pts)
    if r < 10:
        return None

    dists = np.linalg.norm(pts - np.array([[cx, cy]]), axis=1)
    radial_err = np.abs(dists - r).mean() / r

    # corner rejection
    contour = pts.reshape(-1, 1, 2)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    corner_penalty = 0.25 if len(approx) <= 6 else 0.0

    score = max(0.0, 1.0 - radial_err * 4 - corner_penalty)

    if score < 0.6:
        return None

    return {
        "type": "circle",
        "score": score,
        "params": (int(cx), int(cy), int(r))
    }


# ----------------- Rectangle / Square -----------------

def rectangle_score(points):
    pts = np.array(points, dtype=np.int32)
    if len(pts) < 8:
        return None

    contour = pts.reshape(-1, 1, 2)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    if len(approx) != 4:
        return None

    approx = approx.reshape(-1, 2)

    if not cv2.isContourConvex(approx.reshape(-1, 1, 2)):
        return None

    angles = []
    for i in range(4):
        angles.append(_angle(
            approx[(i - 1) % 4],
            approx[i],
            approx[(i + 1) % 4]
        ))

    angle_err = sum(abs(a - 90) for a in angles) / 4
    angle_score = max(0.0, 1.0 - angle_err / 40)

    sides = [_distance(approx[i], approx[(i + 1) % 4]) for i in range(4)]
    w, h = sides[0], sides[1]
    aspect = w / (h + 1e-6)

    if 0.85 <= aspect <= 1.15:
        shape_type = "square"
        aspect_penalty = abs(aspect - 1.0)
    else:
        shape_type = "rectangle"
        aspect_penalty = 0.0

    score = max(0.0, angle_score - aspect_penalty)

    if score < 0.6:
        return None

    corners = [(int(x), int(y)) for x, y in approx]

    return {
        "type": shape_type,
        "score": score,
        "params": corners
    }


# ----------------- MASTER CLASSIFIER -----------------

def autocorrect_stroke(points):
    """
    Returns:
        (shape_type, shape_params)
        or (None, None)
    """

    if len(points) < 5:
        return (None, None)

    candidates = []

    for detector in (line_score, rectangle_score, circle_score):
        result = detector(points)
        if result:
            candidates.append(result)

    if not candidates:
        return (None, None)

    best = max(candidates, key=lambda x: x["score"])

    return best["type"], best["params"]
