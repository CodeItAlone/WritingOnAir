# shape_autocorrect.py

import cv2
import numpy as np

def _fit_line_error(points):
    pts = np.array(points, dtype=np.float32)

    # Fit a line using PCA-like approach
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

    # distance of each point to the fitted line
    diffs = pts - np.array([[x0, y0]])
    # cross product magnitude / |v|
    dist = np.abs(diffs[:, 0] * vy - diffs[:, 1] * vx)
    return dist.mean()


def _line_candidate(points, max_error_ratio=0.03):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 5:
        return None

    # bounding box size as scale
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    diag = np.hypot(x_max - x_min, y_max - y_min)
    if diag < 20:  # too tiny
        return None

    err = _fit_line_error(pts)
    # allow some error relative to scale
    if err / max(diag, 1) > max_error_ratio:
        return None

    # line from first to last point
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    return ("line", (x1, y1, x2, y2))


def _circle_candidate(points, closure_thresh=0.2, radial_error_ratio=0.25):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 10:
        return None

    # must be roughly closed: start and end near each other
    p_start = pts[0]
    p_end   = pts[-1]
    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = np.hypot(bbox[0], bbox[1])
    if diag < 20:
        return None

    dist_end_start = np.linalg.norm(p_end - p_start)
    if dist_end_start > closure_thresh * diag:
        return None

    # approximate circle
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    cx, cy, r = float(cx), float(cy), float(r)
    if r < 10:
        return None

    dists = np.linalg.norm(pts - np.array([[cx, cy]]), axis=1)
    err = np.abs(dists - r).mean()

    if err / r > radial_error_ratio:
        return None

    return ("circle", (cx, cy, r))


def _rectangle_candidate(points, epsilon_ratio=0.04, angle_tol_deg=25):
    """
    Detects rectangle or square from a rough closed stroke.
    Accepts fairly imperfect rectangles.
    """
    pts = np.array(points, dtype=np.int32)
    if len(pts) < 8:
        return None

    # must be roughly closed
    p_start = pts[0].astype(np.float32)
    p_end   = pts[-1].astype(np.float32)
    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.hypot(bbox[0], bbox[1]))
    if diag < 30:
        return None

    dist_end_start = np.linalg.norm(p_end - p_start)
    if dist_end_start > 0.3 * diag:
        # not even roughly closed, forget rectangle
        return None

    contour = pts.reshape(-1, 1, 2)
    peri = cv2.arcLength(contour, True)

    # approximate polygon
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)

    # need 4 points for rectangle
    if len(approx) != 4:
        return None

    approx = approx.reshape(-1, 2)

    # check convexity
    if not cv2.isContourConvex(approx.reshape(-1, 1, 2)):
        return None

    # check that all angles are ~90 degrees
    def angle(a, b, c):
        ab = a - b
        cb = c - b
        cosang = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    angles = []
    for i in range(4):
        p0 = approx[(i - 1) % 4]
        p1 = approx[i]
        p2 = approx[(i + 1) % 4]
        angles.append(angle(p0, p1, p2))

    # all angles close to 90 ± angle_tol_deg
    for ang in angles:
        if not (90 - angle_tol_deg <= ang <= 90 + angle_tol_deg):
            return None

    # rectangle / square
    # reorder points for consistency (optional, we just return them)
    xs = approx[:, 0]
    ys = approx[:, 1]

    # Return the 4 corners in order
    # You can keep approx as-is; the caller just needs the points.
    corners = [(int(x), int(y)) for x, y in approx]

    # check if square (aspect ratio)
    w_rect = np.linalg.norm(approx[1] - approx[0])
    h_rect = np.linalg.norm(approx[2] - approx[1])
    aspect = w_rect / (h_rect + 1e-6)
    if 0.8 <= aspect <= 1.25:
        shape_type = "square"
    else:
        shape_type = "rectangle"

    return (shape_type, corners)


def autocorrect_stroke(points):
    """
    Takes a list of (x, y) points (stroke),
    returns: (shape_type, shape_params) or (None, None)

    shape_type: "line", "circle", "rectangle", "square"
    shape_params:
        - line:      (x1, y1, x2, y2)
        - circle:    (cx, cy, r)
        - rectangle: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        - square:    same as rectangle
    """
    if len(points) < 3:
        return (None, None)

    # 1) Try line first (easiest)
    line = _line_candidate(points)
    if line is not None:
        return line

    # 2) Try circle
    circle = _circle_candidate(points)
    if circle is not None:
        return circle

    # 3) Try rectangle / square
    rect = _rectangle_candidate(points)
    if rect is not None:
        return rect

    return (None, None)
