"""
shape_autocorrect.py - Improved shape detection for air writing
Detects: lines, circles, rectangles, and squares
"""

import numpy as np
import cv2


def autocorrect_stroke(points):
    """
    Analyze a stroke and detect if it's a recognizable shape.
    
    Args:
        points: List of (x, y) tuples representing the drawn stroke
        
    Returns:
        (shape_type, shape_params) or (None, None)
        - shape_type: "line", "circle", "rectangle", "square", or None
        - shape_params: shape-specific parameters
    """
    if len(points) < 5:
        return None, None
    
    # Convert to numpy array for easier processing
    pts = np.array(points, dtype=np.float32)
    
    # Try detection in order of priority
    # 1. Line (most specific)
    line_result = detect_line(pts)
    if line_result[0]:
        return "line", line_result[1]
    
    # 2. Circle
    circle_result = detect_circle(pts)
    if circle_result[0]:
        return "circle", circle_result[1]
    
    # 3. Rectangle/Square
    rect_result = detect_rectangle(pts)
    if rect_result[0]:
        return rect_result[0], rect_result[1]
    
    return None, None


def detect_line(points):
    """
    Detect if stroke is a straight line.
    
    Returns:
        (is_line, (x1, y1, x2, y2)) or (False, None)
    """
    if len(points) < 5:
        return False, None
    
    # Use the first and last points as endpoints
    start = points[0]
    end = points[-1]
    
    # Calculate the length of the line
    line_length = np.linalg.norm(end - start)
    
    # Skip very short strokes
    if line_length < 30:
        return False, None
    
    # Calculate perpendicular distance from each point to the line
    # Line direction vector
    line_vec = end - start
    line_vec_normalized = line_vec / (line_length + 1e-6)
    
    # Calculate distances
    max_deviation = 0
    total_deviation = 0
    
    for point in points:
        # Vector from start to point
        point_vec = point - start
        
        # Project onto line direction
        projection_length = np.dot(point_vec, line_vec_normalized)
        projection = start + projection_length * line_vec_normalized
        
        # Perpendicular distance
        deviation = np.linalg.norm(point - projection)
        max_deviation = max(max_deviation, deviation)
        total_deviation += deviation
    
    avg_deviation = total_deviation / len(points)
    
    # Line criteria:
    # 1. Average deviation should be small relative to line length
    # 2. Max deviation should be small
    deviation_ratio = avg_deviation / line_length
    max_deviation_ratio = max_deviation / line_length
    
    is_line = (deviation_ratio < 0.08 and max_deviation_ratio < 0.15) or \
              (avg_deviation < 15 and max_deviation < 25)
    
    if is_line:
        x1, y1 = start
        x2, y2 = end
        return True, (float(x1), float(y1), float(x2), float(y2))
    
    return False, None


def detect_circle(points):
    """
    Detect if stroke is a circle with simplified, robust logic.
    
    Returns:
        (is_circle, (cx, cy, radius)) or (False, None)
    """
    if len(points) < 8:
        return False, None
    
    # Method 1: Use minimum enclosing circle from OpenCV
    points_for_cv = points.astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), radius = cv2.minEnclosingCircle(points_for_cv)
    
    # Skip very small circles
    if radius < 15:
        return False, None
    
    # Calculate how well points fit this circle
    distances = np.linalg.norm(points - np.array([cx, cy]), axis=1)
    
    # Check 1: Standard deviation of distances (how consistent is the radius?)
    radius_std = np.std(distances)
    radius_variation_coeff = radius_std / (radius + 1e-6)
    
    # Check 2: Mean distance should be close to the fitted radius
    mean_distance = np.mean(distances)
    radius_accuracy = abs(mean_distance - radius) / (radius + 1e-6)
    
    # Check 3: Start and end points should be close (closed shape)
    start_end_dist = np.linalg.norm(points[0] - points[-1])
    closure_ratio = start_end_dist / (radius * 2 + 1e-6)
    
    # Check 4: Angular coverage - ensure points go around the circle
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    
    # Find angle range covered
    angles_unwrapped = np.unwrap(angles)
    total_angle_coverage = abs(angles_unwrapped[-1] - angles_unwrapped[0])
    
    # Should cover at least 300 degrees (5.24 radians) for a circle
    min_coverage = 5.24  # ~300 degrees in radians
    coverage_ok = total_angle_coverage >= min_coverage
    
    # Check 5: Calculate circularity using area ratio
    # Perimeter of the stroke
    perimeter = 0
    for i in range(1, len(points)):
        perimeter += np.linalg.norm(points[i] - points[i-1])
    
    # Ideal perimeter for a circle with this radius
    ideal_perimeter = 2 * np.pi * radius
    perimeter_ratio = perimeter / (ideal_perimeter + 1e-6)
    
    # For a circle, perimeter ratio should be close to 1.0
    # Allow 0.8 to 1.3 range (hand-drawn can be longer or shorter)
    perimeter_ok = 0.75 < perimeter_ratio < 1.4
    
    # Decision criteria - be more lenient
    is_circle = (
        radius_variation_coeff < 0.30 and  # Relaxed from 0.25
        radius_accuracy < 0.20 and         # Relaxed from 0.15
        closure_ratio < 0.5 and            # Relaxed from 0.4
        coverage_ok and                     # Must cover ~300+ degrees
        perimeter_ok                        # Perimeter should be reasonable
    )
    
    # Debug output (optional - remove if too verbose)
    if False:  # Set to True for debugging
        print(f"[CIRCLE DEBUG] var_coeff={radius_variation_coeff:.3f}, "
              f"accuracy={radius_accuracy:.3f}, closure={closure_ratio:.3f}, "
              f"coverage={np.degrees(total_angle_coverage):.1f}°, "
              f"perim_ratio={perimeter_ratio:.3f}, result={is_circle}")
    
    if is_circle:
        return True, (float(cx), float(cy), float(radius))
    
    return False, None


def detect_rectangle(points):
    """
    Detect if stroke is a rectangle or square.
    
    Returns:
        (shape_type, corners) where shape_type is "rectangle" or "square"
        corners: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        or (False, None)
    """
    if len(points) < 15:
        return False, None
    
    # Use convex hull to find the outer boundary
    hull = cv2.convexHull(points.astype(np.float32))
    
    # Approximate the hull with a polygon
    epsilon = 0.04 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    # Check if we have 4 corners
    if len(approx) != 4:
        return False, None
    
    # Get the 4 corners
    corners = approx.reshape(4, 2)
    
    # Calculate the perimeter to ensure it's large enough
    perimeter = cv2.arcLength(approx, True)
    if perimeter < 100:  # Minimum perimeter threshold
        return False, None
    
    # Order corners: top-left, top-right, bottom-right, bottom-left
    corners = order_rectangle_corners(corners)
    
    # Calculate side lengths
    side1 = np.linalg.norm(corners[1] - corners[0])
    side2 = np.linalg.norm(corners[2] - corners[1])
    side3 = np.linalg.norm(corners[3] - corners[2])
    side4 = np.linalg.norm(corners[0] - corners[3])
    
    # Check if opposite sides are approximately equal
    opposite_sides_equal = (
        abs(side1 - side3) / max(side1, side3) < 0.25 and
        abs(side2 - side4) / max(side2, side4) < 0.25
    )
    
    if not opposite_sides_equal:
        return False, None
    
    # Check angles (should be close to 90 degrees)
    angles = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angles.append(np.degrees(angle))
    
    # Check if all angles are close to 90 degrees
    angles_ok = all(70 < angle < 110 for angle in angles)
    
    if not angles_ok:
        return False, None
    
    # Determine if it's a square (all sides approximately equal)
    avg_side = (side1 + side2 + side3 + side4) / 4
    all_sides_equal = all(
        abs(side - avg_side) / avg_side < 0.2
        for side in [side1, side2, side3, side4]
    )
    
    shape_type = "square" if all_sides_equal else "rectangle"
    
    # Convert corners to list of tuples
    corner_list = [(float(x), float(y)) for x, y in corners]
    
    return shape_type, corner_list


def order_rectangle_corners(corners):
    """
    Order rectangle corners as: top-left, top-right, bottom-right, bottom-left
    """
    # Calculate centroid
    centroid = np.mean(corners, axis=0)
    
    # Sort by angle from centroid
    angles = np.arctan2(corners[:, 1] - centroid[1], 
                        corners[:, 0] - centroid[0])
    
    # Sort corners by angle
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    
    # Find the top-left corner (minimum x + y)
    sums = sorted_corners[:, 0] + sorted_corners[:, 1]
    top_left_idx = np.argmin(sums)
    
    # Rotate array so top-left is first
    ordered = np.roll(sorted_corners, -top_left_idx, axis=0)
    
    return ordered


def calculate_stroke_smoothness(points):
    """
    Calculate how smooth/jagged a stroke is.
    Lower values = smoother
    """
    if len(points) < 3:
        return 0
    
    angles = []
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angles.append(abs(angle))
    
    return np.mean(angles) if angles else 0


# Additional utility function for debugging
def get_shape_confidence(points, shape_type, shape_params):
    """
    Calculate confidence score for detected shape (0-1)
    Useful for debugging and fine-tuning thresholds
    """
    if shape_type == "line":
        line_result = detect_line(points)
        return 0.9 if line_result[0] else 0.0
    
    elif shape_type == "circle":
        circle_result = detect_circle(points)
        return 0.85 if circle_result[0] else 0.0
    
    elif shape_type in ["rectangle", "square"]:
        rect_result = detect_rectangle(points)
        return 0.8 if rect_result[0] else 0.0
    
    return 0.0