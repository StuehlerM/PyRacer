"""
Physics utilities for the racing game including collision detection
and line intersection calculations.
"""
import numpy as np


def line_segment_intersection(p1, p2, p3, p4):
    """
    Check if line segments p1-p2 and p3-p4 intersect.
    
    Returns:
        tuple: (intersection_point, True) if they intersect, (None, False) otherwise
    """
    # Convert to numpy arrays if they aren't already
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    p4 = np.array(p4, dtype=float)
    
    # Parametric intersection solves p1 + ua*(p2-p1) = p3 + ub*(p4-p3).
    # Shared denominator becomes zero when segment directions are parallel.
    denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    
    # If lines are parallel
    if abs(denom) < 1e-10:
        return None, False
    
    # ua/ub tell how far along each segment intersection lies: 0=start, 1=end.
    ua_numerator = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])
    ub_numerator = (p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])
    
    ua = ua_numerator / denom
    ub = ub_numerator / denom
    
    # Both parameters must stay in [0, 1], otherwise infinite lines cross outside segment bounds.
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # Calculate intersection point
        intersection = p1 + ua * (p2 - p1)
        return intersection, True
    
    return None, False


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: tuple (x, y) - the point to check
        polygon: list of tuples [(x1, y1), (x2, y2), ...] - polygon vertices
    
    Returns:
        bool: True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    # Shoot horizontal ray and count edge crossings; odd count means point is inside.
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        
        # Check if point is on the edge
        if (x1 == x2 and x == x1 and min(y1, y2) <= y <= max(y1, y2)) or \
           (y1 == y2 and y == y1 and min(x1, x2) <= x <= max(x1, x2)):
            return True
        
        # Each crossing flips inside/outside parity, which is why we toggle boolean.
        if y > min(y1, y2) and y <= max(y1, y2):
            if x <= max(x1, x2):
                if y1 != y2:
                    x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if x1 == x2 or x <= x_intersect:
                    inside = not inside
    
    return inside


def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the distance from a point to a line segment.
    
    Args:
        point: tuple (x, y)
        line_start: tuple (x, y)
        line_end: tuple (x, y)
    
    Returns:
        float: distance from point to line segment
    """
    p = np.array(point, dtype=float)
    a = np.array(line_start, dtype=float)
    b = np.array(line_end, dtype=float)
    
    # Vector from a to b
    ab = b - a
    # Vector from a to p
    ap = p - a
    
    # Projection scalar says where closest point falls on infinite line through segment.
    t = np.dot(ap, ab) / np.dot(ab, ab)
    
    # Clamp keeps closest point on finite segment instead of extending past endpoints.
    t = max(0, min(1, t))
    
    # Closest point on the line segment
    closest = a + t * ab
    
    # Distance from point to closest point
    return float(np.linalg.norm(p - closest))


def ray_cast(start, direction, max_distance, track_segments, segment_indices=None):
    """
    Cast a ray from start in direction and find the first intersection
    with any track segment.
    
    Args:
        start: tuple (x, y) - ray origin
        direction: tuple (dx, dy) - normalized direction vector
        max_distance: float - maximum ray length
        track_segments: list of (start, end) tuples - track boundary segments
        segment_indices: list of int - indices of segments to check (for optimization)
    
    Returns:
        float: distance to first intersection, or max_distance if no intersection
    """
    start = np.array(start, dtype=float)
    direction = np.array(direction, dtype=float)
    
    # Avoid division by zero for zero-length direction vectors
    dir_norm = np.linalg.norm(direction)
    if dir_norm > 1e-10:
        direction = direction / dir_norm
    else:
        return max_distance
    
    seg_starts, seg_ends = _segments_to_arrays(track_segments)
    if seg_starts.size == 0:
        return max_distance

    if segment_indices is not None:
        segment_indices = np.asarray(segment_indices, dtype=np.int64)
        if segment_indices.size == 0:
            return max_distance
        seg_starts = seg_starts[segment_indices]
        seg_ends = seg_ends[segment_indices]

    # Batch version lets NumPy test all candidate segments at once instead of Python looping.
    return ray_cast_batch(start, direction, max_distance, seg_starts, seg_ends)


def _segments_to_arrays(track_segments):
    """Return segment start/end arrays from cached arrays or tuple segments."""
    if isinstance(track_segments, tuple) and len(track_segments) == 2:
        return (
            np.asarray(track_segments[0], dtype=float),
            np.asarray(track_segments[1], dtype=float),
        )

    segments = np.asarray(track_segments, dtype=float)
    if segments.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    return segments[:, 0, :], segments[:, 1, :]


def ray_cast_batch(start, direction, max_distance, seg_starts, seg_ends):
    """
    Cast a ray against many segments using vectorized line intersection.

    Args:
        start: shape (2,) ray origin
        direction: shape (2,) normalized ray direction
        max_distance: float - maximum ray length
        seg_starts: shape (N, 2) segment starts
        seg_ends: shape (N, 2) segment ends

    Returns:
        float: distance to first intersection, or max_distance if none
    """
    if len(seg_starts) == 0:
        return max_distance

    p1 = np.asarray(start, dtype=float)
    direction = np.asarray(direction, dtype=float)
    p2 = p1 + direction * max_distance
    p3 = np.asarray(seg_starts, dtype=float)
    p4 = np.asarray(seg_ends, dtype=float)

    d1 = p2 - p1
    d2 = p4 - p3
    # Same ua/ub math as single-segment case, but broadcast across every boundary segment.
    denom = d2[:, 1] * d1[0] - d2[:, 0] * d1[1]

    valid = np.abs(denom) >= 1e-10
    if not np.any(valid):
        return max_distance

    p1_minus_p3 = p1 - p3
    ua_num = d2[:, 0] * p1_minus_p3[:, 1] - d2[:, 1] * p1_minus_p3[:, 0]
    ub_num = d1[0] * p1_minus_p3[:, 1] - d1[1] * p1_minus_p3[:, 0]

    ua = np.full(len(p3), np.nan, dtype=float)
    ub = np.full(len(p3), np.nan, dtype=float)
    ua[valid] = ua_num[valid] / denom[valid]
    ub[valid] = ub_num[valid] / denom[valid]

    # Vectorized mask keeps only intersections lying on both finite segments.
    hits = valid & (ua >= 0.0) & (ua <= 1.0) & (ub >= 0.0) & (ub <= 1.0)
    if not np.any(hits):
        return max_distance

    # Smallest ua is first wall hit along ray, which is sensor distance we care about.
    return float(np.min(ua[hits]) * max_distance)


def circle_polygon_collision(center, radius, polygon):
    """
    Check if a circle collides with a polygon.
    
    Args:
        center: tuple (x, y) - circle center
        radius: float - circle radius
        polygon: list of tuples - polygon vertices
    
    Returns:
        bool: True if collision detected
    """
    n = len(polygon)
    
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        
        # Check distance from circle center to line segment
        distance = point_to_line_distance(center, p1, p2)
        if distance < radius:
            return True
    
    # Center-inside case catches full penetration where no edge lies within radius anymore.
    if point_in_polygon(center, polygon):
        return True
    
    return False


def get_circle_polygon_collision_point(center, radius, polygon):
    """
    Get the collision point and normal for a circle colliding with a polygon.
    
    Args:
        center: tuple (x, y) - circle center
        radius: float - circle radius
        polygon: list of tuples - polygon vertices
    
    Returns:
        tuple: (collision_point, normal_vector, penetration_depth) or (None, None, 0) if no collision
    """
    n = len(polygon)
    closest_distance = float('inf')
    collision_point = None
    normal = None
    
    for i in range(n):
        p1 = np.array(polygon[i], dtype=float)
        p2 = np.array(polygon[(i + 1) % n], dtype=float)
        
        # Distance from center to line segment
        distance = point_to_line_distance(center, p1, p2)
        
        if distance < radius and distance < closest_distance:
            closest_distance = distance
            
            # Reuse projection idea to find exact contact point on edge.
            center_arr = np.array(center, dtype=float)
            ab = p2 - p1
            ap = center_arr - p1
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0, min(1, t))
            collision_point = p1 + t * ab
            
            # Collision normal points outward from wall so later response can push car away.
            edge_dir = ab / np.linalg.norm(ab)
            to_center = center_arr - collision_point
            to_center = to_center / np.linalg.norm(to_center)
            
            # Normal points from edge to circle
            normal = to_center
            
    if collision_point is not None:
        penetration = radius - closest_distance
        return collision_point, normal, penetration
    
    return None, None, 0


def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def s_curve_acceleration(current_speed, max_speed, acceleration_rate, throttle, dt):
    """
    Apply S-curve acceleration that starts fast and slows as approaching max speed.
    
    This creates a more natural acceleration feel where the car accelerates quickly
    initially but the acceleration tapers off as it approaches top speed.
    
    Args:
        current_speed: float - current speed of the car
        max_speed: float - maximum speed
        acceleration_rate: float - base acceleration rate
        throttle: float - throttle input (0-1)
        dt: float - time delta
    
    Returns:
        float: acceleration amount to apply
    """
    if max_speed <= 0:
        return 0
    
    # Normalize speed so same curve shape works for any chosen top speed.
    speed_ratio = abs(current_speed) / max_speed
    
    # Squaring (1 - ratio) keeps launch punchy, then tapers harder near top speed.
    acceleration_factor = (1 - speed_ratio) ** 2  # Quadratic falloff for S-curve effect
    
    return throttle * acceleration_rate * acceleration_factor * dt


def log_curve_acceleration(current_speed, max_speed, acceleration_rate, throttle, dt):
    """
    Apply log-curve acceleration that feels more natural.
    
    Args:
        current_speed: float - current speed of the car
        max_speed: float - maximum speed
        acceleration_rate: float - base acceleration rate
        throttle: float - throttle input (0-1)
        dt: float - time delta
    
    Returns:
        float: acceleration amount to apply
    """
    if max_speed <= 0:
        return 0
    
    # Normalize speed before applying nonlinear throttle response.
    speed_ratio = abs(current_speed) / max_speed
    
    # Log curve compresses high-speed region, so each extra bit of speed gets harder to earn.
    # `1 + 9*ratio` maps [0,1] to [1,10], giving clean log10 range from 0 to 1.
    if speed_ratio >= 1:
        acceleration_factor = 0
    else:
        # Subtract from 1 so throttle starts strong and fades as ratio approaches 1.
        acceleration_factor = 1 - (np.log10(1 + 9 * speed_ratio) / 1.0)
        acceleration_factor = max(0, min(1, acceleration_factor))
    
    return throttle * acceleration_rate * acceleration_factor * dt


def rotate_vector(vector, angle):
    """
    Rotate a 2D vector by an angle in radians.
    
    Args:
        vector: tuple (x, y)
        angle: float - angle in radians
    
    Returns:
        tuple: rotated (x, y)
    """
    x, y = vector
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    return (
        x * cos_a - y * sin_a,
        x * sin_a + y * cos_a
    )


def angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors in radians.
    
    Args:
        v1: tuple (x, y)
        v2: tuple (x, y)
    
    Returns:
        float: angle in radians
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Calculate dot product
    dot = np.dot(v1, v2)
    
    # Clamp to avoid numerical errors
    dot = clamp(dot, -1.0, 1.0)
    
    return np.arccos(dot)
