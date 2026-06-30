"""
Track generation and representation for the racing game.
Generates procedural 2D tracks with configurable complexity.
"""
import numpy as np
import pygame
from .physics import line_segment_intersection
from utils.config import config


class Track:
    """
    Represents a 2D racing track with procedural generation.
    
    The track is defined by a center line (waypoints) and has a configurable width.
    It includes inner and outer boundaries, start/finish line, and checkpoints.
    """
    
    def __init__(self, width=config.SCREEN_WIDTH, height=config.SCREEN_HEIGHT, 
                 track_width=config.TRACK_WIDTH, complexity=config.TRACK_COMPLEXITY,
                 spline_points=30, seed=None):
        """
        Initialize a new track.
        
        Args:
            width: int - screen width
            height: int - screen height
            track_width: int - width of the track (road)
            complexity: int - number of control points for generation
            spline_points: int - number of intermediate points between control points
            seed: int - optional seed for reproducible track generation
        """
        self.screen_width = width
        self.screen_height = height
        self.track_width = track_width
        self.complexity = complexity
        self.spline_points = spline_points
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None
        
        # Generate track
        self.waypoints = []  # Center line waypoints
        self.inner_boundary = []  # Inner edge points
        self.outer_boundary = []  # Outer edge points
        self.segments = []  # All boundary segments for collision
        self.checkpoints = []  # Checkpoint positions
        self.start_position = (0, 0)  # Will be set after generation
        self.start_angle = 0  # Starting direction in radians
        self.start_idx = 0  # Waypoint index used for start/finish
        self.finish_line = None  # (start_point, end_point)
        self._last_progress_idx = None
        
        self._generate_track()
        self._create_boundaries()
        self._create_checkpoints()
        self._cache_track_data()
        self._build_spatial_grid()
    
    def _generate_track(self):
        """
        Generate a procedural track using Freya Holmer's spline-based road approach.
        
        This creates smooth roads by:
        1. Generating control points in a circular pattern with randomness
        2. Using Catmull-Rom spline to create smooth center line
        3. The waypoints themselves form the center line for the track
        """
        # Set random seed for reproducibility during development
        # np.random.seed(42)  # Uncomment for reproducible tracks
        
        # Center of the screen
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        # Generate control points in a circular pattern with randomness
        # Freya Holmer's approach: use control points that the spline passes through
        num_control_points = max(self.complexity + 4, 6)  # At least 6 control points
        # Circular ordering gives simple closed loop; spline later wraps with modulo indices.
        
        # Create control points around a circle with random offsets
        # Use a base radius and add randomness for interesting tracks
        radius = min(self.screen_width, self.screen_height) * 0.35
        
        control_points = []
        for i in range(num_control_points):
            angle = 2 * np.pi * i / num_control_points
            
            # Random offset from circle for variety
            # This creates the "wobbly" circular track
            # Same angle order + different radii keeps loop readable while adding bends/chicanes.
            offset = self._uniform(-radius * 0.3, radius * 0.3)
            
            x = center_x + (radius + offset) * np.cos(angle)
            y = center_y + (radius + offset) * np.sin(angle)
            
            control_points.append((x, y))
        
        # Generate smooth spline waypoints from control points
        # Use Catmull-Rom spline for smooth curves (Freya Holmer style)
        # The spline passes through all control points
        self.waypoints = self._catmull_rom_spline(control_points, num_intermediate=self.spline_points)
        
        # Store control points for reference/debugging
        self.control_points = control_points
        
        # Cache bounding box for quick rejection
        self._cached_min_x = None
        self._cached_max_x = None
        self._cached_min_y = None
        self._cached_max_y = None
        
        # Cache boundary segments array for faster access
        self._boundary_segments_array = None
        self._segment_starts = None
        self._segment_ends = None
        
        # Set start position and angle
        # Start at the bottom of the screen (approximately)
        start_idx = 0
        for i, (x, y) in enumerate(self.waypoints):
            if y > center_y and abs(x - center_x) < radius * 0.3:
                start_idx = i
                break
        
        # Start position is slightly offset from waypoint
        self.start_idx = start_idx
        start_point = self.waypoints[start_idx]
        next_point = self.waypoints[(start_idx + 1) % len(self.waypoints)]
        # Modulo wraps final waypoint back to first so start/finish still has "next" segment.
        
        # Start in the middle of the track
        self.start_position = self._get_center_point(start_point, next_point, offset=0)
        
        # Start angle is the direction from start to next waypoint
        dx = next_point[0] - start_point[0]
        dy = next_point[1] - start_point[1]
        self.start_angle = np.arctan2(dy, dx)
        
        # Set finish line
        self.finish_line = (start_point, next_point)
        
        # Cache for performance optimization
        self._cached_waypoints_array = None
        self._cached_segment_lengths = None
        self._cached_total_length = None
        self._cached_waypoints_count = 0
        
        # Spatial partitioning for faster ray casting
        self._spatial_grid = None
        self._grid_cell_size = 100  # Size of each grid cell in pixels
    
    def _catmull_rom_spline(self, points, num_intermediate=10):
        """
        Generate smooth Catmull-Rom spline through control points (closed loop).
        
        Args:
            points: list of (x, y) tuples - control points
            num_intermediate: int - number of points to generate between each control point
        
        Returns:
            list: list of (x, y) tuples forming the closed spline
        """
        if len(points) < 2:
            return points
        
        n = len(points)
        spline_points = []
        
        for i in range(n):
            # Closed-loop indexing: p0/p1/p2/p3 are prev, current, next, next-next control points.
            p0 = points[(i - 1) % n]
            p1 = points[i % n]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]
            
            # Generate intermediate points along this segment (endpoint=False
            # avoids duplicating control points shared between segments)
            for t in np.linspace(0, 1, num_intermediate, endpoint=False):
                # t moves from p1 toward p2; p0 and p3 shape tangent so curve stays smooth.
                # Catmull-Rom interpolation formula
                # Key property: curve passes through control points, unlike Bezier handles.
                x = 0.5 * ((2 * p1[0]) + 
                          (-p0[0] + p2[0]) * t + 
                          (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t**2 + 
                          (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t**3)
                y = 0.5 * ((2 * p1[1]) + 
                          (-p0[1] + p2[1]) * t + 
                          (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t**2 + 
                          (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t**3)
                
                spline_points.append((x, y))
        
        # Remove any near-duplicate points that could cause degenerate segments
        cleaned = [spline_points[0]]
        for pt in spline_points[1:]:
            dx = pt[0] - cleaned[-1][0]
            dy = pt[1] - cleaned[-1][1]
            if dx * dx + dy * dy > 1.0:  # skip points closer than 1 pixel
                cleaned.append(pt)
        
        return cleaned
    
    def _smooth_waypoints(self, points, iterations=1, factor=0.5):
        """
        Smooth the waypoints to create a more natural track (fallback method).
        
        Args:
            points: list of (x, y) tuples
            iterations: int - number of smoothing passes
            factor: float - smoothing factor (0-1)
        
        Returns:
            list: smoothed waypoints
        """
        for _ in range(iterations):
            smoothed = []
            n = len(points)
            
            for i in range(n):
                prev = points[(i - 1) % n]
                curr = points[i]
                next_p = points[(i + 1) % n]
                
                # Average with neighbors
                x = curr[0] + factor * (prev[0] + next_p[0] - 2 * curr[0])
                y = curr[1] + factor * (prev[1] + next_p[1] - 2 * curr[1])
                
                smoothed.append((x, y))
            
            points = smoothed
        
        return points
    
    def _create_boundaries(self):
        """
        Create smooth inner and outer boundaries from the center waypoints.
        Uses normals at each waypoint with miter limiting to prevent blow-up on sharp turns.
        """
        n = len(self.waypoints)
        half_width = self.track_width / 2
        # Maximum miter scale factor — limits offset on sharp corners
        max_miter = 2.0
        
        # Calculate normals at each waypoint for smooth offset
        normals = []
        miter_scales = []
        for i in range(n):
            # Get previous and next waypoints
            prev = np.array(self.waypoints[(i - 1) % n])
            curr = np.array(self.waypoints[i])
            next_wp = np.array(self.waypoints[(i + 1) % n])
            
            # Calculate tangent vector (average of incoming and outgoing edges)
            edge_in = curr - prev
            edge_out = next_wp - curr
            
            # Normalize edges
            len_in = np.linalg.norm(edge_in)
            len_out = np.linalg.norm(edge_out)
            
            if len_in > 0 and len_out > 0:
                edge_in_n = edge_in / len_in
                edge_out_n = edge_out / len_out
            elif len_out > 0:
                edge_in_n = edge_out / len_out
                edge_out_n = edge_in_n
            else:
                edge_in_n = np.array([1.0, 0.0])
                edge_out_n = edge_in_n
            
            # Average tangent direction
            tangent = (edge_in_n + edge_out_n) / 2
            tangent_len = np.linalg.norm(tangent)
            
            if tangent_len > 1e-6:
                tangent = tangent / tangent_len
                # Miter scale: how much we need to extend offset to maintain
                # constant distance from both edges. Clamped to prevent blow-up.
                # Sharp corners make tangent tiny, which would explode offset without clamp.
                miter_scale = min(1.0 / tangent_len, max_miter)
            else:
                # Near-180° turn: edges point in opposite directions
                tangent = np.array([-edge_in_n[1], edge_in_n[0]])
                miter_scale = 1.0
            
            # Calculate normal (perpendicular to tangent, pointing left)
            # Normal says "which way is road edge" relative to centerline travel direction.
            normal = np.array([-tangent[1], tangent[0]])
            normals.append(normal)
            miter_scales.append(miter_scale)
        
        # Create inner and outer boundary points with miter-limited offsets
        for i in range(n):
            curr = np.array(self.waypoints[i])
            normal = normals[i]
            scale = miter_scales[i]
            
            offset = half_width * scale
            # Using +/- same normal keeps both borders symmetric around centerline.
            inner_point = curr + normal * offset
            outer_point = curr - normal * offset
            
            self.inner_boundary.append(tuple(inner_point))
            self.outer_boundary.append(tuple(outer_point))
        
        # Create segments for collision detection by connecting consecutive boundary points
        n_boundary = len(self.inner_boundary)
        for i in range(n_boundary):
            inner_p1 = self.inner_boundary[i]
            inner_p2 = self.inner_boundary[(i + 1) % n_boundary]
            outer_p1 = self.outer_boundary[i]
            outer_p2 = self.outer_boundary[(i + 1) % n_boundary]
            
            self.segments.append((inner_p1, inner_p2))
            self.segments.append((outer_p1, outer_p2))
        
        # Cache segments as numpy array for faster access
        self._boundary_segments_array = np.asarray(self.segments, dtype=float)
        self._segment_starts = self._boundary_segments_array[:, 0, :]
        self._segment_ends = self._boundary_segments_array[:, 1, :]
        
        # Pre-calculate bounding box for quick rejection
        self._update_bounding_box()
    
    def _create_checkpoints(self):
        """
        Create checkpoints along the track for progress measurement.
        """
        n = len(self.waypoints)
        num_checkpoints = config.NUM_CHECKPOINTS
        
        # Place checkpoints after the start line, evenly around the lap.
        for i in range(num_checkpoints):
            idx = (self.start_idx + int((i + 1) * n / (num_checkpoints + 1))) % n
            checkpoint_pos = self.waypoints[idx]
            self.checkpoints.append({
                'position': checkpoint_pos,
                'index': i
            })
        
        # The finish line also serves as the last checkpoint
        if self.finish_line:
            self.checkpoints.append({
                'position': self.finish_line[0],
                'index': num_checkpoints,
                'is_finish': True
            })
    
    def _get_center_point(self, p1, p2, offset=0):
        """
        Get a point in the center of the track at a specific offset.
        
        Args:
            p1: tuple - first waypoint
            p2: tuple - second waypoint
            offset: float - offset along the segment (0-1)
        
        Returns:
            tuple: (x, y) position in the center of the track
        """
        p1 = np.array(p1)
        p2 = np.array(p2)
        
        # Interpolate along the segment
        center = p1 + (p2 - p1) * offset
        
        # Calculate normal
        edge = p2 - p1
        edge_length = np.linalg.norm(edge)
        
        if edge_length < 1:
            return tuple(center)
        
        normal = np.array([-edge[1], edge[0]]) / edge_length
        
        # Center is the waypoint itself (waypoints are the center line)
        return tuple(center)
    
    def _update_bounding_box(self):
        """Update the cached bounding box for the track."""
        if not self.segments:
            self._cached_min_x = self._cached_max_x = self.screen_width / 2
            self._cached_min_y = self._cached_max_y = self.screen_height / 2
            return
        
        all_x = [p[0] for seg in self.segments for p in seg]
        all_y = [p[1] for seg in self.segments for p in seg]
        
        self._cached_min_x = min(all_x)
        self._cached_max_x = max(all_x)
        self._cached_min_y = min(all_y)
        self._cached_max_y = max(all_y)
    
    def _cache_track_data(self):
        """Cache track data for faster progress calculations."""
        if not self.waypoints:
            return
        
        # Convert waypoints to numpy array for faster access
        self._cached_waypoints_array = np.array(self.waypoints)
        self._cached_waypoints_count = len(self.waypoints)
        
        # Pre-calculate segment lengths and total length
        segment_lengths = []
        total_length = 0.0
        
        for i in range(self._cached_waypoints_count):
            p1 = self._cached_waypoints_array[i]
            p2 = self._cached_waypoints_array[(i + 1) % self._cached_waypoints_count]
            seg_len = np.linalg.norm(p2 - p1)
            segment_lengths.append(seg_len)
            total_length += seg_len
        
        self._cached_segment_lengths = np.array(segment_lengths)
        self._cached_total_length = total_length
    
    def _build_spatial_grid(self):
        """Build a spatial grid for faster ray casting and collision detection."""
        if not self.segments:
            return
        
        # Calculate grid dimensions
        grid_width = int(np.ceil(self.screen_width / self._grid_cell_size))
        grid_height = int(np.ceil(self.screen_height / self._grid_cell_size))
        
        # Initialize grid as a dictionary
        self._spatial_grid = {}
        
        # Assign each segment to grid cells
        # Spatial grid turns "check all walls" into "check walls near this region".
        for idx, (start, end) in enumerate(self.segments):
            start = np.array(start)
            end = np.array(end)
            
            # Find all grid cells this segment intersects
            min_cell_x = int(np.floor(start[0] / self._grid_cell_size))
            max_cell_x = int(np.floor(end[0] / self._grid_cell_size))
            min_cell_y = int(np.floor(start[1] / self._grid_cell_size))
            max_cell_y = int(np.floor(end[1] / self._grid_cell_size))
            
            # Ensure we cover all cells the segment might pass through
            min_cell_x = max(0, min_cell_x - 1)
            max_cell_x = min(grid_width - 1, max_cell_x + 1)
            min_cell_y = max(0, min_cell_y - 1)
            max_cell_y = min(grid_height - 1, max_cell_y + 1)
            
            # Add this segment to all relevant grid cells
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    cell_key = (cell_x, cell_y)
                    if cell_key not in self._spatial_grid:
                        self._spatial_grid[cell_key] = []
                    self._spatial_grid[cell_key].append(idx)
    
    def get_boundaries(self):
        """
        Get all boundary segments for collision detection.
        
        Returns:
            list: all boundary segments as ((x1, y1), (x2, y2)) tuples
        """
        return self.segments

    def get_boundary_arrays(self):
        """
        Get cached boundary segment arrays for vectorized ray casting.

        Returns:
            tuple: (segment_starts, segment_ends), each shaped (N, 2)
        """
        if self._segment_starts is None or self._segment_ends is None:
            self._boundary_segments_array = np.asarray(self.segments, dtype=float)
            if self._boundary_segments_array.size == 0:
                empty = np.empty((0, 2), dtype=float)
                return empty, empty
            self._segment_starts = self._boundary_segments_array[:, 0, :]
            self._segment_ends = self._boundary_segments_array[:, 1, :]
        return self._segment_starts, self._segment_ends
    
    def get_nearby_segments(self, point, radius=None):
        """
        Get segments that are near a given point using spatial partitioning.
        
        Args:
            point: tuple (x, y) - the point to check near
            radius: float - radius to search within (defaults to max sensor distance + some margin)
        
        Returns:
            list: indices of segments that might intersect with the area around the point
        """
        if self._spatial_grid is None:
            # Fallback: return all segment indices
            return list(range(len(self.segments)))
        
        if radius is None:
            radius = config.SENSOR_MAX_DISTANCE + 50  # Default to sensor range plus margin
        
        x, y = point
        
        # Find grid cells that the area around this point intersects
        min_cell_x = int(np.floor((x - radius) / self._grid_cell_size))
        max_cell_x = int(np.floor((x + radius) / self._grid_cell_size))
        min_cell_y = int(np.floor((y - radius) / self._grid_cell_size))
        max_cell_y = int(np.floor((y + radius) / self._grid_cell_size))
        
        # Ensure we're within grid bounds
        grid_width = int(np.ceil(self.screen_width / self._grid_cell_size))
        grid_height = int(np.ceil(self.screen_height / self._grid_cell_size))
        
        min_cell_x = max(0, min_cell_x)
        max_cell_x = min(grid_width - 1, max_cell_x)
        min_cell_y = max(0, min_cell_y)
        max_cell_y = min(grid_height - 1, max_cell_y)
        
        # Collect all unique segment indices from nearby cells
        # Query cost scales with nearby buckets, not total track segment count.
        nearby_segments = set()
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                cell_key = (cell_x, cell_y)
                if cell_key in self._spatial_grid:
                    nearby_segments.update(self._spatial_grid[cell_key])
        
        return list(nearby_segments)
    
    def check_collision(self, position, radius=15):
        """
        Check if a point (with radius) is colliding with track boundaries.
        
        Args:
            position: tuple (x, y) - position to check
            radius: float - radius of the object
        
        Returns:
            tuple: (is_colliding, collision_info)
        """
        x, y = position
        
        # Use spatial partitioning to check only nearby segments
        nearby_segment_indices = None
        if self._spatial_grid is not None:
            nearby_segment_indices = self.get_nearby_segments(position, radius + 10)
        
        # Determine which segments to check
        if nearby_segment_indices is not None and len(nearby_segment_indices) < len(self.segments):
            segments_to_check = [self.segments[i] for i in nearby_segment_indices]
        else:
            segments_to_check = self.segments
        
        # Check against each relevant boundary segment
        for seg_start, seg_end in segments_to_check:
            # Calculate distance from position to segment
            from .physics import point_to_line_distance
            distance = point_to_line_distance(position, seg_start, seg_end)
            
            if distance < radius:
                return True, {'type': 'wall', 'segment': (seg_start, seg_end)}
        
        # Check if inside the track area (between inner and outer boundaries)
        # We use a simpler check: if not colliding with boundaries and within rough bounds
        # This could be improved with a proper point-in-polygon check
        
        return False, None
    
    def get_progress(self, position, last_idx_hint=None, return_idx=False):
        """
        Calculate the progress along the track (0-1).
        
        Args:
            position: tuple (x, y) - current position
            last_idx_hint: int - optional previous closest segment index
            return_idx: bool - return (progress, closest_idx) if True
        
        Returns:
            float: progress from 0 to 1, or tuple if return_idx is True
        """
        # Use cached data for performance
        if (self._cached_waypoints_array is None or 
            self._cached_segment_lengths is None or 
            self._cached_total_length is None):
            self._cache_track_data()
        
        position = np.array(position)
        n = self._cached_waypoints_count
        waypoints_array = self._cached_waypoints_array
        segment_lengths = self._cached_segment_lengths
        total_length = self._cached_total_length
        
        hint = self._last_progress_idx if last_idx_hint is None else last_idx_hint
        if hint is None:
            search_indices = np.arange(n)
        else:
            # Car usually advances locally, so search near last segment before global fallback.
            search_indices = (int(hint) + np.arange(-5, 6)) % n

        closest_segment_idx, closest_point_on_segment, closest_distance = (
            self._closest_progress_segment(position, search_indices)
        )

        if hint is not None and closest_distance > self.track_width:
            closest_segment_idx, closest_point_on_segment, _ = (
                self._closest_progress_segment(position, np.arange(n))
            )

        self._last_progress_idx = closest_segment_idx
        
        # Calculate distance from start along the track using cached segment lengths
        distance_from_start = 0.0
        if closest_segment_idx > 0:
            # Sum all segment lengths before the closest segment
            distance_from_start = float(np.sum(segment_lengths[:closest_segment_idx]))
        
        # Add distance along the closest segment
        p1 = waypoints_array[closest_segment_idx]
        if closest_point_on_segment is not None:
            # Progress comes from projection onto centerline, not raw Euclidean distance to start.
            distance_from_start += float(np.linalg.norm(closest_point_on_segment - p1))

        # Return progress (0-1)
        if total_length > 0:
            start_distance = float(np.sum(segment_lengths[:self.start_idx])) if self.start_idx > 0 else 0.0
            distance_from_start = (distance_from_start - start_distance) % total_length
            progress = distance_from_start / total_length
            progress = float(progress)
            return (progress, closest_segment_idx) if return_idx else progress
        
        return (0.0, closest_segment_idx) if return_idx else 0.0

    def _closest_progress_segment(self, position, indices):
        """Find closest center-line segment among candidate indices."""
        n = self._cached_waypoints_count
        waypoints = self._cached_waypoints_array
        indices = np.asarray(indices, dtype=np.int64)
        p1 = waypoints[indices]
        p2 = waypoints[(indices + 1) % n]
        edges = p2 - p1
        ap = position - p1
        denom = np.einsum('ij,ij->i', edges, edges)
        t = np.zeros(len(indices), dtype=float)
        valid = denom > 1e-12
        # Dot product projects point onto each segment's direction vector.
        t[valid] = np.einsum('ij,ij->i', ap[valid], edges[valid]) / denom[valid]
        # Clamp keeps closest point on finite segment instead of infinite line.
        t = np.clip(t, 0.0, 1.0)
        # Whole batch runs vectorized in NumPy, so many segment tests happen in one pass.
        closest_points = p1 + edges * t[:, None]
        distances = np.linalg.norm(closest_points - position, axis=1)
        best = int(np.argmin(distances))
        return int(indices[best]), closest_points[best], float(distances[best])

    def reset_progress_hint(self):
        """Reset cached closest segment hint used by get_progress."""
        self._last_progress_idx = None
    
    def check_checkpoint(self, position, last_checkpoint_idx):
        """
        Check if a position has passed a checkpoint.
        
        Args:
            position: tuple (x, y) - current position
            last_checkpoint_idx: int - index of last reached checkpoint
        
        Returns:
            tuple: (new_checkpoint_idx, is_finish_line)
        """
        next_checkpoint_idx = last_checkpoint_idx + 1
        for checkpoint in self.checkpoints:
            checkpoint_idx = checkpoint['index']
            if checkpoint_idx != next_checkpoint_idx:
                # Ordered checkpoints stop shortcutting lap by touching later markers first.
                continue

            # Calculate distance from position to the next expected checkpoint
            cx, cy = checkpoint['position']
            px, py = position
            distance = np.sqrt((px - cx)**2 + (py - cy)**2)

            if distance < self.track_width * 0.8:  # Wide threshold for checkpoint
                is_finish = checkpoint.get('is_finish', False)
                return checkpoint_idx, is_finish
        
        return last_checkpoint_idx, False
    
    def draw(self, screen):
        """
        Draw the track on a pygame screen.
        
        Args:
            screen: pygame.Surface - surface to draw on
        """
        colors = config.Colors
        
        # Draw grass background
        pygame.draw.rect(screen, colors.GRASS_COLOR, 
                        (0, 0, self.screen_width, self.screen_height))
        
        # Draw control points (for debugging)
        if hasattr(self, 'control_points') and self.control_points:
            for cp in self.control_points:
                pygame.draw.circle(screen, config.Colors.RED, 
                                  (int(cp[0]), int(cp[1])), 5)
        
        # Draw track surface as filled ring (outer polygon filled, inner cut with grass)
        if len(self.inner_boundary) >= 3 and len(self.outer_boundary) >= 3:
            # Fill outer boundary with road color
            pygame.draw.polygon(screen, colors.ROAD_COLOR, self.outer_boundary)
            # Cut out inner area with grass to form the ring
            # Two fills fake polygon-with-hole, since pygame polygons do not support holes directly.
            pygame.draw.polygon(screen, colors.GRASS_COLOR, self.inner_boundary)
            
            # Draw track boundaries (white lines) — closed=True for seamless loop
            pygame.draw.lines(screen, colors.WHITE, True, self.inner_boundary, 3)
            pygame.draw.lines(screen, colors.WHITE, True, self.outer_boundary, 3)
        
        # Draw center line (dashed)
        for i in range(len(self.waypoints)):
            p1 = self.waypoints[i]
            p2 = self.waypoints[(i + 1) % len(self.waypoints)]
            # Modulo closes loop so final dash segment reconnects to first waypoint cleanly.
            
            # Draw dashed line
            segment_length = np.linalg.norm(np.array(p2) - np.array(p1))
            num_dashes = int(segment_length / 20)
            
            for j in range(num_dashes):
                t1 = j / num_dashes
                t2 = (j + 0.5) / num_dashes
                
                x1 = p1[0] + (p2[0] - p1[0]) * t1
                y1 = p1[1] + (p2[1] - p1[1]) * t1
                x2 = p1[0] + (p2[0] - p1[0]) * t2
                y2 = p1[1] + (p2[1] - p1[1]) * t2
                
                pygame.draw.line(screen, colors.WHITE, (x1, y1), (x2, y2), 2)
        
        # Draw checkpoints
        for checkpoint in self.checkpoints:
            pos = checkpoint['position']
            if checkpoint.get('is_finish', False):
                # Draw finish line
                if self.finish_line:
                    pygame.draw.line(screen, colors.FINISH_LINE_COLOR, 
                                    self.finish_line[0], self.finish_line[1], 5)
            else:
                pygame.draw.circle(screen, colors.CHECKPOINT_COLOR, 
                                  (int(pos[0]), int(pos[1])), 8)
        
        # Draw start position
        pygame.draw.circle(screen, colors.GREEN, 
                          (int(self.start_position[0]), int(self.start_position[1])), 10)
    
    def get_random_start_position(self):
        """
        Get a random starting position on the track.
        
        Returns:
            tuple: (x, y, angle) - position and starting angle
        """
        # Pick a random waypoint
        idx = self._integers(0, len(self.waypoints))
        p1 = self.waypoints[idx]
        p2 = self.waypoints[(idx + 1) % len(self.waypoints)]
        
        # Random offset along the segment
        offset = self._uniform(0, 1)
        
        position = self._get_center_point(p1, p2, offset)
        
        # Calculate angle
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(dy, dx)
        
        return position, angle

    def _uniform(self, low, high):
        if self._rng is not None:
            return self._rng.uniform(low, high)
        return np.random.uniform(low, high)

    def _integers(self, low, high):
        if self._rng is not None:
            return int(self._rng.integers(low, high))
        return int(np.random.randint(low, high))
