import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_mesh_connections


class GazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Extract landmark indices from MediaPipe connections dynamically
        self.left_eye_indices = self._extract_indices_from_connections(face_mesh_connections.FACEMESH_LEFT_EYE)
        self.right_eye_indices = self._extract_indices_from_connections(face_mesh_connections.FACEMESH_RIGHT_EYE)
        self.left_iris_indices = self._extract_indices_from_connections(face_mesh_connections.FACEMESH_LEFT_IRIS)
        self.right_iris_indices = self._extract_indices_from_connections(face_mesh_connections.FACEMESH_RIGHT_IRIS)
        
        # Blink detection - more sensitive to work even when looking left/right
        self.blink_threshold = 0.18  # Slightly higher to catch blinks at extremes
        self.blink_cooldown = 0.15  # Shorter cooldown for faster response
        self.previous_eye_ratio = None
        self.last_blink_time = 0
        
        # Adaptive baseline for horizontal gaze detection
        self.baseline_ratios = []
        self.baseline_max_samples = 30
        self.baseline_avg = None
        self.min_samples = 10
        self.gaze_threshold = 0.015

    def _extract_indices_from_connections(self, connections):
        """Extract unique landmark indices from MediaPipe connection set."""
        indices = set()
        for connection in connections:
            indices.add(connection[0])
            indices.add(connection[1])
        return sorted(list(indices))

    def _get_landmark_point(self, landmarks, idx, image_width=None, image_height=None):
        """Get landmark point as numpy array."""
        if idx >= len(landmarks):
            return None
        
        landmark = landmarks[idx]
        if image_width and image_height:
            return np.array([landmark.x * image_width, landmark.y * image_height])
        return np.array([landmark.x, landmark.y])

    def _get_points_from_indices(self, landmarks, indices, image_width, image_height):
        """Get all points from landmark indices."""
        points = []
        for idx in indices:
            point = self._get_landmark_point(landmarks, idx, image_width, image_height)
            if point is not None:
                points.append(point)
        return np.array(points) if points else None

    def get_iris_positions(self, landmarks, image_width, image_height):
        """Get center positions of both irises."""
        if not landmarks:
            return None, None
        
        left_points = self._get_points_from_indices(landmarks, self.left_iris_indices, image_width, image_height)
        right_points = self._get_points_from_indices(landmarks, self.right_iris_indices, image_width, image_height)
        
        if left_points is None or right_points is None:
            return None, None
        
        return left_points.mean(axis=0), right_points.mean(axis=0)

    def get_eye_boundaries(self, landmarks, image_width, image_height):
        """Get eye boundaries (min/max x and y) from all eye landmarks."""
        if not landmarks:
            return None, None
        
        left_points = self._get_points_from_indices(landmarks, self.left_eye_indices, image_width, image_height)
        right_points = self._get_points_from_indices(landmarks, self.right_eye_indices, image_width, image_height)
        
        if left_points is None or right_points is None:
            return None, None
        
        def get_bounds(points):
            return {
                'left': np.array([points[:, 0].min(), points[points[:, 0].argmin(), 1]]),
                'right': np.array([points[:, 0].max(), points[points[:, 0].argmax(), 1]]),
                'top': np.array([points[points[:, 1].argmin(), 0], points[:, 1].min()]),
                'bottom': np.array([points[points[:, 1].argmax(), 0], points[:, 1].max()]),
            }
        
        return get_bounds(left_points), get_bounds(right_points)

    def detect_blink(self, landmarks):
        """Detect if user is blinking - works even when looking left/right."""
        if not landmarks:
            return False
        
        # Get all eye points and find boundaries dynamically
        left_points = self._get_points_from_indices(landmarks, self.left_eye_indices, None, None)
        right_points = self._get_points_from_indices(landmarks, self.right_eye_indices, None, None)
        
        if left_points is None or right_points is None or len(left_points) == 0 or len(right_points) == 0:
            return False
        
        # Calculate eye opening (vertical distance) - more reliable than ratio
        left_vertical = left_points[:, 1].max() - left_points[:, 1].min()
        right_vertical = right_points[:, 1].max() - right_points[:, 1].min()
        
        # Use minimum of both eyes (if either eye closes, it's a blink)
        min_vertical = min(left_vertical, right_vertical)
        
        # Normalize by average eye width for consistency across distances
        left_horizontal = left_points[:, 0].max() - left_points[:, 0].min()
        right_horizontal = right_points[:, 0].max() - right_points[:, 0].min()
        avg_horizontal = (left_horizontal + right_horizontal) / 2.0
        
        if avg_horizontal == 0:
            return False
        
        # Calculate normalized eye opening (works better at extremes)
        normalized_opening = min_vertical / avg_horizontal
        
        # Detect new blink (transition from open to closed)
        current_time = time.time()
        is_blinking = normalized_opening < self.blink_threshold
        was_open = self.previous_eye_ratio is None or self.previous_eye_ratio >= self.blink_threshold
        
        if is_blinking and was_open and (current_time - self.last_blink_time > self.blink_cooldown):
            self.last_blink_time = current_time
            self.previous_eye_ratio = normalized_opening
            return True
        
        self.previous_eye_ratio = normalized_opening
        return False

    def calculate_horizontal_ratio(self, iris_pos, eye_bounds):
        """Calculate horizontal position of iris within eye (0.0 = left, 1.0 = right)."""
        if iris_pos is None or eye_bounds is None:
            return None
        
        eye_width = eye_bounds['right'][0] - eye_bounds['left'][0]
        if abs(eye_width) < 1e-6:
            return 0.5
        
        ratio = (iris_pos[0] - eye_bounds['left'][0]) / eye_width
        return max(0.0, min(1.0, ratio))

    def _update_baseline(self, avg_ratio):
        """Update rolling baseline for adaptive gaze detection."""
        self.baseline_ratios.append(avg_ratio)
        if len(self.baseline_ratios) > self.baseline_max_samples:
            self.baseline_ratios.pop(0)
        
        if len(self.baseline_ratios) >= self.min_samples:
            self.baseline_avg = sum(self.baseline_ratios) / len(self.baseline_ratios)

    def get_gaze_direction(self, landmarks, image_width, image_height):
        """Detect horizontal gaze direction using adaptive baseline."""
        left_iris, right_iris = self.get_iris_positions(landmarks, image_width, image_height)
        left_bounds, right_bounds = self.get_eye_boundaries(landmarks, image_width, image_height)
        
        if left_iris is None or right_iris is None or left_bounds is None or right_bounds is None:
            return None
        
        # Calculate average horizontal ratio
        left_ratio = self.calculate_horizontal_ratio(left_iris, left_bounds)
        right_ratio = self.calculate_horizontal_ratio(right_iris, right_bounds)
        
        if left_ratio is None or right_ratio is None:
            return None
        
        avg_ratio = (left_ratio + right_ratio) / 2.0
        
        # Update baseline
        self._update_baseline(avg_ratio)
        
        # Need enough samples before detecting direction
        if self.baseline_avg is None:
            return "CENTER"
        
        # Calculate deviation from baseline
        deviation = avg_ratio - self.baseline_avg
        
        if abs(deviation) < self.gaze_threshold:
            return "CENTER"
        return "LEFT" if deviation < 0 else "RIGHT"
