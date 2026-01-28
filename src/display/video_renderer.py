import cv2


class VideoRenderer:
    """Handles all rendering and display operations on the camera view."""

    def __init__(self, show_debug=True):

        self.show_debug = show_debug

    def render(
        self,
        frame,
        tracker,
        landmarks,
        direction,
        is_blink,
        image_width,
        image_height,
    ):

        # Draw iris in blue
        frame = self._draw_iris(
            frame, tracker, landmarks, image_width, image_height, color=(255, 0, 0), radius=10
        )

        # Draw eye debug lines
        frame = self._draw_eye_debug(frame, tracker, landmarks, image_width, image_height)

        # Draw debug information if enabled
        if self.show_debug:
            frame = self._draw_debug_info(
                frame, tracker, landmarks, image_width, image_height
            )

        # Draw direction/status text
        frame = self._draw_direction_text(frame, direction, is_blink)

        return frame

    def render_no_face(self, frame):

        cv2.putText(
            frame,
            "No face detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return frame

    def _draw_debug_info(self, frame, tracker, landmarks, image_width, image_height):
        left_iris, right_iris = tracker.get_iris_positions(
            landmarks, image_width, image_height
        )
        left_bounds, right_bounds = tracker.get_eye_boundaries(
            landmarks, image_width, image_height
        )

        if (
            left_iris is not None
            and right_iris is not None
            and left_bounds is not None
            and right_bounds is not None
        ):
            left_h = tracker.calculate_horizontal_ratio(left_iris, left_bounds)
            right_h = tracker.calculate_horizontal_ratio(right_iris, right_bounds)

            if left_h is not None and right_h is not None:
                avg_ratio = (left_h + right_h) / 2.0
                baseline = (
                    tracker.baseline_avg if tracker.baseline_avg is not None else 0.0
                )
                deviation = (
                    avg_ratio - baseline if tracker.baseline_avg is not None else 0.0
                )
                samples = len(tracker.baseline_ratios)
                debug_text = f"Avg:{avg_ratio:.3f} Base:{baseline:.3f} Dev:{deviation:.3f} Samples:{samples}"

                cv2.putText(
                    frame,
                    debug_text,
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return frame

    def _draw_iris(self, frame, tracker, landmarks, image_width, image_height, color=(255, 0, 0), radius=8):
        left_iris, right_iris = tracker.get_iris_positions(landmarks, image_width, image_height)
        
        for iris in [left_iris, right_iris]:
            if iris is not None:
                center = (int(iris[0]), int(iris[1]))
                
                # Outer glow effect (cyan)
                cv2.circle(frame, center, radius + 4, (255, 255, 0), 2)
                cv2.circle(frame, center, radius + 2, (200, 255, 100), 2)
                
                # Main iris circle (blue)
                cv2.circle(frame, center, radius, color, -1)
                
                # Inner highlight (bright cyan dot)
                cv2.circle(frame, center, 3, (255, 255, 0), -1)
                
                # Crosshair at iris center (cyan)
                line_length = 8
                cv2.line(frame, 
                        (center[0] - line_length, center[1]), 
                        (center[0] + line_length, center[1]), 
                        (255, 255, 0), 1)
                cv2.line(frame, 
                        (center[0], center[1] - line_length), 
                        (center[0], center[1] + line_length), 
                        (255, 255, 0), 1)
        
        return frame

    def _draw_eye_debug(self, frame, tracker, landmarks, image_width, image_height):
        left_iris, right_iris = tracker.get_iris_positions(landmarks, image_width, image_height)
        left_bounds, right_bounds = tracker.get_eye_boundaries(landmarks, image_width, image_height)
        
        if left_bounds is not None:
            # Draw left eye boundaries
            left_corner_left = (int(left_bounds['left'][0]), int(left_bounds['left'][1]))
            left_corner_right = (int(left_bounds['right'][0]), int(left_bounds['right'][1]))
            left_top = (int(left_bounds['top'][0]), int(left_bounds['top'][1]))
            left_bottom = (int(left_bounds['bottom'][0]), int(left_bounds['bottom'][1]))
            
            # Calculate center y for horizontal line (average of top and bottom)
            center_y = int((left_bounds['top'][1] + left_bounds['bottom'][1]) / 2)
            
            # Draw horizontal line across eye at center y (straight line with glow)
            left_x = int(left_bounds['left'][0])
            right_x = int(left_bounds['right'][0])
            
            # Glow effect for the line
            cv2.line(frame, (left_x, center_y), (right_x, center_y), (0, 200, 0), 4)
            cv2.line(frame, (left_x, center_y), (right_x, center_y), (0, 255, 100), 2)
            
            # Draw eye corner dots with glow effect
            cv2.circle(frame, (left_x, center_y), 7, (0, 200, 0), 2)
            cv2.circle(frame, (left_x, center_y), 5, (0, 255, 100), -1)
            cv2.circle(frame, (right_x, center_y), 7, (0, 200, 0), 2)
            cv2.circle(frame, (right_x, center_y), 5, (0, 255, 100), -1)
            
            # Draw center point of eye with glow
            eye_center_x = (left_x + right_x) // 2
            eye_center_y = center_y
            cv2.circle(frame, (eye_center_x, eye_center_y), 5, (200, 255, 0), 2)
            cv2.circle(frame, (eye_center_x, eye_center_y), 3, (255, 255, 0), -1)
        
        if right_bounds is not None:
            # Draw right eye boundaries
            right_corner_left = (int(right_bounds['left'][0]), int(right_bounds['left'][1]))
            right_corner_right = (int(right_bounds['right'][0]), int(right_bounds['right'][1]))
            right_top = (int(right_bounds['top'][0]), int(right_bounds['top'][1]))
            right_bottom = (int(right_bounds['bottom'][0]), int(right_bounds['bottom'][1]))
            
            # Calculate center y for horizontal line (average of top and bottom)
            center_y = int((right_bounds['top'][1] + right_bounds['bottom'][1]) / 2)
            
            # Draw horizontal line across eye at center y (straight line with glow)
            left_x = int(right_bounds['left'][0])
            right_x = int(right_bounds['right'][0])
            
            # Glow effect for the line
            cv2.line(frame, (left_x, center_y), (right_x, center_y), (0, 200, 0), 4)
            cv2.line(frame, (left_x, center_y), (right_x, center_y), (0, 255, 100), 2)
            
            # Draw eye corner dots with glow effect
            cv2.circle(frame, (left_x, center_y), 7, (0, 200, 0), 2)
            cv2.circle(frame, (left_x, center_y), 5, (0, 255, 100), -1)
            cv2.circle(frame, (right_x, center_y), 7, (0, 200, 0), 2)
            cv2.circle(frame, (right_x, center_y), 5, (0, 255, 100), -1)
            
            # Draw center point of eye with glow
            eye_center_x = (left_x + right_x) // 2
            eye_center_y = center_y
            cv2.circle(frame, (eye_center_x, eye_center_y), 5, (200, 255, 0), 2)
            cv2.circle(frame, (eye_center_x, eye_center_y), 3, (255, 255, 0), -1)
        
        return frame

    def _draw_direction_text(self, frame, direction, is_blink):
        # Combine direction and blink
        if is_blink:
            if direction == "LEFT":
                text = "LEFT + BLINK - JUMP!"
            elif direction == "RIGHT":
                text = "RIGHT + BLINK - JUMP!"
            else:
                text = "BLINK - JUMP!"
            color = (0, 255, 255)  # Yellow/Cyan
        elif direction == "LEFT":
            text = "LEFT"
            color = (0, 165, 255)  # Orange
        elif direction == "RIGHT":
            text = "RIGHT"
            color = (255, 0, 255)  # Magenta
        elif direction == "CENTER":
            text = "CENTER"
            color = (0, 255, 0)  # Green
        else:
            text = "Detecting..."
            color = (255, 255, 255)  # White

        # Draw text with background for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (20, 20),
            (20 + text_width + 10, 20 + text_height + 10),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (30, 50),
            font,
            font_scale,
            color,
            thickness,
        )

        return frame
