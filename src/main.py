import cv2
from vision.gaze_tracker import GazeTracker
from input.keyboard_controller import KeyboardController
from display.video_renderer import VideoRenderer


def main():
    """Main function - gaze-controlled simulation."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    tracker = GazeTracker()
    keyboard_controller = KeyboardController(jump_hold_duration=0.6)
    renderer = VideoRenderer(show_debug=True)

    print("=" * 50)
    print("Eye Gaze Tracker - Simulation Mode")
    print("=" * 50)
    print("\nControls:")
    print("- Look LEFT: Press LEFT arrow key")
    print("- Look RIGHT: Press RIGHT arrow key")
    print("- BLINK: Press UP arrow key (jump)")
    print("- Can combine: LEFT+BLINK or RIGHT+BLINK")
    print("\nPress any key in the video window to start keyboard input")
    print("Press 'q' to quit\n")

    # Flag to track if keyboard input is enabled
    keyboard_enabled = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Detect blink
            is_blink = tracker.detect_blink(landmarks)

            # Get gaze direction
            direction = tracker.get_gaze_direction(landmarks, w, h)

            # Only update keyboard controller if enabled
            if keyboard_enabled:
                keyboard_controller.update(direction, is_blink)
            else:
                # Release all keys if keyboard is not enabled
                keyboard_controller.release_all()

            # Render all visual elements
            frame = renderer.render(
                frame, tracker, landmarks, direction, is_blink, w, h
            )
            
            # Show message if keyboard is not enabled
            if not keyboard_enabled:
                cv2.putText(
                    frame,
                    "Press any key to start keyboard input",
                    (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
        else:
            # No face detected - release all keys
            keyboard_controller.release_all()

            # Render no face message
            frame = renderer.render_no_face(frame)
            
            # Show message if keyboard is not enabled
            if not keyboard_enabled:
                cv2.putText(
                    frame,
                    "Press any key to start keyboard input",
                    (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Eye Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key != 0xFF and not keyboard_enabled:  # Any key pressed (0xFF = no key)
            keyboard_enabled = True
            print("Keyboard input enabled!")

    # Clean up
    keyboard_controller.cleanup()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
