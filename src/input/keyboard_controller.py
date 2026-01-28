import time
from pynput.keyboard import Key, Controller


class KeyboardController:
    """Handles keyboard input based on gaze direction and blinks."""

    def __init__(self, jump_hold_duration=0.6):
        self.keyboard = Controller()
        self.jump_hold_duration = jump_hold_duration
        
        # Track currently pressed keys
        self.current_key = None  # LEFT or RIGHT
        self.last_direction = None  # Keep last valid horizontal direction
        self.last_blink_state = False
        
        # Jump timing
        self.jump_start_time = None
        self.jump_key_pressed = False

    def update(self, direction, is_blink):
        current_time = time.time()
        
        # Handle blink (up arrow) - hold for longer duration for high jump
        # Jump works independently of direction keys (can press both simultaneously)
        if is_blink and not self.last_blink_state:
            # Blink detected - start holding the key
            # Press jump key even if direction key is already pressed
            self.keyboard.press(Key.up)
            self.jump_start_time = current_time
            self.jump_key_pressed = True
            print(f"JUMP! (while moving {self.last_direction or 'none'})")
        elif self.jump_key_pressed:
            # Check if we've held the key long enough
            if current_time - self.jump_start_time >= self.jump_hold_duration:
                self.keyboard.release(Key.up)
                self.jump_key_pressed = False
                self.jump_start_time = None

        self.last_blink_state = is_blink

        # Handle direction keys (LEFT/RIGHT)
        # Keep last valid LEFT/RIGHT direction, but allow CENTER to release keys
        if direction == "CENTER":
            # Clear last direction when centering
            self.last_direction = None
            active_direction = "CENTER"
        elif direction in ["LEFT", "RIGHT"]:
            self.last_direction = direction
            active_direction = direction
        else:
            # None or invalid - keep last direction (for extremes)
            active_direction = self.last_direction
        
        target_key = None
        if active_direction == "LEFT":
            target_key = Key.left
        elif active_direction == "RIGHT":
            target_key = Key.right
        # If CENTER or None, target_key stays None (releases key)

        # Only press/release if direction changed
        if target_key != self.current_key:
            # Release old key
            if self.current_key is not None:
                self.keyboard.release(self.current_key)

            # Press new key
            if target_key is not None:
                self.keyboard.press(target_key)

            self.current_key = target_key

    def release_all(self):
        if self.current_key is not None:
            self.keyboard.release(self.current_key)
            self.current_key = None
        
        if self.jump_key_pressed:
            self.keyboard.release(Key.up)
            self.jump_key_pressed = False
            self.jump_start_time = None
            self.last_blink_state = False
        
        self.last_direction = None

    def cleanup(self):
        self.release_all()
