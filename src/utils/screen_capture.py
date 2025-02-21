import numpy as np
import cv2
import mss
import mss.tools
from typing import Tuple, Optional
import platform
import os

class ScreenCapture:
    def __init__(self, headless=False):
        self.sct = None
        self.monitor = None
        self.headless = headless

        if not self.headless:
            try:
                if platform.system() != 'Linux' or 'DISPLAY' in os.environ:
                    self.sct = mss.mss()
                    self.monitor = self.sct.monitors[1]  # Primary monitor
            except Exception as e:
                print(f"Warning: Screen capture initialization failed: {e}")
                print("Running in fallback mode - some features may be limited")

    def capture_window(self, window_title: str = None) -> Optional[np.ndarray]:
        """Capture the game window or full screen if window_title is None"""
        if self.headless or self.sct is None:
            # Return a dummy frame for testing in headless mode
            return np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            if window_title:
                # Implementation for specific window capture would go here
                # For now, capturing full screen
                screenshot = self.sct.grab(self.monitor)
            else:
                screenshot = self.sct.grab(self.monitor)

            # Convert to numpy array
            frame = np.array(screenshot)

            # Convert from BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            return frame
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Return dummy frame on error

    def set_capture_area(self, x: int, y: int, width: int, height: int):
        """Set a specific area to capture"""
        if self.headless or self.sct is None:
            return

        self.monitor = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }

    def get_screen_size(self) -> Tuple[int, int]:
        """Return the screen size"""
        if self.headless or self.sct is None:
            return (640, 480)  # Default size for headless mode
        return self.monitor["width"], self.monitor["height"]