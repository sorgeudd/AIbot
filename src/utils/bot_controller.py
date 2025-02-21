import time
from typing import Tuple, List
import random

class BotController:
    def __init__(self):
        self.running = False
        self.movement_cooldown = 1.5  # seconds
        self.last_movement = 0
        self.headless = True

    def start(self):
        """Start the bot"""
        self.running = True

    def stop(self):
        """Stop the bot"""
        self.running = False

    def move_to_resource(self, position: Tuple[int, int], confidence: float = 0.7):
        """Simulate moving to a resource location"""
        if not self.running:
            return

        current_time = time.time()
        if current_time - self.last_movement < self.movement_cooldown:
            return

        try:
            x, y = position
            # In headless mode, just log the movement
            print(f"Would move to resource at {x}, {y} with confidence {confidence}")
            self.last_movement = current_time

        except Exception as e:
            print(f"Error during movement simulation: {e}")

    def gather_resources(self, resource_locations: List[Tuple[int, int, float]]):
        """Process a list of resource locations"""
        if not self.running:
            return

        for x, y, confidence in resource_locations:
            if not self.running:
                break

            self.move_to_resource((x, y), confidence)
            time.sleep(random.uniform(0.5, 1.5))