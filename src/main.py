import os
import sys
import threading
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

try:
    from src.ai_model.detector import ResourceDetector
    from src.utils.screen_capture import ScreenCapture
    from src.utils.bot_controller import BotController
    from src.gui.main_window import MainWindow
    from src.config import Config
    logger.info("Successfully imported all required modules")
except Exception as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

class GathererAI:
    def __init__(self, headless=False):
        self.headless = headless
        self.running = False
        self.detection_thread = None
        self.bot_controller = None
        logger.info(f"Initializing GathererAI (Headless: {headless})")

        try:
            self.config = Config.load()
            logger.info("Config loaded successfully")

            self.detector = ResourceDetector(self.config.model_path)
            logger.info("Resource detector initialized")

            self.screen_capture = ScreenCapture(headless=headless)
            logger.info("Screen capture initialized")

            self.bot_controller = BotController()
            logger.info("Bot controller initialized")

            if not self.headless:
                try:
                    self.main_window = MainWindow(headless=headless)
                    logger.info("Main window initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing GUI: {e}")
                    logger.warning("Falling back to headless mode")
                    self.headless = True
                    self.main_window = None
            else:
                self.main_window = None

            if self.bot_controller is None:
                raise ValueError("Failed to initialize bot controller")

            self.setup_callbacks()
            logger.info("GathererAI initialization completed")

        except Exception as e:
            logger.error(f"Error initializing GathererAI: {e}")
            if not self.headless:
                sys.exit(1)

    def setup_callbacks(self):
        if self.main_window:
            logger.debug("Setting up bot callbacks")
            self.main_window.set_bot_callback(self.handle_bot_control)

    def handle_bot_control(self, action: str):
        logger.info(f"Bot control action received: {action}")
        if action == "start":
            self.start_bot()
        elif action == "stop":
            self.stop_bot()

    def start_bot(self):
        if not self.running and self.bot_controller:
            logger.info("Starting bot")
            self.running = True
            self.bot_controller.start()
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.start()
            logger.info("Bot started successfully")
        else:
            logger.warning("Bot controller not initialized or already running")

    def stop_bot(self):
        logger.info("Stopping bot")
        self.running = False
        if self.bot_controller:
            self.bot_controller.stop()
        if self.detection_thread:
            self.detection_thread.join()
        logger.info("Bot stopped successfully")

    def detection_loop(self):
        logger.debug("Starting detection loop")
        while self.running:
            try:
                frame = self.screen_capture.capture_window(self.config.window_title)
                if frame is None:
                    logger.warning("Failed to capture frame, retrying...")
                    time.sleep(1)
                    continue

                detections = self.detector.detect_resources(frame)
                logger.debug(f"Found {len(detections)} resources")

                if not self.headless:
                    preview_frame = self.detector.visualize_detections(frame, detections)
                    self.main_window.update_preview(preview_frame)

                if detections and self.bot_controller:
                    resource_locations = [
                        (int((box[0] + box[2])/2), int((box[1] + box[3])/2), conf)
                        for _, conf, box in detections
                    ]
                    self.bot_controller.gather_resources(resource_locations)

                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1)

    def run(self):
        try:
            if self.headless:
                logger.info("Running in headless mode")
                if self.bot_controller:
                    self.start_bot()
                    while True:
                        time.sleep(1)
            else:
                logger.info("Running in GUI mode")
                self.main_window.run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.stop_bot()
        except Exception as e:
            logger.error(f"Error running application: {e}")
            self.stop_bot()
            if not self.headless:
                sys.exit(1)

if __name__ == "__main__":
    try:
        headless = "--headless" in sys.argv
        logger.info(f"Starting application (Headless: {headless})")
        app = GathererAI(headless=headless)
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)