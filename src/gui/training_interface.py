import customtkinter as ctk
from tkinter import filedialog
import os
from typing import List, Dict
import cv2
from PIL import Image, ImageTk
from ..ai_model.ai_service import AIService
from ..config import Config
import threading
import logging
import time

logger = logging.getLogger(__name__)

class TrainingInterface:
    def __init__(self, parent=None):
        self.window = ctk.CTkToplevel(parent) if parent else ctk.CTk()
        self.window.title("AI Training Interface")
        self.window.geometry("1200x800")

        self.config = Config.load()
        self.ai_service = AIService(self.config)

        self.training_images: List[str] = []
        self.annotations: Dict = {}
        self.current_image_index = 0

        try:
            self.setup_ui()
            logger.info("Training interface initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up training interface: {e}")
            raise

    def setup_ui(self):
        # Main container with modern styling
        self.main_container = ctk.CTkFrame(self.window, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel for controls
        self.control_panel = ctk.CTkFrame(self.main_container)
        self.control_panel.pack(side="left", fill="y", padx=10)

        # Right panel for image preview
        self.preview_panel = ctk.CTkFrame(self.main_container)
        self.preview_panel.pack(side="right", fill="both", expand=True, padx=10)

        self.setup_control_panel()
        self.setup_preview_panel()

    def setup_control_panel(self):
        # Welcome message
        welcome_label = ctk.CTkLabel(
            self.control_panel,
            text="Welcome to AI Training!",
            font=("Helvetica", 20, "bold")
        )
        welcome_label.pack(pady=10)

        help_text = """
        Train your AI to recognize different resources:
        1. Load training images
        2. Let AI analyze each image
        3. Verify resource types
        4. Start the training process

        The AI will learn from your examples!
        """
        help_label = ctk.CTkLabel(
            self.control_panel,
            text=help_text,
            wraplength=250,
            justify="left"
        )
        help_label.pack(pady=10)

        # Image loading with improved styling
        self.load_button = ctk.CTkButton(
            self.control_panel,
            text="📁 Load Images",
            command=self.load_images,
            font=("Helvetica", 14),
            height=40
        )
        self.load_button.pack(pady=10)

        # Auto-detect button with modern styling
        self.detect_button = ctk.CTkButton(
            self.control_panel,
            text="🔍 Auto-Detect Resource",
            command=self.auto_detect_resource,
            font=("Helvetica", 14),
            height=40
        )
        self.detect_button.pack(pady=10)

        # Resource class selection
        class_frame = ctk.CTkFrame(self.control_panel)
        class_frame.pack(pady=10, fill="x")

        ctk.CTkLabel(
            class_frame,
            text="Resource Type:",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)

        self.class_selector = ctk.CTkComboBox(
            class_frame,
            values=["Stone", "Wood", "Ore", "Fiber", "Hide", "Flower"],
            font=("Helvetica", 12)
        )
        self.class_selector.pack(pady=5)

        # AI Analysis display with better formatting
        analysis_frame = ctk.CTkFrame(self.control_panel)
        analysis_frame.pack(pady=10, fill="x")

        ctk.CTkLabel(
            analysis_frame,
            text="AI Analysis",
            font=("Helvetica", 14, "bold")
        ).pack(pady=5)

        self.analysis_label = ctk.CTkLabel(
            analysis_frame,
            text="No image analyzed yet",
            wraplength=250,
            height=100,
            font=("Helvetica", 12)
        )
        self.analysis_label.pack(pady=5)

        # Training controls with progress indicators
        training_frame = ctk.CTkFrame(self.control_panel)
        training_frame.pack(pady=10, fill="x")

        self.train_button = ctk.CTkButton(
            training_frame,
            text="🎓 Start Training",
            command=self.start_training,
            font=("Helvetica", 14),
            height=40,
            fg_color="green",
            hover_color="dark green"
        )
        self.train_button.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(training_frame)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(
            training_frame,
            text="Ready to train",
            font=("Helvetica", 12)
        )
        self.status_label.pack(pady=5)

    def setup_preview_panel(self):
        preview_label = ctk.CTkLabel(
            self.preview_panel,
            text="Training Images",
            font=("Helvetica", 20, "bold")
        )
        preview_label.pack(pady=10)

        self.preview_label = ctk.CTkLabel(self.preview_panel, text="No image loaded")
        self.preview_label.pack(expand=True)

        # Navigation controls with modern styling
        nav_frame = ctk.CTkFrame(self.preview_panel, fg_color="transparent")
        nav_frame.pack(fill="x", pady=10)

        self.prev_button = ctk.CTkButton(
            nav_frame,
            text="◀ Previous",
            command=self.prev_image,
            width=120
        )
        self.prev_button.pack(side="left", padx=10)

        self.next_button = ctk.CTkButton(
            nav_frame,
            text="Next ▶",
            command=self.next_image,
            width=120
        )
        self.next_button.pack(side="right", padx=10)

    def auto_detect_resource(self):
        if not self.training_images or self.current_image_index >= len(self.training_images):
            return

        try:
            # Load and convert image
            image_path = self.training_images[self.current_image_index]
            image = Image.open(image_path)

            # Get AI analysis
            analysis_result = self.ai_service.analyze_resource_image(image)

            if "error" not in analysis_result:
                # Update the class selector
                if analysis_result["resource_type"] != "unknown":
                    self.class_selector.set(analysis_result["resource_type"].title())

                # Update analysis display with improved formatting
                self.analysis_label.configure(
                    text=f"AI Analysis:\n{analysis_result['analysis']}"
                )
                self.status_label.configure(
                    text="Resource analyzed successfully!"
                )
            else:
                self.analysis_label.configure(
                    text=f"Error during analysis:\n{analysis_result['error']}"
                )
                self.status_label.configure(
                    text="Analysis failed"
                )
        except Exception as e:
            logger.error(f"Error in auto detect: {e}")
            self.analysis_label.configure(
                text=f"Error during analysis:\n{str(e)}"
            )

    def show_current_image(self):
        if not self.training_images:
            return

        try:
            image_path = self.training_images[self.current_image_index]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image to fit preview panel while maintaining aspect ratio
            max_size = 800
            height, width = image.shape[:2]
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))

            photo = ImageTk.PhotoImage(Image.fromarray(image))
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo

            # Update status
            self.status_label.configure(
                text=f"Viewing image {self.current_image_index + 1} of {len(self.training_images)}"
            )
        except Exception as e:
            logger.error(f"Error showing image: {e}")
            self.preview_label.configure(text="Error loading image")

    def load_images(self):
        try:
            files = filedialog.askopenfilenames(
                title="Select Training Images",
                filetypes=[("Image files", "*.png *.jpg *.jpeg")]
            )

            if files:
                self.training_images.extend(files)
                self.current_image_index = 0
                self.show_current_image()
                self.auto_detect_resource()

                self.status_label.configure(
                    text=f"Loaded {len(files)} new images"
                )
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            self.status_label.configure(text="Error loading images")

    def start_training(self):
        """Start the model training process with AI guidance"""
        try:
            # Disable training button during process
            self.train_button.configure(state="disabled")
            self.progress_bar.set(0)
            self.status_label.configure(text="Preparing training data...")

            # Prepare training data in a separate thread
            threading.Thread(target=self._train_model_thread).start()

        except Exception as e:
            logger.error(f"Error starting training: {e}")
            self.status_label.configure(text=f"Training error: {str(e)}")
            self.train_button.configure(state="normal")

    def _train_model_thread(self):
        try:
            # Update UI to show progress
            def update_progress(epoch, total_epochs, loss):
                progress = (epoch + 1) / total_epochs
                self.window.after(0, self.progress_bar.set, progress)
                self.window.after(0, self.status_label.configure, 
                                text=f"Training Epoch {epoch+1}/{total_epochs}\nLoss: {loss:.4f}")

            # Start training process
            self.window.after(0, self.status_label.configure, text="Training model...")

            # Training would go here...
            # Simulate training progress for now
            for i in range(10):
                time.sleep(1)  # Simulate training time
                progress = (i + 1) / 10
                loss = 1.0 - progress * 0.5  # Simulate decreasing loss
                self.window.after(0, update_progress, i, 10, loss)

            # Training complete
            self.window.after(0, self.status_label.configure, text="Training completed!")
            self.window.after(0, self.train_button.configure, {"state": "normal"})
            self.window.after(0, self.progress_bar.set, 1.0)

        except Exception as e:
            logger.error(f"Error in training thread: {e}")
            self.window.after(0, self.status_label.configure, text=f"Training error: {str(e)}")
            self.window.after(0, self.train_button.configure, {"state": "normal"})

    def prev_image(self):
        if self.training_images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()
            self.auto_detect_resource()

    def next_image(self):
        if self.training_images and self.current_image_index < len(self.training_images) - 1:
            self.current_image_index += 1
            self.show_current_image()
            self.auto_detect_resource()

    def run(self):
        self.window.mainloop()