import customtkinter as ctk
from typing import Callable
import cv2
from PIL import Image, ImageTk
import os
from .training_interface import TrainingInterface
import logging

logger = logging.getLogger(__name__)

class MainWindow:
    def __init__(self, headless=False):
        # Set display environment variable for Replit
        os.environ['DISPLAY'] = ':0'

        self.headless = headless
        if not self.headless:
            try:
                # Initialize customtkinter with Replit-compatible settings
                ctk.set_appearance_mode("dark")
                ctk.set_default_color_theme("blue")

                self.window = ctk.CTk()
                self.window.title("The Gatherer AI - Your Smart Resource Gathering Assistant")
                self.window.geometry("1200x800")

                # Force window to stay on top for VNC visibility
                self.window.attributes('-topmost', True)

                self.setup_ui()
                logger.info("GUI initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing GUI: {e}")
                self.window = None
                self.headless = True
        else:
            self.window = None

        self.bot_callback = None
        self.simulation_active = False

    def setup_ui(self):
        if self.headless:
            return

        try:
            # Create main frames with improved layout
            self.control_frame = ctk.CTkFrame(self.window)
            self.control_frame.pack(side="left", fill="y", padx=20, pady=20)

            self.preview_frame = ctk.CTkFrame(self.window)
            self.preview_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

            # Help section at the top
            self.setup_help_section()

            # Control elements
            self.setup_control_panel()

            # Preview area with AI feedback
            self.setup_preview_area()

            # AI Learning Progress
            self.setup_learning_progress()

        except Exception as e:
            print(f"Error setting up UI: {e}")
            self.headless = True
            self.window = None

    def setup_help_section(self):
        help_frame = ctk.CTkFrame(self.control_frame)
        help_frame.pack(pady=10, fill="x")

        help_label = ctk.CTkLabel(
            help_frame,
            text="Welcome to The Gatherer AI!",
            font=("Helvetica", 16, "bold")
        )
        help_label.pack(pady=5)

        instructions = """
1. Select your desired resource type
2. Choose your game resolution
3. Click 'Start Bot' to begin gathering
4. The AI will learn from your actions
5. Use 'Train New Model' for custom training
        """

        help_text = ctk.CTkLabel(
            help_frame,
            text=instructions,
            justify="left",
            wraplength=250
        )
        help_text.pack(pady=5)

    def setup_control_panel(self):
        if self.headless:
            return

        try:
            # Bot controls with improved styling
            self.start_button = ctk.CTkButton(
                self.control_frame,
                text="▶ Start Bot",
                command=lambda: self.bot_callback("start") if self.bot_callback else None,
                fg_color="green",
                hover_color="dark green"
            )
            self.start_button.pack(pady=10)

            self.stop_button = ctk.CTkButton(
                self.control_frame,
                text="⬛ Stop Bot",
                command=lambda: self.bot_callback("stop") if self.bot_callback else None,
                fg_color="red",
                hover_color="dark red"
            )
            self.stop_button.pack(pady=5)

            # Material selection with improved visuals
            material_frame = ctk.CTkFrame(self.control_frame)
            material_frame.pack(pady=15, fill="x")

            ctk.CTkLabel(
                material_frame,
                text="Resource Type",
                font=("Helvetica", 14, "bold")
            ).pack()

            self.material_selector = ctk.CTkComboBox(
                material_frame,
                values=["ore", "fish", "flower", "hide", "stone", "wood"],
                button_color="navy",
                button_hover_color="royal blue"
            )
            self.material_selector.pack(pady=5)

            # Resolution settings with tooltips
            resolution_frame = ctk.CTkFrame(self.control_frame)
            resolution_frame.pack(pady=15, fill="x")

            ctk.CTkLabel(
                resolution_frame,
                text="Game Resolution",
                font=("Helvetica", 14, "bold")
            ).pack()

            self.resolution_selector = ctk.CTkComboBox(
                resolution_frame,
                values=["1920x1080", "2560x1440", "3840x2160"]
            )
            self.resolution_selector.pack(pady=5)

            # Simulation controls
            self.setup_simulation_controls()

            # Training controls with improved visibility
            self.train_button = ctk.CTkButton(
                self.control_frame,
                text="🎓 Train New Model",
                command=self.open_training_interface,
                fg_color="purple",
                hover_color="dark purple"
            )
            self.train_button.pack(pady=10)

            # Settings with improved layout
            self.setup_settings_panel()

        except Exception as e:
            print(f"Error setting up control panel: {e}")

    def setup_simulation_controls(self):
        sim_frame = ctk.CTkFrame(self.control_frame)
        sim_frame.pack(pady=15, fill="x")

        # Add title with improved styling
        title_label = ctk.CTkLabel(
            sim_frame,
            text="AI Learning Simulation",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(5,10))

        # Add description
        desc_label = ctk.CTkLabel(
            sim_frame,
            text="Watch the AI learn and improve its gathering strategies",
            wraplength=250,
            font=("Helvetica", 12)
        )
        desc_label.pack(pady=(0,10))

        # Improved button styling
        self.sim_button = ctk.CTkButton(
            sim_frame,
            text="▶ Start Simulation",
            command=self.toggle_simulation,
            font=("Helvetica", 13),
            height=40,
            fg_color="blue",
            hover_color="dark blue"
        )
        self.sim_button.pack(pady=5)

        # Add progress indicators
        self.sim_progress = ctk.CTkProgressBar(sim_frame)
        self.sim_progress.pack(pady=10, padx=20, fill="x")
        self.sim_progress.set(0)

        # Add status label with more space
        self.learning_label = ctk.CTkLabel(
            sim_frame,
            text="AI Learning: Ready to start",
            font=("Helvetica", 12),
            wraplength=250,
            height=50
        )
        self.learning_label.pack(pady=5)

    def toggle_simulation(self):
        try:
            self.simulation_active = not self.simulation_active
            if hasattr(self, 'sim_button'):
                if self.simulation_active:
                    logger.info("Starting AI learning simulation")
                    self.sim_button.configure(text="⬛ Stop Simulation")
                    self.start_simulation()
                else:
                    logger.info("Stopping AI learning simulation")
                    self.sim_button.configure(text="▶ Start Simulation")
                    self.stop_simulation()
        except Exception as e:
            logger.error(f"Error toggling simulation: {e}")

    def start_simulation(self):
        """Enhanced simulation with more detailed feedback"""
        try:
            if not hasattr(self, 'sim_progress'):
                return

            self.sim_progress.set(0)
            if hasattr(self, 'learning_label'):
                logger.info("Initializing AI learning simulation")
                self.learning_label.configure(
                    text="AI Learning: Initializing simulation...\nPreparing learning environment"
                )

            # Update progress periodically with smoother transitions
            if self.simulation_active and hasattr(self, 'window'):
                current = self.sim_progress.get()
                if current < 1.0:
                    # Smaller increments for smoother animation
                    increment = 0.05
                    self.sim_progress.set(current + increment)
                    self.update_learning_status(current + increment)
                    logger.debug(f"Simulation progress: {(current + increment) * 100:.1f}%")

                    # Shorter interval for more frequent updates
                    if self.window:
                        self.window.after(500, self.start_simulation)
                else:
                    # Show completion message
                    logger.info("AI learning simulation completed")
                    self.learning_label.configure(
                        text="AI Learning: Complete!\nNew behaviors have been learned"
                    )
                    self.sim_progress.configure(progress_color="green")

                    # Reset simulation after a delay
                    self.window.after(3000, self.stop_simulation)

        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            self.stop_simulation()

    def stop_simulation(self):
        try:
            logger.info("Stopping AI learning simulation")
            if hasattr(self, 'learning_label'):
                self.learning_label.configure(text="AI Learning: Idle")
            if hasattr(self, 'sim_progress'):
                self.sim_progress.set(0)
                self.sim_progress.configure(progress_color="blue")
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")

    def update_learning_status(self, progress):
        """Enhanced learning status updates with more detailed messages"""
        try:
            status_messages = [
                ("Analyzing movement patterns", "Learning optimal pathfinding between resources..."),
                ("Learning resource locations", "Building memory of resource spawn points..."),
                ("Optimizing gathering routes", "Calculating efficient gathering sequences..."),
                ("Improving detection accuracy", "Fine-tuning visual recognition model..."),
                ("Finalizing learned behaviors", "Consolidating learned patterns into behavior model...")
            ]

            index = min(int(progress * 5), 4)
            main_status, detail = status_messages[index]

            logger.debug(f"Learning status update: {main_status} - {detail}")

            if hasattr(self, 'learning_label'):
                self.learning_label.configure(
                    text=f"AI Learning: {main_status}\n{detail}"
                )

            # Update UI to show learning progress visually
            if hasattr(self, 'sim_progress'):
                self.sim_progress.set(progress)

                # Add color feedback based on progress
                color = "green" if progress > 0.8 else "orange" if progress > 0.4 else "blue"
                self.sim_progress.configure(progress_color=color)

        except Exception as e:
            logger.error(f"Error updating learning status: {e}")


    def setup_settings_panel(self):
        if self.headless:
            return

        try:
            settings_frame = ctk.CTkFrame(self.control_frame)
            settings_frame.pack(pady=15, fill="x")

            ctk.CTkLabel(
                settings_frame,
                text="Advanced Settings",
                font=("Helvetica", 14, "bold")
            ).pack()

            # Detection confidence threshold
            ctk.CTkLabel(settings_frame, text="Detection Confidence").pack()
            self.threshold_slider = ctk.CTkSlider(
                settings_frame,
                from_=0,
                to=1,
                number_of_steps=100
            )
            self.threshold_slider.pack()

            # Detection interval
            ctk.CTkLabel(settings_frame, text="Detection Interval (ms)").pack()
            self.interval_entry = ctk.CTkEntry(settings_frame)
            self.interval_entry.pack()
            self.interval_entry.insert(0, "1000")

        except Exception as e:
            print(f"Error setting up settings panel: {e}")

    def setup_learning_progress(self):
        learning_frame = ctk.CTkFrame(self.control_frame)
        learning_frame.pack(pady=15, fill="x")

        self.learning_label = ctk.CTkLabel(
            learning_frame,
            text="AI Learning: Idle",
            font=("Helvetica", 12)
        )
        self.learning_label.pack()

        self.sim_progress = ctk.CTkProgressBar(learning_frame)
        self.sim_progress.pack(pady=5)
        self.sim_progress.set(0)

    def setup_preview_area(self):
        if self.headless:
            return

        try:
            preview_label = ctk.CTkLabel(
                self.preview_frame,
                text="Game Preview",
                font=("Helvetica", 16, "bold")
            )
            preview_label.pack(pady=10)

            self.preview_label = ctk.CTkLabel(self.preview_frame, text="")
            self.preview_label.pack(expand=True)

        except Exception as e:
            print(f"Error setting up preview area: {e}")

    def update_preview(self, frame):
        if self.headless or frame is None:
            return

        try:
            # Convert OpenCV frame to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)

            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
        except Exception as e:
            print(f"Error updating preview: {e}")

    def set_bot_callback(self, callback: Callable):
        self.bot_callback = callback

    def open_training_interface(self):
        if self.headless:
            return
        try:
            training_window = TrainingInterface(self.window)
            training_window.run()
        except Exception as e:
            print(f"Error opening training interface: {e}")

    def run(self):
        if not self.headless and self.window:
            try:
                self.window.mainloop()
            except Exception as e:
                print(f"Error in main loop: {e}")
                self.headless = True