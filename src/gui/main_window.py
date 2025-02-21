import customtkinter as ctk
from typing import Callable
import cv2
from PIL import Image, ImageTk
import os
from .training_interface import TrainingInterface

class MainWindow:
    def __init__(self, headless=False):
        self.headless = headless  # Set headless first
        if not self.headless:
            try:
                self.window = ctk.CTk()
                self.window.title("The Gatherer AI - Your Smart Resource Gathering Assistant")
                self.window.geometry("1200x800")
                self.setup_ui()
            except Exception as e:
                print(f"Error initializing GUI: {e}")
                self.window = None
                self.headless = True  # Fall back to headless mode
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

        ctk.CTkLabel(
            sim_frame,
            text="AI Learning Simulation",
            font=("Helvetica", 14, "bold")
        ).pack()

        self.sim_button = ctk.CTkButton(
            sim_frame,
            text="▶ Start Simulation",
            command=self.toggle_simulation,
            fg_color="blue",
            hover_color="dark blue"
        )
        self.sim_button.pack(pady=5)

    def toggle_simulation(self):
        try:
            self.simulation_active = not self.simulation_active
            if hasattr(self, 'sim_button'):
                if self.simulation_active:
                    self.sim_button.configure(text="⬛ Stop Simulation")
                    self.start_simulation()
                else:
                    self.sim_button.configure(text="▶ Start Simulation")
                    self.stop_simulation()
        except Exception as e:
            print(f"Error toggling simulation: {e}")

    def start_simulation(self):
        """Simulate AI learning process"""
        try:
            if not hasattr(self, 'sim_progress'):
                return

            self.sim_progress.set(0)
            if hasattr(self, 'learning_label'):
                self.learning_label.configure(text="AI Learning: Observing user actions...")

            # Update progress periodically to simulate learning
            if self.simulation_active and hasattr(self, 'window'):
                current = self.sim_progress.get()
                if current < 1.0:
                    self.sim_progress.set(current + 0.1)
                    self.update_learning_status(current + 0.1)
                    if self.window:
                        self.window.after(1000, self.start_simulation)
        except Exception as e:
            print(f"Error in simulation: {e}")
            self.stop_simulation()

    def stop_simulation(self):
        try:
            if hasattr(self, 'learning_label'):
                self.learning_label.configure(text="AI Learning: Idle")
            if hasattr(self, 'sim_progress'):
                self.sim_progress.set(0)
        except Exception as e:
            print(f"Error stopping simulation: {e}")

    def update_learning_status(self, progress):
        try:
            status_messages = [
                "Analyzing movement patterns...",
                "Learning resource locations...",
                "Optimizing gathering routes...",
                "Improving detection accuracy...",
                "Finalizing learned behaviors..."
            ]

            index = min(int(progress * 5), 4)
            if hasattr(self, 'learning_label'):
                self.learning_label.configure(text=f"AI Learning: {status_messages[index]}")
        except Exception as e:
            print(f"Error updating learning status: {e}")


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