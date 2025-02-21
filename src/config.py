from dataclasses import dataclass
from typing import Dict, Any
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    # AI Model settings
    model_path: str = "models/resource_detector.pth"
    detection_threshold: float = 0.7

    # OpenAI settings
    openai_model: str = "gpt-4-vision-preview"
    max_tokens: int = 1000
    temperature: float = 0.7

    # Bot settings
    movement_delay: float = 1.5
    click_delay: float = 0.2

    # Window settings
    window_title: str = "Albion Online Client"

    # Training settings
    training_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 0.005

    def __post_init__(self):
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # Load settings from environment variables if they exist
        if os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if os.getenv("DETECTION_THRESHOLD"):
            self.detection_threshold = float(os.getenv("DETECTION_THRESHOLD"))

        if os.getenv("TRAINING_EPOCHS"):
            self.training_epochs = int(os.getenv("TRAINING_EPOCHS"))

    @classmethod
    def load(cls, path: str = "config.json") -> 'Config':
        if not os.path.exists(path):
            config = cls()
            config.save(path)
            return config

        with open(path, 'r') as f:
            data = json.load(f)
            return cls(**data)

    def save(self, path: str = "config.json"):
        # Don't save sensitive information to the config file
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.endswith('_key') and not k.endswith('_token')}

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def update(self, settings: Dict[str, Any]):
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)