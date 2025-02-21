from dataclasses import dataclass
from typing import Dict, Any, List
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class AILearningConfig:
    # Model architecture parameters
    hidden_layers: List[int] = None  # Number of neurons in each hidden layer
    dropout_rate: float = 0.2        # Dropout rate for regularization
    activation_function: str = "relu" # Activation function for hidden layers

    # Training parameters
    learning_rate: float = 0.005
    batch_size: int = 4
    training_epochs: int = 10
    validation_split: float = 0.2    # Portion of data used for validation
    early_stopping_patience: int = 3  # Number of epochs to wait before early stopping

    # Optimization parameters
    optimizer: str = "adam"          # Optimizer type (adam, sgd, rmsprop)
    momentum: float = 0.9            # Momentum for SGD optimizer
    beta1: float = 0.9              # Beta1 for Adam optimizer
    beta2: float = 0.999            # Beta2 for Adam optimizer

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 32]  # Default architecture

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

    # Advanced AI Learning Configuration
    ai_learning: AILearningConfig = None

    def __post_init__(self):
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # Initialize AI learning config if not provided
        if self.ai_learning is None:
            self.ai_learning = AILearningConfig()

        # Load settings from environment variables if they exist
        if os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if os.getenv("DETECTION_THRESHOLD"):
            self.detection_threshold = float(os.getenv("DETECTION_THRESHOLD"))

        # Load AI learning parameters from environment if available
        if os.getenv("LEARNING_RATE"):
            self.ai_learning.learning_rate = float(os.getenv("LEARNING_RATE"))
        if os.getenv("BATCH_SIZE"):
            self.ai_learning.batch_size = int(os.getenv("BATCH_SIZE"))
        if os.getenv("TRAINING_EPOCHS"):
            self.ai_learning.training_epochs = int(os.getenv("TRAINING_EPOCHS"))

    @classmethod
    def load(cls, path: str = "config.json") -> 'Config':
        if not os.path.exists(path):
            config = cls()
            config.save(path)
            return config

        with open(path, 'r') as f:
            data = json.load(f)
            # Handle nested AILearningConfig
            if 'ai_learning' in data:
                ai_learning_data = data.pop('ai_learning')
                data['ai_learning'] = AILearningConfig(**ai_learning_data)
            return cls(**data)

    def save(self, path: str = "config.json"):
        # Don't save sensitive information to the config file
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.endswith('_key') and not k.endswith('_token')}

        # Convert AILearningConfig to dict for JSON serialization
        if 'ai_learning' in config_dict:
            config_dict['ai_learning'] = vars(config_dict['ai_learning'])

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)

    def update(self, settings: Dict[str, Any]):
        for key, value in settings.items():
            if hasattr(self, key):
                if key == 'ai_learning' and isinstance(value, dict):
                    # Update nested AI learning config
                    for k, v in value.items():
                        if hasattr(self.ai_learning, k):
                            setattr(self.ai_learning, k, v)
                else:
                    setattr(self, key, value)