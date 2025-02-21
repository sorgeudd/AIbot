import os
from openai import OpenAI
from typing import List, Dict, Any
import base64
from PIL import Image
import io
import torch

class AIService:
    def __init__(self, config):
        self.config = config
        # Initialize OpenAI client with error handling
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            self.client = None

    def analyze_resource_image(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze a resource image using OpenAI's Vision model"""
        if not self.client:
            return {"error": "OpenAI client not initialized"}

        try:
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this game resource image and identify what type of resource it is (Stone, Wood, Ore, Fish, Flower, Hide). Also describe its visual characteristics."
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{image_base64}"
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            return {
                "analysis": response.choices[0].message.content,
                "resource_type": self._extract_resource_type(response.choices[0].message.content)
            }
        except Exception as e:
            print(f"Error analyzing image with OpenAI: {e}")
            return {"error": str(e)}

    def _extract_resource_type(self, analysis: str) -> str:
        """Extract resource type from the analysis text"""
        analysis = analysis.lower()
        resource_types = {
            "stone": ["stone", "rock", "boulder"],
            "wood": ["wood", "tree", "log"],
            "ore": ["ore", "mineral", "metal"],
            "fish": ["fish", "fishing", "aquatic"],
            "flower": ["flower", "plant", "bloom"],
            "hide": ["hide", "leather", "skin"]
        }

        for resource_type, keywords in resource_types.items():
            if any(keyword in analysis for keyword in keywords):
                return resource_type

        return "unknown"

    def get_training_suggestions(self, training_metrics: Dict[str, Any]) -> str:
        """Get AI suggestions for improving model training based on metrics"""
        if not self.client:
            return "Error: OpenAI client not initialized"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI expert helping to improve resource detection model training for a game bot.
                        Focus on practical suggestions for:
                        1. Data quality and quantity
                        2. Class balance
                        3. Image size and preprocessing
                        4. Training parameters
                        Be concise and specific."""
                    },
                    {
                        "role": "user",
                        "content": f"""Based on these training metrics:
                        - Number of images: {training_metrics.get('num_images', 0)}
                        - Classes: {', '.join(str(c) for c in training_metrics.get('classes', []))}
                        - Sample image sizes: {training_metrics.get('image_sizes', [])}

                        What suggestions do you have for improving the model's performance?"""
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting training suggestions: {e}")
            return f"Error getting suggestions: {e}"

    def load_model(self, path: str):
        """Load the PyTorch model with proper error handling"""
        try:
            if os.path.exists(path):
                # First try loading with weights_only=True (new default)
                try:
                    self.model = torch.load(path, map_location='cpu', weights_only=True)
                except Exception as e:
                    # If that fails, try with weights_only=False for backward compatibility
                    print(f"Warning: Loading with weights_only=True failed, attempting legacy load: {e}")
                    self.model = torch.load(path, map_location='cpu', weights_only=False)
                print("Model loaded successfully")
            else:
                print(f"Model file not found at {path}")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None