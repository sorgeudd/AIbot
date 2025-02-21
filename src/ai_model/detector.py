import torch
import torchvision
import cv2
import numpy as np
from typing import List, Tuple
import os

class ResourceDetector:
    def __init__(self, model_path: str = "models/resource_detector.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Define resource classes
        self.classes = ["background", "stone", "wood", "ore", "fish", "flower", "hide"]
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _create_new_model(self) -> torch.nn.Module:
        """Create a new model instance with our custom number of classes"""
        # Create a new model with pretrained weights
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

        # Modify the box predictor for our number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, len(self.classes)
        )

        return model

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load or create a new model with proper error handling"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            if os.path.exists(model_path):
                try:
                    # First try loading with weights_only=True (new default)
                    print(f"Attempting to load model from {model_path} with weights_only=True")
                    model = self._create_new_model()  # Create the model structure first
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                except Exception as e:
                    print(f"Warning: Failed to load model with weights_only=True: {e}")
                    try:
                        # Try loading the entire model (legacy mode)
                        print("Attempting legacy model load...")
                        model = torch.load(model_path, map_location=self.device, weights_only=False)
                    except Exception as e2:
                        print(f"Warning: Legacy load also failed: {e2}")
                        print("Creating new model...")
                        model = self._create_new_model()
                        # Save the new model in the correct format
                        torch.save(model.state_dict(), model_path)
            else:
                print(f"No model found at {model_path}, creating new one")
                model = self._create_new_model()
                # Save the new model in the correct format
                torch.save(model.state_dict(), model_path)

            print("Model initialized successfully")
            return model

        except Exception as e:
            print(f"Error in model initialization: {e}")
            print("Falling back to new model instance")
            return self._create_new_model()

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0)
        return image.to(self.device)

    def detect_resources(self, image: np.ndarray) -> List[Tuple[str, float, List[int]]]:
        """Detect resources in the image"""
        try:
            processed_image = self.preprocess_image(image)

            with torch.no_grad():
                predictions = self.model(processed_image)

            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            threshold = 0.7
            detections = []

            for box, score, label in zip(boxes, scores, labels):
                if score > threshold:
                    class_name = self.classes[label]  # Get class name from index
                    detections.append((class_name, score, box.tolist()))

            return detections
        except Exception as e:
            print(f"Error during resource detection: {e}")
            return []

    def visualize_detections(self, image: np.ndarray, detections: List[Tuple[str, float, List[int]]]) -> np.ndarray:
        """Visualize detected resources on the image"""
        output = image.copy()
        for class_name, confidence, box in detections:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # Different colors for different resource types
            color = {
                'stone': (120, 120, 120),  # Gray
                'wood': (101, 67, 33),     # Brown
                'ore': (255, 215, 0),      # Gold
                'fish': (0, 255, 255),     # Cyan
                'flower': (147, 20, 255),  # Purple
                'hide': (210, 180, 140)    # Tan
            }.get(class_name, (0, 255, 0))  # Default to green

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f'{class_name} {confidence:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return output