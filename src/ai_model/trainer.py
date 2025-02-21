import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List, Dict
import os

class ResourceDataset(Dataset):
    def __init__(self, image_paths: List[str], annotations: List[Dict], transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = self.annotations[idx]['boxes']
        labels = self.annotations[idx]['labels']
        
        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        
        return image, target

class ModelTrainer:
    def __init__(self, num_classes: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(num_classes)
        self.model.to(self.device)

    def _create_model(self, num_classes: int) -> torch.nn.Module:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes)
        return model

    def train(self, train_dataset: ResourceDataset, num_epochs: int = 10, 
              batch_size: int = 4, progress_callback=None):
        """Train the model with progress updates"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x))
        )

        optimizer = optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9)

        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for images, targets in train_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()

            avg_loss = epoch_loss/len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

            if progress_callback:
                progress_callback(epoch, num_epochs, avg_loss)

    def save_model(self, path: str):
        """Save the trained model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")