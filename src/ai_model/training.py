import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Callable
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ResourceDataset(Dataset):
    def __init__(self, image_paths: List[str], annotations: List[Dict], transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        # Get annotation
        label = self.annotations[idx]['label']
        return image, label

class ResourceDetectorModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Using a pre-trained ResNet18 backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 
                                     'resnet18', pretrained=True)
        # Replace the final layer for our specific number of classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    def __init__(self, num_classes: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResourceDetectorModel(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, 
             dataset: ResourceDataset,
             num_epochs: int = 10,
             batch_size: int = 4,
             progress_callback: Callable = None):
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if progress_callback:
                    progress_callback(epoch, num_epochs, running_loss / (i + 1))

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
