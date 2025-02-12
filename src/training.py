""" MIT License

Copyright (c) 2025 HackConEdu - Eduardo Jara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import pandas as pd

# Dispositivo de cómputo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset personalizado
class ActionDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    # Función para obtener la longitud del dataset
    def __len__(self):
        return len(self.data)

    # Función para obtener un ítem del dataset
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = row['screenshot_path']
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        acceleration = row['aceleration']
        direction = row['direction']

        w = 1.0 if acceleration == 'w' else 0.0
        s = 1.0 if acceleration == 's' else 0.0
        a = 1.0 if direction == 'a' else 0.0
        d = 1.0 if direction == 'd' else 0.0

        target = torch.tensor([w, s, a, d], dtype=torch.float32)

        return img, target

# Modelo de red neuronal
class ResNet18DrivingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Cargar modelo base con pesos pre-entrenados
        base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        # Congelar varias capas para transferencia de aprendizaje con el objetivo de mejorar la generalización
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        for param in self.features[:8].parameters():  # Aumentamos capas congeladas
            param.requires_grad = False

        # Definir capa de pooling y tamaño de características
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_dim = base_model.fc.in_features

        # Definir dos cabezas de salida
        self.acceleration_head = nn.Sequential(
            nn.Linear(feature_dim, 128),  # 128 neuronas
            nn.ReLU(),
            nn.Dropout(0.5),  # Aumentado a 0.5
            nn.Linear(128, 2)
        )

        self.direction_head = nn.Sequential(
            nn.Linear(feature_dim, 128),  # 128 neuronas
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    # Propagación hacia adelante
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.cat([self.acceleration_head(x), self.direction_head(x)], dim=1)

# Función de pérdida para varias tareas
class MultiTaskLoss(nn.Module):
    def __init__(self, accel_weight=1.2, dir_weight=1.0):
        super().__init__()
        self.accel_weight = accel_weight
        self.dir_weight = dir_weight

    def forward(self, outputs, targets):
        accel_loss = func.binary_cross_entropy_with_logits(outputs[:, :2], targets[:, :2])
        dir_loss = func.binary_cross_entropy_with_logits(outputs[:, 2:], targets[:, 2:])
        return self.accel_weight * accel_loss + self.dir_weight * dir_loss

# Transformaciones de imagen
def get_transforms(is_train=True):
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]

    if is_train:
        transform_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomRotation(20),  # Mayor rotación
            transforms.RandomAffine(degrees=15, translate=(0.25, 0.25)),
        ]

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # Retornar composición de transformaciones
    return transforms.Compose(transform_list)

# Entrenamiento con validación cruzada
def train_kfold(csv_path, num_folds=5, num_epochs=50):
    full_dataset = ActionDataset(csv_path, transform=get_transforms(is_train=True))
    kfold = KFold(n_splits=num_folds, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n=== Fold {fold + 1}/{num_folds} ===")

        train_loader = DataLoader(Subset(full_dataset, train_idx),
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

        val_loader = DataLoader(Subset(full_dataset, val_idx),
                                batch_size=128,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True)

        model = ResNet18DrivingModel().to(device)

        optimizer = optim.AdamW(model.parameters(),
                                lr=1e-4,
                                weight_decay=1e-3)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        criterion = MultiTaskLoss()

        best_val_loss = float('inf')

        patience = 10
        patience_counter = 0

        # Entrenamiento por épocas
        for epoch in range(num_epochs):
            model.train()

            train_loss = 0.0

            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validación
            model.eval()

            val_loss = 0.0

            # Deshabilitar cálculo de gradientes
            with torch.no_grad():
                for images, targets in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    val_loss += criterion(model(images), targets).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch + 1:02d}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"best_fold{fold + 1}.pth")
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break


if __name__ == "__main__":
    # Entrenar modelo con validación cruzada, el dataset se crea al ejecutar el script de captura de datos
    train_kfold("dataset/actions.csv", num_folds=5, num_epochs=50)