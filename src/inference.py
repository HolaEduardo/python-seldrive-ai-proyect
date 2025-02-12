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

import cv2
import numpy as np
import pyautogui
import torch
import torch.nn as nn
import torchvision
from mss import mss
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo de red neuronal
class ResNet18DrivingModel(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        self.features = nn.Sequential(*list(base_model.children())[:-2])
        for param in self.features[:8].parameters():
            param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_dim = base_model.fc.in_features

        self.acceleration_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

        self.direction_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.cat([self.acceleration_head(x), self.direction_head(x)], dim=1)


# Cargar el modelo con configuración segura
model_path = "best_fold1.pth"
model = ResNet18DrivingModel().to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)  # Corrección de seguridad
)
model.eval()
LINE_Y = 300

# Transformaciones IDÉNTICAS a las de entrenamiento
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),  # Mismo que en entrenamiento
    transforms.CenterCrop(224),  # Mismo que en entrenamiento
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Configuración de captura de pantalla
screen_width, screen_height = pyautogui.size()
monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}
sct = mss()

current_keys = set()

# Función para procesar un frame
def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    img_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        w_prob = torch.sigmoid(outputs[0, 0]).item()  # Probabilidad para 'w'
        s_prob = torch.sigmoid(outputs[0, 1]).item()  # Probabilidad para 's'
        a_prob = torch.sigmoid(outputs[0, 2]).item()  # Probabilidad para 'a'
        d_prob = torch.sigmoid(outputs[0, 3]).item()  # Probabilidad para 'd'

    acceleration = 'w' if w_prob > 0.72 and (a_prob < 0.055 > d_prob) else 's' if s_prob > 0.020 else 'none'
    direction = 'a' if a_prob > 0.072 and w_prob < 0.72 else 'd' if d_prob > 0.072 and w_prob < 0.72 else 'none'

    print(f"Aceleración: {acceleration}, Dirección: {direction}")
    print(f"Probabilidades - W: {w_prob:.4f}, S: {s_prob:.4f}, A: {a_prob:.4f}, D: {d_prob:.4f}")

    active_keys = []
    if acceleration != 'none':
        active_keys.append(acceleration)
    if direction != 'none':
        active_keys.append(direction)

    return active_keys

def update_keys(active_keys):
    global current_keys
    for key in current_keys - set(active_keys):
        pyautogui.keyUp(key)

    for key in set(active_keys) - current_keys:
        pyautogui.keyDown(key)

    current_keys = set(active_keys)

def draw_static_lanes(frame):
    cv2.line(frame, (-100, 1000), (910, 0), (0, 255, 0), 8)
    cv2.line(frame, (2000, 1000), (910, -100), (0, 255, 0), 8)

    h, w = frame.shape[:2]
    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (255, 0, 0), 2)
    return frame

# Función principal
def main():
    try:
        print("Control automático activado (Presiona Q para salir)...")
        while True:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)

            frame = draw_static_lanes(frame)

            active_keys = process_frame(frame)
            update_keys(active_keys)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupción del usuario")
    finally:
        for key in current_keys:
            pyautogui.keyUp(key)
        cv2.destroyAllWindows()
        print("Control liberado")

if __name__ == "__main__":
    main()