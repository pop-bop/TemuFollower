import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from jepa import LineJEPA

class SyntheticLineDataset(Dataset):
    """
    Self-generates examples of lines on noisy backgrounds to train the JEPA model.
    Ground truth labels are the exact Center of Gravity (position) and Angle.
    """
    def __init__(self, num_samples=5000, img_size=(160, 120)):
        self.num_samples = num_samples
        self.w, self.h = img_size
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # 1. Generate noisy light background (representing floor)
        bg_color = np.random.randint(180, 255)
        img = np.full((self.h, self.w, 3), bg_color, dtype=np.uint8)
        noise = np.random.randint(-30, 30, (self.h, self.w, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # 2. Randomly define line parameters (Ground Truth)
        # Position: Center of Gravity X (-1.0 to 1.0)
        pos = np.random.uniform(-0.8, 0.8) 
        # Angle: Radians (-60 to +60 degrees)
        angle = np.random.uniform(-np.pi/3, np.pi/3)
        
        cx = int((pos + 1.0) / 2.0 * self.w)
        cy = int(self.h / 2) # Center of Gravity Y is center of frame
        
        line_width = np.random.randint(15, 35)
        line_length = self.h * 2
        
        # Line color (dark)
        color = np.random.randint(0, 60)
        
        # Calculate endpoints for drawing
        dx = int(line_length * np.sin(angle))
        dy = int(line_length * np.cos(angle))
        
        pt1 = (cx - dx, cy + dy)
        pt2 = (cx + dx, cy - dy)
        
        cv2.line(img, pt1, pt2, (color, color, color), line_width)
        
        # 3. Format for PyTorch
        img_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = img_tensor.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1)) # (C, H, W)
        
        state = np.array([pos, angle], dtype=np.float32)
        
        return torch.tensor(img_tensor), torch.tensor(state)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LineJEPA().to(device)
    
    print(f"Training on {device}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Generate 10,000 synthetic examples on the fly
    dataset = SyntheticLineDataset(num_samples=10000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    epochs = 8
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            _, preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {avg_loss:.4f}")
        
    weights_path = os.path.join(os.path.dirname(__file__), "jepa_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Training complete! Weights saved to {weights_path}")

if __name__ == "__main__":
    train_model()
