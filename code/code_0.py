import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from diffusers import UNet2DModel
import lpips 

BLUR_KERNEL_SIZE = (7, 7)
CANNY_LOW = 50
CANNY_HIGH = 100

class OwlSketchDataset(Dataset):
    def __init__(self, image_dir, max_samples=None, transform=None):
        self.image_dir = image_dir
        self.image_names = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Limit dataset size for testing
        if max_samples is not None:
            self.image_names = self.image_names[:max_samples]
            
        # Convert numpy array to float tensor [0.0, 1.0]
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(), 
        ])

    def __len__(self):
        return len(self.image_names)

    def preprocess(self, image, low, high):
        # Resize and convert to grayscale
        resize = image.resize((256, 256))
        gray = resize.convert("L")
        np_gray = np.array(gray)
        
        # Apply blur and Canny edge detection
        blurred = cv2.GaussianBlur(np_gray, BLUR_KERNEL_SIZE, sigmaX=0)
        edges = cv2.Canny(blurred, threshold1=low, threshold2=high)
        
        # Apply morphological filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        noiseless = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
        gapclose = cv2.morphologyEx(noiseless, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Invert colors
        sketch = cv2.bitwise_not(gapclose)
        return sketch

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        try:
            opened = Image.open(img_path)
        except Exception as e:
            print(f"Warning: Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Generate stage 3 (input) and stage 2 (target) sketches
        rough_sketch = self.preprocess(opened, CANNY_LOW * 3.5, CANNY_HIGH * 3.5)
        refined_sketch = self.preprocess(opened, CANNY_LOW * 2, CANNY_HIGH * 2)
        
        x_input = self.transform(rough_sketch)
        y_target = self.transform(refined_sketch)
        
        return x_input, y_target


def train_experiment(run_name, data_path, max_samples, epochs, batch_size, learning_rate):
    # Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize DataLoader
    dataset = OwlSketchDataset(image_dir=data_path, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize UNet model for 1-channel grayscale input/output
    model = UNet2DModel(
        sample_size=256,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    # Define optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    # Suppress LPIPS verbose output
    import logging
    logging.getLogger("lpips").setLevel(logging.ERROR)

    experiment_history = []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs).sample
            
            # Backward pass and optimization
            loss = mse_loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_mse_loss = epoch_loss / len(dataloader)
        
        # Calculate perceptual loss (LPIPS) on the final batch
        model.eval()
        with torch.no_grad():
            # Format tensors to [-1, 1] and 3 channels for LPIPS
            inputs_eval = inputs.repeat(1, 3, 1, 1) * 2.0 - 1.0
            targets_eval = targets.repeat(1, 3, 1, 1) * 2.0 - 1.0
            outputs_eval = outputs.repeat(1, 3, 1, 1) * 2.0 - 1.0
            perceptual_loss = loss_fn_vgg(outputs_eval, targets_eval).mean().item()
            
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] | Time: {epoch_time:.1f}s | MSE: {avg_mse_loss:.4f} | LPIPS: {perceptual_loss:.4f}")
        
        experiment_history.append({
            'epoch': epoch + 1,
            'mse': avg_mse_loss,
            'lpips': perceptual_loss
        })

    # Export trained model weights
    save_path = f"{run_name}_model.pth"
    torch.save(model.state_dict(), save_path)
    
    return experiment_history


def main():
    PATH = "/Users/christiansandoval/Documents/AdvancedComputerVision/adlcv-project/owls1000/"
    
    # Trial execution for script validation
    train_experiment(
        run_name="trial_run",
        data_path=PATH,
        max_samples=32, 
        epochs=2,
        batch_size=8,
        learning_rate=1e-4
    )

    # Hyperparameter comparison executions
    experiments = [
        {"name": "exp1_baseline", "lr": 1e-4, "batch": 16},
        {"name": "exp2_high_lr",  "lr": 5e-4, "batch": 16},
        {"name": "exp3_large_bs", "lr": 1e-4, "batch": 32},
    ]
    
    results = {}
    
    for exp in experiments:
        history = train_experiment(
            run_name=exp["name"],
            data_path=PATH,
            max_samples=None, 
            epochs=20,        
            batch_size=exp["batch"],
            learning_rate=exp["lr"]
        )
        results[exp["name"]] = history

if __name__ == "__main__":
    main()