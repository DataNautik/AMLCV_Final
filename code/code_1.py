import os
import time
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers import UNet2DModel
import lpips 

BLUR_KERNEL_SIZE = (7, 7)
CANNY_LOW = 50
CANNY_HIGH = 100

def fetch_images_if_missing(image_dir, num_images):
    os.makedirs(image_dir, exist_ok=True)
    existing_images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(existing_images) >= num_images:
        return
        
    count = len(existing_images)
    while count < num_images:
        img_path = os.path.join(image_dir, f"test_dataset_{count:04d}.jpg")
        try:
            urllib.request.urlretrieve("https://picsum.photos/256", img_path)
            count += 1
        except Exception:
            time.sleep(1) 

class OwlSketchDataset(Dataset):
    def __init__(self, image_dir, max_samples=None, transform=None):
        self.image_dir = image_dir
        self.image_names = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if max_samples is not None:
            self.image_names = self.image_names[:max_samples]
            
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(), 
        ])

    def __len__(self):
        return len(self.image_names)

    def preprocess(self, image, low, high):
        resize = image.resize((256, 256))
        gray = resize.convert("L")
        np_gray = np.array(gray)
        
        blurred = cv2.GaussianBlur(np_gray, BLUR_KERNEL_SIZE, sigmaX=0)
        edges = cv2.Canny(blurred, threshold1=low, threshold2=high)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        noiseless = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
        gapclose = cv2.morphologyEx(noiseless, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        sketch = cv2.bitwise_not(gapclose)
        return sketch

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        try:
            opened = Image.open(img_path)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
        
        rough_sketch = self.preprocess(opened, CANNY_LOW * 3.5, CANNY_HIGH * 3.5)
        refined_sketch = self.preprocess(opened, CANNY_LOW * 2, CANNY_HIGH * 2)
        
        x_input = self.transform(rough_sketch)
        y_target = self.transform(refined_sketch)
        
        return x_input, y_target


def train_experiment(run_name, data_path, max_samples, epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = OwlSketchDataset(image_dir=data_path, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    os.makedirs(f"output_images/{run_name}", exist_ok=True)
    fixed_inputs, fixed_targets = next(iter(dataloader))
    fixed_inputs, fixed_targets = fixed_inputs.to(device), fixed_targets.to(device)

    model = UNet2DModel(
        sample_size=256,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 256), 
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    import logging
    logging.getLogger("lpips").setLevel(logging.ERROR)

    experiment_history = []
    save_epochs = [1, epochs // 4, epochs // 2, (epochs * 3) // 4, epochs]
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs, timestep=0).sample
                loss = mse_loss_fn(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        avg_mse_loss = epoch_loss / len(dataloader)
        
        model.eval()
        with torch.no_grad():
            inputs_eval = inputs.repeat(1, 3, 1, 1) * 2.0 - 1.0
            targets_eval = targets.repeat(1, 3, 1, 1) * 2.0 - 1.0
            outputs_eval = outputs.repeat(1, 3, 1, 1) * 2.0 - 1.0
            perceptual_loss = loss_fn_vgg(outputs_eval, targets_eval).mean().item()
            
            if (epoch + 1) in save_epochs:
                fixed_predictions = model(fixed_inputs, timestep=0).sample
                comparison = torch.cat([fixed_inputs, fixed_predictions, fixed_targets], dim=3)
                vutils.save_image(
                    comparison, 
                    f"output_images/{run_name}/epoch_{epoch+1}.png", 
                    normalize=True, 
                    nrow=batch_size
                )

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | Time: {epoch_time:.1f}s | MSE: {avg_mse_loss:.4f} | LPIPS: {perceptual_loss:.4f}")
        
        experiment_history.append({
            'epoch': epoch + 1,
            'mse': avg_mse_loss,
            'lpips': perceptual_loss
        })

    save_path = f"{run_name}_model.pth"
    torch.save(model.state_dict(), save_path)
    
    return experiment_history


def main():
    PATH = "./owls1000/"
    
    fetch_images_if_missing(image_dir=PATH, num_images=1000)
    
    train_experiment(
        run_name="trial_run",
        data_path=PATH,
        max_samples=32, 
        epochs=2,
        batch_size=2,  
        learning_rate=1e-4
    )

    experiments = [
        {"name": "exp1_baseline", "lr": 1e-4, "batch": 2},
        {"name": "exp2_high_lr",  "lr": 5e-4, "batch": 2},
        {"name": "exp3_large_bs", "lr": 1e-4, "batch": 4}, 
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