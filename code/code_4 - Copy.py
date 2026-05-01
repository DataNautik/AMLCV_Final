"""
Conditional DDPM/DDIM for Speed-Painting Refinement (Optimized)
=================================================================
Optimizations included:
1. Precomputed dataset (No CPU bottleneck from OpenCV during training).
2. Concurrent image downloading (Multi-threaded).
3. DDIM Scheduler for 4x faster validation sampling.
4. Modern PyTorch AMP syntax.
5. Safe dataset loading (No infinite recursion).
"""

import os
import time
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────


BLUR_KERNEL_SIZE = (5, 5)  # Reduced from (7, 7) to keep fine details
CANNY_LOW        = 30      # Lowered base thresholds
CANNY_HIGH       = 80  
IMAGE_SIZE       = 256


# ─────────────────────────────────────────────────────────────────────────────
# 1. Faster Data Fetching (Multi-threaded)
# ─────────────────────────────────────────────────────────────────────────────
def download_single_image(img_url: str, img_path: str) -> bool:
    try:
        req_img = urllib.request.Request(
            img_url, headers={"User-Agent": "owl-sketch-project/2.0"}
        )
        with urllib.request.urlopen(req_img, timeout=10) as img_r:
            with open(img_path, "wb") as f:
                f.write(img_r.read())
        return True
    except Exception:
        return False

def fetch_images_if_missing(image_dir: str, num_images: int) -> None:
    os.makedirs(image_dir, exist_ok=True)
    existing = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(existing) >= num_images:
        print(f"Dataset already contains {len(existing)} images. Skipping download.")
        return

    print(f"Fetching images to reach {num_images} total...")
    count = len(existing)
    page = 1
    urls_to_download = []

    # Gather URLs
    while len(urls_to_download) < (num_images - count):
        params = urllib.parse.urlencode({
            "taxon_id": 19350, "quality_grade": "research",
            "photos": "true", "per_page": 100, "page": page,
            "order": "desc", "order_by": "created_at",
        })
        url = f"https://api.inaturalist.org/v1/observations?{params}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "owl-sketch-project/2.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())

            results = data.get("results", [])
            if not results: break

            for obs in results:
                photos = obs.get("photos", [])
                if photos:
                    img_url = photos[0]["url"].replace("square", "medium")
                    urls_to_download.append(img_url)
                    if len(urls_to_download) >= (num_images - count): break
            page += 1
        except Exception as e:
            print(f"API Error: {e}")
            break

    # Download concurrently
    print(f"Downloading {len(urls_to_download)} images concurrently...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, url in enumerate(urls_to_download):
            img_path = os.path.join(image_dir, f"owl_{count + i:04d}.jpg")
            futures.append(executor.submit(download_single_image, url, img_path))
        
        successes = sum(1 for f in as_completed(futures) if f.result())
    print(f"\nDone. Successfully downloaded {successes} new images.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Precompute Dataset (Avoid OpenCV bottleneck during training)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# REPLACE YOUR EXISTING _process_image AND precompute_dataset WITH THIS
# ─────────────────────────────────────────────────────────────────────────────
def _process_image(pil_img: Image.Image, low: float, high: float) -> Image.Image:
    img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("L")
    gray = np.array(img)
    
    # NEW: Use Bilateral Filter instead of Gaussian Blur. 
    # This magically erases background texture while keeping the main edges sharp.
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. Canny Edge Detection
    edges = cv2.Canny(blur, threshold1=int(low), threshold2=int(high))
    
    # 3. Tiny morphology to connect broken lines
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    
    # 4. Invert to black lines on white background
    clean = cv2.bitwise_not(clean)
    
    return Image.fromarray(clean)

def precompute_dataset(raw_dir: str, proc_dir: str):
    rough_dir = os.path.join(proc_dir, "rough")
    refined_dir = os.path.join(proc_dir, "refined")
    os.makedirs(rough_dir, exist_ok=True)
    os.makedirs(refined_dir, exist_ok=True)

    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    processed_files = os.listdir(rough_dir)
    
    if len(processed_files) == len(raw_files):
        return # Already precomputed

    print("Precomputing OpenCV edges with Bilateral Filtering...")
    for f_name in raw_files:
        try:
            path = os.path.join(raw_dir, f_name)
            img = Image.open(path)
            
            # Stage 3 – rough (condition y)
            rough = _process_image(img, CANNY_LOW * 3.0, CANNY_HIGH * 3.0)
            rough.save(os.path.join(rough_dir, f_name))
            
            # Stage 2 – refined (target x_0) - Raised multiplier to avoid noise
            refined = _process_image(img, CANNY_LOW * 1.5, CANNY_HIGH * 1.5)
            refined.save(os.path.join(refined_dir, f_name))
        except Exception as e:
            print(f"Skipping corrupted file {f_name}: {e}")



class OwlSketchDataset(Dataset):
    """ Loads PRECOMPUTED images. No OpenCV operations here. Safe and fast. """
    def __init__(self, proc_dir: str, max_samples: int | None = None):
        self.rough_dir = os.path.join(proc_dir, "rough")
        self.refined_dir = os.path.join(proc_dir, "refined")
        names = [f for f in os.listdir(self.rough_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_names = names[:max_samples] if max_samples else names
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        # Guaranteed to work since precompute step filters bad files. No recursion needed.
        name = self.image_names[idx]
        rough_img = Image.open(os.path.join(self.rough_dir, name))
        refined_img = Image.open(os.path.join(self.refined_dir, name))
        
        y = self.to_tensor(rough_img)    # (1, H, W) [0, 1]
        x_0 = self.to_tensor(refined_img) # (1, H, W) [0, 1]
        return y, x_0

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model & Schedulers
# ─────────────────────────────────────────────────────────────────────────────
def build_schedulers(num_train_timesteps: int = 1000):
    # DDPM for Training
    train_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear", beta_start=1e-4, beta_end=2e-2,
        prediction_type="epsilon", clip_sample=False,
    )
    # DDIM for Inference (Much faster validation)
    infer_scheduler = DDIMScheduler.from_config(train_scheduler.config)
    return train_scheduler, infer_scheduler

# ─────────────────────────────────────────────────────────────────────────────
# REPLACE YOUR EXISTING build_conditional_unet WITH THIS
# ─────────────────────────────────────────────────────────────────────────────
def build_conditional_unet() -> UNet2DModel:
    return UNet2DModel(
        sample_size=IMAGE_SIZE, 
        in_channels=2, 
        out_channels=1,
        layers_per_block=2, 
        # Increased channel dimensions to support Transformer logic
        block_out_channels=(64, 128, 256, 512), 
        
        # Injecting Transformer Self-Attention in the deeper layers
        down_block_types=(
            "DownBlock2D",      # 64 channels (Standard Convolution)
            "DownBlock2D",      # 128 channels (Standard Convolution)
            "AttnDownBlock2D",  # 256 channels (Transformer Self-Attention!)
            "DownBlock2D",      # 512 channels (Standard Convolution)
        ),
        up_block_types=(
            "UpBlock2D",        # 512 channels (Standard Convolution)
            "AttnUpBlock2D",    # 256 channels (Transformer Self-Attention!)
            "UpBlock2D",        # 128 channels (Standard Convolution)
            "UpBlock2D",        # 64 channels  (Standard Convolution)
        ),
    )
# ─────────────────────────────────────────────────────────────────────────────
# 4. Core Logic
# ─────────────────────────────────────────────────────────────────────────────
def compute_l_simple(model, scheduler, x_0, y, device):
    x_0_scaled = x_0 * 2.0 - 1.0      
    y_scaled   = y   * 2.0 - 1.0

    t = torch.randint(0, scheduler.config.num_train_timesteps, (x_0.shape[0],), device=device, dtype=torch.long)
    epsilon = torch.randn_like(x_0_scaled)
    x_t = scheduler.add_noise(x_0_scaled, epsilon, t)
    model_input = torch.cat([x_t, y_scaled], dim=1)
    
    noise_pred = model(model_input, timestep=t).sample
    return torch.nn.functional.mse_loss(noise_pred, epsilon)

@torch.no_grad()
def sample_speed_painting(model, scheduler, y, device, num_inference_steps: int = 50):
    """ Uses DDIM instead of DDPM to jump steps, resulting in massive speedups """
    model.eval()
    y_scaled = (y * 2.0 - 1.0).to(device)
    x_t = torch.randn(y.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
    
    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:
        model_input = torch.cat([x_t, y_scaled], dim=1)
        t_batch = t.expand(x_t.shape[0]).to(device)
        noise_pred = model(model_input, timestep=t_batch).sample
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    return (x_t * 0.5 + 0.5).clamp(0.0, 1.0) # Map back to [0, 1]

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_conditional_ddpm(
    run_name: str, raw_data_path: str, proc_data_path: str, max_samples: int | None,
    epochs: int, batch_size: int, learning_rate: float = 1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    
    # Setup Data
    precompute_dataset(raw_data_path, proc_data_path)
    dataset = OwlSketchDataset(proc_data_path, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    # Setup Model
    train_scheduler, infer_scheduler = build_schedulers()
    model = build_conditional_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Modern PyTorch AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    out_dir = f"output_images/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    fixed_y, fixed_x0 = next(iter(dataloader))
    fixed_y, fixed_x0 = fixed_y.to(device), fixed_x0.to(device)
    
    save_epochs = sorted({1, epochs // 4, epochs // 2, (3 * epochs) // 4, epochs} - {0})

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for y_batch, x0_batch in dataloader:
            y_batch, x0_batch = y_batch.to(device), x0_batch.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                loss = compute_l_simple(model, train_scheduler, x0_batch, y_batch, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}] | {time.time() - t0:.1f}s | L_simple: {avg_loss:.6f}")

        # Validation
        if epoch in save_epochs:
            # Use DDIM and 50 steps for fast evaluation
            samples = sample_speed_painting(model, infer_scheduler, fixed_y, device, num_inference_steps=50)
            
            # Map grid items to [-1, 1] purely for robust vutils saving with normalize=True
            grid = torch.cat([fixed_y, samples, fixed_x0], dim=3) * 2.0 - 1.0 
            vutils.save_image(
                grid, f"{out_dir}/epoch_{epoch:04d}.png",
                normalize=True, value_range=(-1, 1), nrow=batch_size
            )

    torch.save(model.state_dict(), f"{run_name}_ddpm.pth")
    return {"run": run_name, "final_loss": avg_loss}

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    RAW_PATH = "./owlsclean/"
    PROC_PATH = "./owlsclean_processed/"
    
    # 1. Fetch images if needed
    #fetch_images_if_missing(RAW_PATH, num_images=1000)

    # 2. Run the quick smoke test
    print("\n=== Running Initial Smoke Test ===")
    train_conditional_ddpm(
        run_name="trial_ddpm",
        raw_data_path=RAW_PATH,
        proc_data_path=PROC_PATH,
        max_samples=64,
        epochs=2,
        batch_size=2,
        learning_rate=1e-4,
    )

    # 3. Ablation experiments loop
    experiments = [
        {"name": "exp1_baseline", "lr": 1e-4, "batch": 2},
        {"name": "exp2_high_lr",  "lr": 5e-4, "batch": 2},
        {"name": "exp3_large_bs", "lr": 1e-4, "batch": 4},
    ]

    results = {}
    print("\n=== Starting Ablation Experiments ===")
    for exp in experiments:
        print(f"\n--- Running: {exp['name']} ---")
        history = train_conditional_ddpm(
            run_name=exp["name"],
            raw_data_path=RAW_PATH,
            proc_data_path=PROC_PATH,
            max_samples=None,  # Use full dataset
            epochs=20,         # Full 20 epochs
            batch_size=exp["batch"],
            learning_rate=exp["lr"],
        )
        results[exp["name"]] = history

    # 4. Print final results
    print("\n=== Final L_simple per experiment ===")
    for name, hist in results.items():
        print(f"  {name}: {hist['final_loss']:.6f}")

if __name__ == "__main__":
    main()