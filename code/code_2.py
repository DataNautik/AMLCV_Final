"""
Conditional DDPM for Speed-Painting Refinement
================================================
Goal  : Given a rough sketch (Stage 3) as condition y,
        learn to generate the refined sketch (Stage 2) as x_0.

Key changes vs. original code
------------------------------
1.  DDPMScheduler  – linear beta schedule, T=1000 steps.
2.  Conditional U-Net – 2-channel input: [x_t ; y]  (noisy target + condition).
3.  L_simple        – MSE between predicted noise ε_θ and true noise ε.
4.  sample_speed_painting – full reverse-diffusion loop conditioned on y.
5.  Memory          – channel dims (32, 64, 128, 256), batch 2-4 per GPU,
                      DataParallel + AMP kept throughout.
"""

import os
import time
import urllib.request
import urllib.parse

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import json

from diffusers import UNet2DModel, DDPMScheduler

# ─────────────────────────────────────────────────────────────────────────────
# Global preprocessing constants
# ─────────────────────────────────────────────────────────────────────────────
BLUR_KERNEL_SIZE = (7, 7)
CANNY_LOW        = 50
CANNY_HIGH       = 100
IMAGE_SIZE       = 256

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Noise scheduler  (linear β schedule, T = 1000)
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(num_train_timesteps: int = 1000) -> DDPMScheduler:
    """
    Returns a DDPMScheduler with a linear beta schedule.

    β_1 = 1e-4  →  β_T = 0.02  (standard Ho et al. 2020 values).
    'epsilon' prediction_type means the U-Net predicts the noise ε.
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",          # linear β_1 … β_T
        beta_start=1e-4,
        beta_end=2e-2,
        prediction_type="epsilon",       # model predicts ε, not x_0
        clip_sample=False,               # keep raw predictions during training
    )
    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Conditional U-Net  (2-channel input: [x_t ; y])
# ─────────────────────────────────────────────────────────────────────────────
def build_conditional_unet() -> UNet2DModel:
    """
    Standard UNet2DModel where:
      in_channels  = 2   (noisy refined sketch x_t  ||  rough sketch y)
      out_channels = 1   (predicted noise  ε_θ  for the refined sketch only)

    The concatenation makes the conditioning explicit and requires zero extra
    architectural complexity.  Channel dims kept slim for 6 GB VRAM.
    """
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=2,                        # x_t (1ch) + y (1ch)
        out_channels=1,                       # predicted ε
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (unchanged logic, kept here for self-containment)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_images_if_missing(image_dir: str, num_images: int) -> None:
    os.makedirs(image_dir, exist_ok=True)
    existing = [f for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    count = len(existing)
    page  = 1

    while count < num_images:
        # Fix 1: parameters must be passed as a proper query string
        # Fix 2: taxon_id for owls is 19350 (Strigiformes on iNaturalist v1 API)
        # Fix 3: add iconic_taxa=Aves to help filter correctly
        params = urllib.parse.urlencode({
            "taxon_id":     19350,
            "quality_grade": "research",
            "photos":        "true",
            "per_page":      50,
            "page":          page,
            "order":         "desc",
            "order_by":      "created_at",
        })
        url = f"https://api.inaturalist.org/v1/observations?{params}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "owl-sketch-project/1.0"}  # Fix 4: API requires a User-Agent
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())

            results = data.get("results", [])
            if not results:
                print("No more results from API.")
                break

            for obs in results:
                if count >= num_images:
                    break
                photos = obs.get("photos", [])
                if not photos:
                    continue
                # Replace 'square' (75px) with 'medium' (440px)
                img_url  = photos[0]["url"].replace("square", "medium")
                img_path = os.path.join(image_dir, f"owl_{count:04d}.jpg")
                try:
                    req_img = urllib.request.Request(
                        img_url,
                        headers={"User-Agent": "owl-sketch-project/1.0"}
                    )
                    with urllib.request.urlopen(req_img, timeout=15) as img_r:
                        with open(img_path, "wb") as f:
                            f.write(img_r.read())
                    count += 1
                    print(f"Downloaded {count}/{num_images}", end="\r")
                except Exception as e:
                    print(f"  Skipping image: {e}")
                    continue

            page += 1
            time.sleep(1)

        except Exception as e:
            print(f"API error on page {page}: {e}")
            time.sleep(3)
            break  # stop retrying same page infinitely

    print(f"\nDone. {count} images in {image_dir}")



class OwlSketchDataset(Dataset):
    """
    Returns (rough_sketch y,  refined_sketch x_0) pairs,
    both as single-channel float tensors in [0, 1].

    rough_sketch  = higher Canny thresholds  → fewer / bolder edges  (Stage 3)
    refined_sketch = lower  Canny thresholds  → denser  edges         (Stage 2)
    """

    _to_tensor = transforms.ToTensor()   # uint8 HxW  →  float 1xHxW in [0,1]

    def __init__(self, image_dir: str, max_samples: int | None = None):
        self.image_dir = image_dir
        names = [f for f in os.listdir(image_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_names = names[:max_samples] if max_samples else names

    def __len__(self) -> int:
        return len(self.image_names)

    def _preprocess(self, pil_img: Image.Image, low: float, high: float) -> np.ndarray:
        img   = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("L")
        gray  = np.array(img)
        blur  = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, sigmaX=0)
        edges = cv2.Canny(blur, threshold1=low, threshold2=high)
        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        clean = cv2.morphologyEx(edges, cv2.MORPH_OPEN,  k, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=3)
        return cv2.bitwise_not(clean)          # white background, black lines

    def __getitem__(self, idx: int):
        path = os.path.join(self.image_dir, self.image_names[idx])
        try:
            img = Image.open(path)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        # Stage 3 – rough  (condition  y)
        rough   = self._preprocess(img, CANNY_LOW * 3.5, CANNY_HIGH * 3.5)
        # Stage 2 – refined (target x_0)
        refined = self._preprocess(img, CANNY_LOW * 2.0, CANNY_HIGH * 2.0)

        y   = self._to_tensor(rough)    # shape (1, H, W)
        x_0 = self._to_tensor(refined)  # shape (1, H, W)
        return y, x_0


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training objective  – L_simple
# ─────────────────────────────────────────────────────────────────────────────
def compute_l_simple(
    model:     nn.Module,
    scheduler: DDPMScheduler,
    x_0:       torch.Tensor,   # (B, 1, H, W) refined sketch, in [0, 1]
    y:         torch.Tensor,   # (B, 1, H, W) rough sketch condition, in [0, 1]
    device:    torch.device,
) -> torch.Tensor:
    """
    Samples a random timestep t and noise ε, forms x_t, then computes:

        L_simple = || ε  −  ε_θ(x_t, t, y) ||²

    The condition y is concatenated channel-wise to x_t before the forward pass.
    """
    batch_size = x_0.shape[0]

    # ── Rescale inputs to [-1, 1] as expected by the scheduler ──────────────
    x_0_scaled = x_0 * 2.0 - 1.0      # [0,1] → [-1,1]
    y_scaled   = y   * 2.0 - 1.0

    # ── Sample random timesteps ──────────────────────────────────────────────
    t = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (batch_size,), device=device, dtype=torch.long
    )

    # ── Sample Gaussian noise ε ──────────────────────────────────────────────
    epsilon = torch.randn_like(x_0_scaled)

    # ── Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε ────────────────
    x_t = scheduler.add_noise(x_0_scaled, epsilon, t)   # (B, 1, H, W)

    # ── Build 2-channel conditional input  [x_t ; y] ─────────────────────────
    model_input = torch.cat([x_t, y_scaled], dim=1)     # (B, 2, H, W)

    # ── Predict noise ε_θ ────────────────────────────────────────────────────
    noise_pred = model(model_input, timestep=t).sample  # (B, 1, H, W)

    # ── L_simple = E[ || ε - ε_θ ||² ] ──────────────────────────────────────
    return torch.nn.functional.mse_loss(noise_pred, epsilon)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sampling / inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_speed_painting(
    model:     nn.Module,
    scheduler: DDPMScheduler,
    y:         torch.Tensor,    # (B, 1, H, W) rough sketch condition, [0, 1]
    device:    torch.device,
    num_inference_steps: int = 1000,
) -> torch.Tensor:
    """
    Full DDPM reverse-diffusion loop conditioned on the rough sketch y.

    Algorithm
    ---------
    1. x_T ~ N(0, I)
    2. For t = T-1, …, 0:
         a. build model input = [x_t ; y]
         b. predict ε_θ(x_t, t, y)
         c. compute x_{t-1}  via  scheduler.step()
    3. Return x_0 rescaled to [0, 1]

    Parameters
    ----------
    num_inference_steps : use T for full quality; fewer for speed.
    """
    model.eval()

    # ── Rescale condition to [-1, 1] ─────────────────────────────────────────
    y_scaled = (y * 2.0 - 1.0).to(device)

    # ── Start from pure Gaussian noise ──────────────────────────────────────
    x_t = torch.randn(y.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # ── Set timestep schedule ────────────────────────────────────────────────
    scheduler.set_timesteps(num_inference_steps)

    for t in scheduler.timesteps:                    # T-1, T-2, …, 0
        # Build 2-channel input
        model_input = torch.cat([x_t, y_scaled], dim=1)   # (B, 2, H, W)

        # Batched scalar timestep expected by UNet2DModel
        t_batch = t.expand(x_t.shape[0]).to(device)

        # Predict noise
        noise_pred = model(model_input, timestep=t_batch).sample

        # Reverse step:  p_θ(x_{t-1} | x_t) → sample x_{t-1}
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    # ── Rescale [-1, 1] → [0, 1] and clamp ──────────────────────────────────
    x_0 = (x_t * 0.5 + 0.5).clamp(0.0, 1.0)
    return x_0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_conditional_ddpm(
    run_name:    str,
    data_path:   str,
    max_samples: int | None,
    epochs:      int,
    batch_size:  int,            # per-GPU batch size
    learning_rate: float = 1e-4,
    num_train_timesteps: int = 1000,
    save_epochs: list[int] | None = None,
) -> list[dict]:
    """
    Trains the conditional DDPM and saves checkpoints + visual comparisons.

    Returns a list of per-epoch metric dicts:
        [{'epoch': int, 'loss': float}, …]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    effective_batch = batch_size * max(1, n_gpus)
    print(f"[{run_name}] device={device}, GPUs={n_gpus}, "
          f"effective batch={effective_batch}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset    = OwlSketchDataset(data_path, max_samples=max_samples)
    dataloader = DataLoader(
        dataset, batch_size=effective_batch,
        shuffle=True, drop_last=True, num_workers=2, pin_memory=True,
    )

    # ── Model + scheduler ────────────────────────────────────────────────────
    scheduler = build_scheduler(num_train_timesteps)
    model     = build_conditional_unet()

    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # ── Output dir + fixed batch for visual logging ──────────────────────────
    out_dir = f"output_images/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    fixed_y, fixed_x0 = next(iter(dataloader))
    fixed_y  = fixed_y.to(device)
    fixed_x0 = fixed_x0.to(device)

    if save_epochs is None:
        e = epochs
        save_epochs = sorted({1, e // 4, e // 2, (3 * e) // 4, e} - {0})

    history = []

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for y_batch, x0_batch in dataloader:
            y_batch  = y_batch.to(device)
            x0_batch = x0_batch.to(device)

            optimizer.zero_grad()

            # Dynamically set device_type and enable/disable AMP
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
                loss = compute_l_simple(model, scheduler, x0_batch, y_batch, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        elapsed  = time.time() - t0
        print(f"Epoch [{epoch}/{epochs}] | {elapsed:.1f}s | L_simple: {avg_loss:.6f}")
        history.append({"epoch": epoch, "loss": avg_loss})

        # ── Visual checkpoint ─────────────────────────────────────────────────
        if epoch in save_epochs:
            model.eval()
            samples = sample_speed_painting(
                model, scheduler, fixed_y, device,
                num_inference_steps=200,    # faster for logging
            )
            # Grid: rough sketch | generated x_0 | ground-truth x_0
            grid = torch.cat([fixed_y, samples, fixed_x0], dim=3)
            vutils.save_image(
                grid,
                f"{out_dir}/epoch_{epoch:04d}.png",
                normalize=True,
                nrow=effective_batch,
            )

    # ── Save weights ──────────────────────────────────────────────────────────
    ckpt_path = f"{run_name}_ddpm.pth"
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) \
            else model.state_dict()
    torch.save(state, ckpt_path)
    print(f"Model saved → {ckpt_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    DATA_PATH = "./owlsclean/"
    fetch_images_if_missing(DATA_PATH, num_images=1000)

    # ── Quick smoke-test ──────────────────────────────────────────────────────
    train_conditional_ddpm(
        run_name="trial_ddpm",
        data_path=DATA_PATH,
        max_samples=32,
        epochs=2,
        batch_size=2,
        learning_rate=1e-4,
    )

    # ── Ablation experiments ──────────────────────────────────────────────────
    experiments = [
        {"name": "exp1_baseline", "lr": 1e-4, "batch": 2},
        {"name": "exp2_high_lr",  "lr": 5e-4, "batch": 2},
        {"name": "exp3_large_bs", "lr": 1e-4, "batch": 4},
    ]

    results = {}
    for exp in experiments:
        history = train_conditional_ddpm(
            run_name=exp["name"],
            data_path=DATA_PATH,
            max_samples=None,
            epochs=20,
            batch_size=exp["batch"],
            learning_rate=exp["lr"],
        )
        results[exp["name"]] = history

    # ── Print final losses ────────────────────────────────────────────────────
    print("\n=== Final L_simple per experiment ===")
    for name, hist in results.items():
        print(f"  {name}: {hist[-1]['loss']:.6f}")


if __name__ == "__main__":
    main()