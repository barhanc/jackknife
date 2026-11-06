import os
import math
import time
import argparse
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data_generator import SyntheticDataGenerator, ReversingDataset
from src.model import MultiTrailerPFN, NUM_STEER_CLASSES

def train_model(
    num_trajectories: int = 60,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = "models/pfn_best.pt",
    device_str: Optional[str] = None,
    from_scratch: bool = True,
    stream: bool = True
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"==================================================")
    print(f" Multi-Trailer Reversing RL-PFN Training Pipeline ")
    print(f" Device: {device} | From Scratch: {from_scratch} | Stream: {stream}")
    print(f"==================================================")

    generator = SyntheticDataGenerator()

    # Fixed validation set to evaluate generalisation
    print("Generating fixed validation benchmark set (30 trajectories)...")
    val_dataset = generator.generate_dataset(num_trajectories=30, perturb_ratio=0.5)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTrailerPFN().to(device)
    if not from_scratch and os.path.exists(save_path):
        try:
            ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded existing checkpoint from {save_path}")
        except Exception:
            pass
    elif from_scratch:
        print("Training model from fresh random weights (from scratch).")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MultiTrailerPFN architecture initialized ({total_params:,} parameters).")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0

    # If not streaming, generate a single static training set
    if not stream:
        print(f"Generating static training set ({num_trajectories} trajectories)...")
        train_dataset = generator.generate_dataset(num_trajectories=num_trajectories, perturb_ratio=0.5)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    metrics_history = []
    metrics_path = os.path.join(os.path.dirname(save_path) or ".", "training_metrics.json")
    metrics_csv_path = os.path.join(os.path.dirname(save_path) or ".", "training_metrics.csv")

    # If resuming from checkpoint, try loading existing metrics
    if not from_scratch and os.path.exists(metrics_path):
        try:
            import json
            with open(metrics_path, "r") as f:
                metrics_history = json.load(f)
            print(f"Loaded existing training metrics history ({len(metrics_history)} epochs).")
        except Exception:
            metrics_history = []

    for epoch in range(1, epochs + 1):
        if stream:
            # True PFN dynamic sampling: sample fresh trajectories every epoch
            t_gen = time.time()
            train_dataset = generator.generate_dataset(num_trajectories=num_trajectories, perturb_ratio=0.5)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            gen_time = time.time() - t_gen
        else:
            gen_time = 0.0

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_adj_correct = 0
        total_train = 0

        for tokens, hitch_state, targets in train_loader:
            tokens, hitch_state, targets = tokens.to(device), hitch_state.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, _ = model(tokens, hitch_state, return_attention=False)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == targets).sum().item()
            train_adj_correct += (torch.abs(preds - targets) <= 1).sum().item()
            total_train += len(targets)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_adj_correct = 0
        total_val = 0

        with torch.no_grad():
            for tokens, hitch_state, targets in val_loader:
                tokens, hitch_state, targets = tokens.to(device), hitch_state.to(device), targets.to(device)

                logits, _ = model(tokens, hitch_state, return_attention=False)
                loss = criterion(logits, targets)

                val_loss += loss.item() * len(targets)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == targets).sum().item()
                val_adj_correct += (torch.abs(preds - targets) <= 1).sum().item()
                total_val += len(targets)

        train_acc = train_correct / total_train * 100.0
        train_adj = train_adj_correct / total_train * 100.0
        val_acc = val_correct / total_val * 100.0
        val_adj = val_adj_correct / total_val * 100.0
        avg_train_loss = train_loss / total_train
        avg_val_loss = val_loss / total_val

        stream_info = f" [Stream +{gen_time:.1f}s]" if stream else ""
        print(
            f"Epoch {epoch:2d}/{epochs:2d}{stream_info} | "
            f"Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.1f}%, Adj: {train_adj:.1f}%) | "
            f"Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.1f}%, Adj: {val_adj:.1f}%)"
        )

        # Record metrics history for loss analysis & grokking detection
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "train_acc": float(train_acc),
            "train_adj_acc": float(train_adj),
            "val_acc": float(val_acc),
            "val_adj_acc": float(val_adj),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "timestamp": time.time()
        }
        metrics_history.append(epoch_record)

        try:
            import json
            with open(metrics_path, "w") as f:
                json.dump(metrics_history, f, indent=2)

            # Also append to CSV for easy spreadsheet/plotting access
            file_exists = os.path.exists(metrics_csv_path) and os.path.getsize(metrics_csv_path) > 0
            with open(metrics_csv_path, "a") as f:
                if not file_exists:
                    f.write("epoch,train_loss,val_loss,train_acc,train_adj_acc,val_acc,val_adj_acc,lr,timestamp\n")
                f.write(f"{epoch},{avg_train_loss:.6f},{avg_val_loss:.6f},{train_acc:.2f},{train_adj:.2f},{val_acc:.2f},{val_adj:.2f},{optimizer.param_groups[0]['lr']:.6e},{time.time():.2f}\n")
        except Exception as e:
            print(f"Warning: Failed to save metrics file: {e}")

        # Always save the newest model and preserve per-epoch intermediate checkpoints
        checkpoint_dict = {
            'model_state_dict': model.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_acc': train_acc,
            'train_adj_acc': train_adj,
            'val_acc': val_acc,
            'val_adj_acc': val_adj,
            'epoch': epoch,
            'total_params': total_params,
            'metrics_history': metrics_history
        }
        torch.save(checkpoint_dict, save_path)
        epoch_save_path = os.path.join(os.path.dirname(save_path), f"pfn_epoch_{epoch}.pt")
        torch.save(checkpoint_dict, epoch_save_path)
        print(f" -> Checkpoint saved to {save_path} and {epoch_save_path} (Latest: Epoch {epoch}, Val Adj Acc: {val_adj:.1f}%)")

    print(f"\nTraining completed! Latest checkpoint saved to: {save_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=60, help="Number of trajectories sampled per epoch in dynamic streaming mode")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="models/pfn_best.pt")
    parser.add_argument("--no-stream", action="store_true", help="Disable on-the-fly streaming (not recommended for PFN)")
    parser.add_argument("--from-scratch", action="store_true", help="Start training from random weights instead of existing checkpoint")
    args = parser.parse_args()

    train_model(
        num_trajectories=args.trajectories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        from_scratch=args.from_scratch,
        stream=not args.no_stream
    )
