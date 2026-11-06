#!/usr/bin/env python3
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Multi-Trailer Reversing ControlTransformer Simulation & Game")
    parser.add_argument("--train", action="store_true", help="Run the synthetic training pipeline")
    parser.add_argument("--trajectories", type=int, default=180, help="Number of trajectories for training")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model-path", type=str, default="models/model.pt", help="Path to checkpoint")
    parser.add_argument("--from-scratch", action="store_true", default=True, help="Train model from scratch (do not load old checkpoint)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--stream", action="store_true", help="Dynamic streaming mode (sample fresh trajectories continuously)")
    args = parser.parse_args()

    if args.train:
        from src.train import train_model
        print("Launching ControlTransformer synthetic training pipeline...")
        train_model(
            num_trajectories=args.trajectories,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.model_path,
            from_scratch=not args.resume,
            stream=args.stream
        )
    else:
        from src.game import JackknifeGame25D
        print(f"Launching 2.5D Pygame simulation (Model: {args.model_path})...")
        game = JackknifeGame25D(model_path=args.model_path)
        game.run()

if __name__ == "__main__":
    main()
