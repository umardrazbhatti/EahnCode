"""
run_full_pipeline.py — Entry point for Kaggle / local execution.

Usage (Kaggle):
    !python run_full_pipeline.py \\
        --data_root /kaggle/input/.../ffpp_data \\
        --dataset_name ff++ \\
        --epochs 10 \\
        --batch_size 4 \\
        --eval_after_train

Usage (local synthetic smoke test):
    python run_full_pipeline.py --dataset_name synthetic --epochs 2 --batch_size 2
"""

import os
from config import EAHNConfig, parse_args
from scripts.train_real import main as train_main


def main():
    args   = parse_args()
    config = EAHNConfig.from_args(args)
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Output directory: {config.output_dir}")
    print(f"Device: {config.device}")
    print(f"Dataset: {config.dataset_name}")
    train_main(config)
    print("Full pipeline completed. Outputs in", config.output_dir)


if __name__ == "__main__":
    main()
