"""Project entry point."""
import argparse
import json
import torch

from config import config_as_dict, SimulationConfig


def main():
    parser = argparse.ArgumentParser(description="MAHDRL paper replication")
    parser.add_argument("--check-config", action="store_true",
                        help="print configuration and check runtime dependencies")
    args = parser.parse_args()

    if args.check_config:
        cfg = config_as_dict()
        print(json.dumps(cfg, indent=2))
        print("\\nRuntime check:")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Selected device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print("Configuration check passed.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
