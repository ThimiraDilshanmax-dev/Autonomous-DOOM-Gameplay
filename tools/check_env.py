"""Environment sanity check: OS, GPU, vizdoom, torch, snnTorch."""
import platform
import sys


def main():
    print(f"OS: {platform.system()} {platform.release()} | python {sys.version.split()[0]}")
    if platform.system() == "Windows":
        print("WARN: native Windows ViZDoom is unreliable — use WSL2 or Colab for training.")
    try:
        import torch
        print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f" ({torch.cuda.get_device_name(0)})", end="")
        print()
    except ImportError:
        print("torch: MISSING (pip install torch --index-url https://download.pytorch.org/whl/cu121)")
    for pkg in ["snnTorch", "spikingjelly", "vizdoom", "stable_baselines3", "gymnasium"]:
        try:
            __import__(pkg)
            print(f"{pkg}: OK")
        except ImportError:
            print(f"{pkg}: MISSING")


if __name__ == "__main__":
    main()
