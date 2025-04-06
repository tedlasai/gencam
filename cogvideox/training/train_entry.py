import signal
import sys
import os
from accelerate import Accelerator
from accelerate.utils import send_example_telemetry
import argparse

import yaml
# Import your main training function from train_controlnet
from train_controlnet import main  # assuming your actual training code is in `main()`.

def parse_args():
    parser = argparse.ArgumentParser(description="Training with Accelerator")
    # Define the config argument with a flag (e.g., --config)
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    args = argparse.Namespace(**config)
    
    return parser.parse_args()

def handler(signum, frame):
    print(f"[PID {os.getpid()}] Caught SIGUSR1, saving checkpoint and exiting...")
    #get --config argument
    parser = argparse.ArgumentParser(description="Training script")
    save_path = os.path.join(args.output_dir, f"checkpoint")
    accelerator.save_state(save_path)
    sys.exit(0)

# Register the signal handler for SIGUSR1
signal.signal(signal.SIGUSR1, handler)

if __name__ == "__main__":
    # Now you need to call `accelerate.launch()` inside this script, to manage multi-GPU, etc.
    accelerator = Accelerator()
    # Ensure `main()` runs inside this, which can now handle the signal.
    accelerator.launch(main, args=[sys.argv[1]])  # Pass the config file argument here