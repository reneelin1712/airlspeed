import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the current CUDA device
    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    print("CUDA is not available. Using CPU.")
