import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
import time
# Changed from tqdm.notebook to standard tqdm
from tqdm import tqdm
# Removed: from IPython import display - not needed for file saving

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Data Loading (remains the same) ---
def load_cifar10(batch_size=64, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Calculate split sizes
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    # Split training dataset into train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# --- Checkpointing (remains the same) ---
def save_checkpoint(model, optimizer, epoch, loss, save_path='checkpoints'):
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}") # Added newline for better formatting with tqdm

def load_checkpoint(model, optimizer, checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Resuming from Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    return checkpoint['epoch'], checkpoint['loss']

# --- Visualization Function (Modified for Saving Files) ---
def save_plots(metrics, save_path, filename="metrics_plot.png"):
    """Saves the training metrics plots to a file."""
    # Exclude 'images' key if present, as it's handled separately
    plot_metrics = {k: v for k, v in metrics.items() if k != 'images'}
    if not plot_metrics: # Don't create plot if only 'images' key exists or metrics is empty
        return

    max_cols = 4
    n_plots = len(plot_metrics)
    n_cols = min(n_plots, max_cols)
    n_rows = math.ceil(n_plots / n_cols)
    figsize = (15, 3 * n_rows)

    # Use a non-interactive backend suitable for saving files without displaying
    # plt.switch_backend('Agg') # Usually not strictly necessary but good practice

    fig = plt.figure(num=200, figsize=figsize) # Keep using a specific figure number
    plt.clf()  # Clear the current figure

    for idx, name in enumerate(list(plot_metrics.keys())):
        plt.subplot(n_rows, n_cols, idx + 1)
        if len(metrics[name]) > 0:  # Only plot if we have data
            plt.plot(metrics[name], label=name)
        plt.title(name)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend() # Added legend

    plt.suptitle(filename.replace('.png','').replace('_', ' ').title()) # Add overall title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    full_save_path = os.path.join(save_path, filename)
    os.makedirs(save_path, exist_ok=True) # Ensure directory exists
    plt.savefig(full_save_path)
    # print(f"Saved metrics plot to {full_save_path}") # Optional: print save confirmation
    plt.close(fig) # Close the figure to free memory

# --- Main Training Function (Modified for Script Execution) ---
def train_diffusion(diffusion_process, train_loader,
                    num_epochs=30, lr=1e-4, beta1=0.9, # Default Adam beta1 is 0.9
                    num_vis_samples=25, # Number of samples to generate for visualization
                    vis_interval=200, # How often (in iterations) to save plots
                    save_checkpoint_interval=5, # How often (in epochs) to save checkpoints
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    output_dir="diffusion_output",
                    resume_checkpoint=None): # Path to checkpoint to resume from
    """Train the Diffusion Model, saving plots and checkpoints."""

    # --- Setup Optimizer (moved inside train function for clarity) ---
    # Assuming diffusion_process has an optimizer attribute setup in its __init__
    # If not, you need to create it here based on diffusion_process.model.parameters()
    if not hasattr(diffusion_process, 'optimizer') or diffusion_process.optimizer is None:
         diffusion_process.optimizer = optim.Adam(diffusion_process.model.parameters(), lr=lr, betas=(beta1, 0.999))
         print("Optimizer created in train_diffusion function.")
    else:
        # Update LR and Betas if already exists
        print("Using optimizer provided by DiffusionProcess.")
        for param_group in diffusion_process.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['betas'] = (beta1, 0.999)

    optimizer = diffusion_process.optimizer
    start_epoch = 0
    # --- Resume from Checkpoint ---
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_epoch, _ = load_checkpoint(diffusion_process.model, optimizer, resume_checkpoint)
        start_epoch += 1 # Start from the next epoch
        print(f"Resuming training from epoch {start_epoch}")
    else:
         print("Starting training from scratch.")


    diffusion_process.model.to(device)

    # Metrics tracking
    metrics = {
        'MSE_losses': [], # Store the MSE loss per batch
        # 'images' key is no longer needed here, images saved directly
    }

    # --- Output Directories ---
    samples_dir = os.path.join(output_dir, "samples")
    plots_dir = os.path.join(output_dir, "plots")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Outputs will be saved in: {output_dir}")

    print(f"Starting Standard Diffusion Training on device: {device}...")
    print(f"Params: Epochs={num_epochs}, Start Epoch={start_epoch}, LR={lr}, Beta1={beta1}")

    global_step = 0 # Keep track of total iterations

    for epoch in range(start_epoch, num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        start_time = time.time()
        diffusion_process.model.train() # Ensure model is in training mode
        epoch_loss_sum = 0.0
        num_batches = 0

        # Use standard tqdm for terminal progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

        for i, (real_images, _) in pbar:
            real_images = real_images.to(device)

            loss = diffusion_process.train_step(real_images) # Assumes train_step returns scalar loss

            # Check for NaN loss
            # if torch.isnan(loss).any():
            #     print(f"Warning: NaN loss detected at epoch {epoch+1}, iteration {i}. Stopping training.")
            #     # Optionally save a checkpoint before exiting
            #     # save_checkpoint(diffusion_process.model, optimizer, epoch, float('nan'), save_path=checkpoints_dir)
            #     return metrics # Or raise an error

            metrics['MSE_losses'].append(loss) # Store loss value
            epoch_loss_sum += loss
            num_batches += 1
            global_step += 1

            pbar.set_postfix({'MSE_Loss': f"{loss:.4f}"}) # Update progress bar

            # --- Save Metrics Plot periodically ---
            if global_step % vis_interval == 0:
                 save_plots(metrics, plots_dir, filename=f"metrics.png")

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
        epoch_duration = time.time() - start_time
        print(f"\nEpoch {epoch+1} completed in {epoch_duration:.2f}s. Average Loss: {avg_epoch_loss:.4f}")

        # --- Save Generated Samples ---
        diffusion_process.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Generate samples
            vis_samples = diffusion_process.sample(num_samples=num_vis_samples) # sample() should handle device placement
            vis_samples = vis_samples.detach().cpu() # Move to CPU for saving

            # Save the grid of samples
            sample_filename = os.path.join(samples_dir, f"samples_epoch_{epoch+1:04d}.png")
            vutils.save_image(
                vis_samples,
                sample_filename,
                nrow=int(math.sqrt(num_vis_samples)), # Arrange in a square grid
                normalize=True, # Normalize images to [0, 1] for saving
                padding=2
            )
            print(f"Saved generated samples to {sample_filename}")

        # --- Save Final Metrics Plot for the Epoch ---
        save_plots(metrics, plots_dir, filename=f"metrics_epoch_{epoch+1}_final.png")

        # --- Save Checkpoints ---
        if (epoch + 1) % save_checkpoint_interval == 0 or epoch == num_epochs - 1:
            save_checkpoint(
                model=diffusion_process.model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_epoch_loss,
                save_path=checkpoints_dir
            )

    print("\n--- Training Finished ---")
    return metrics


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 128 # Increased batch size for potentially faster training
    VAL_SPLIT = 0.0 # Set to 0 if no validation set is needed during diffusion training
    IMG_SIZE = (32, 32)
    CHANNELS = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NUM_EPOCHS = 750       # Number of training epochs
    LEARNING_RATE = 1e-4  # Adjusted learning rate (common for diffusion)
    ADAM_BETA1 = 0.9      # Adam optimizer beta1
    NOISE_STEPS = 1000    # Number of steps in diffusion process
    VIS_SAMPLES = 25      # Number of samples to generate (should be a square number ideally)
    VIS_INTERVAL = 20000    # Save metrics plot every N iterations
    SAVE_INTERVAL = 150    # Save checkpoint every N epochs
    OUTPUT_DIR = "diffusion_cifar10_output" # Folder for all outputs
    # CHECKPOINT_TO_RESUME = None # Set path like "diffusion_output/checkpoints/checkpoint_epoch_X.pt" to resume
    CHECKPOINT_TO_RESUME = None 

    # --- Load Data ---
    print("Loading cifar10 dataset...")
    train_loader, _, test_loader = load_cifar10(batch_size=BATCH_SIZE, val_split=VAL_SPLIT)
    print(f"Training batches: {len(train_loader)}")

    # --- Initialize Diffusion Process ---
    # Make sure the DiffusionProcess class is defined in 'diffusion.py'
    # or imported correctly.
    try:
        from diffusion import DiffusionProcess # Assuming diffusion.py is in the same directory
    except ImportError:
        print("Error: Could not import DiffusionProcess from diffusion.py.")
        print("Please ensure diffusion.py is in the same directory or accessible.")
        exit()

    print("Initializing Diffusion Process...")
    diffusion_proc = DiffusionProcess(
        image_size=IMG_SIZE,
        channels=CHANNELS,
        noise_steps=NOISE_STEPS,
        device=DEVICE
        # Ensure DiffusionProcess initializes its model and optionally optimizer
    )

    # --- Start Training ---
    print("Starting training...")
    training_metrics = train_diffusion(
        diffusion_process=diffusion_proc,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        beta1=ADAM_BETA1,
        num_vis_samples=VIS_SAMPLES,
        vis_interval=VIS_INTERVAL,
        save_checkpoint_interval=SAVE_INTERVAL,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        resume_checkpoint=CHECKPOINT_TO_RESUME
    )

    print("\nTraining complete. Final metrics might include:")
    # Print summary of final metrics if needed (e.g., last loss)
    if 'MSE_losses' in training_metrics and training_metrics['MSE_losses']:
         print(f" - Last recorded MSE Loss: {training_metrics['MSE_losses'][-1]:.4f}")

    # You can add code here to use the trained model for generation,
    # load the final checkpoint, etc.
    print(f"\nCheck '{OUTPUT_DIR}' for saved samples, plots, and checkpoints.")