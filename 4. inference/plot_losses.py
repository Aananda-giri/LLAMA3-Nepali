from previous_chapters import Llama3Model# , plot_losses
from huggingface_hub import hf_hub_download
from pathlib import Path

import torch
import os
import matplotlib.pyplot as plt

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# ------------------------------------------------------------------------
# please change checkpoinit name and output plot would be at model_checkpoints/losses.pdf

# Download the model checkpoint
model_checkpoint_path = "model_checkpoints_3/model_pg_159000_steps.pth"
# ------------------------------------------------------------------------

hf_hub_download(repo_id="Aananda-giri/LLAMA3-Nepali", filename=model_checkpoint_path, local_dir='./')



def plot_losses(steps, tokens_seen, train_losses, val_losses, output_dir):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against steps
    ax1.plot(steps, train_losses, label=f"Training loss (final:{train_losses[-1]})")
    ax1.plot(steps, val_losses, linestyle="-.", label=f"Validation loss(final:{val_losses[-1]})")
    ax1.set_xlabel(f"Steps: {int(steps[-1])}")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    
    # Format x-axis to show steps in scientific notation if numbers are large
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig.tight_layout()  # Adjust layout to make room

    plt.savefig(output_dir / "losses.pdf")


# load the weights from checkpoint
def plot_losses2(model, model_checkpoint_path, loss_plot=True):
    checkpoint = torch.load(model_checkpoint_path, weights_only=False)
        
    # # modified (added model loading code)
    # model.load_state_dict(checkpoint["model_state_dict"])
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)  # the book accidentally omitted the lr assignment
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_losses = checkpoint["train_losses"]
    print(f'train_losses: {type(train_losses)}  len: {len(train_losses)}')

    val_losses = checkpoint["val_losses"]
    print(f'val_losses: {type(val_losses)}  len: {len(val_losses)}')

    track_tokens_seen = checkpoint["track_tokens_seen"]
    print(f'track_tokens_seen: {type(track_tokens_seen)}  len: {len(track_tokens_seen)}')

    track_lrs = checkpoint["track_lrs"]
    print(f'track_lrs: {type(track_lrs)}  len: {len(track_lrs)}')

    previous_epochs = checkpoint["epochs"]
    print(f'previous epochs: {type(previous_epochs)} {previous_epochs}')

    previous_global_step = checkpoint["global_step"]
    print(f'previous global step: {previous_global_step} \n previous epochs: {previous_epochs}')
    print(end = '\n' + '-'*70 + '\n')

    if loss_plot:
        # epochs_tensor = torch.linspace(0, 1, len(train_losses)) # n_epochs=1
        steps_tensor = torch.linspace(0,previous_global_step, len(train_losses))
        # plot losses
        plot_losses(steps_tensor, track_tokens_seen, train_losses, val_losses, output_dir=Path('./model_checkpoints/'))

    # return model

plot_losses2(model, model_checkpoint_path, loss_plot=True)