import torch


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    """Save model weights and useful information about the run."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": args,
    }
    torch.save(checkpoint, path)
