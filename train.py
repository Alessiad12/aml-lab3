
import torch
from torch import nn
import wandb
from eval import validate
from utils.visualization import plot_training_history

from models.customnet import CustomNet
from dataset.tinyimagenet import get_dataloaders


def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total

    print(
        f"Train Epoch: {epoch} "
        f"Loss: {train_loss:.6f} "
        f"Acc: {train_accuracy:.2f}%"
    )

    return train_loss, train_accuracy


def main():
    # Configuration
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    learning_rate = 0.001
    momentum = 0.9
    batch_size = 32
    num_epochs = 10

    data_dir = "tiny-imagenet/tiny-imagenet-200"

    # Initialize W&B
    wandb.init(
        project="aml-lab3",
        config={
            "learning_rate": learning_rate,
            "momentum": momentum,
            "batch_size": batch_size,
            "epochs": num_epochs,
        }
    )

    # Load dataset
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    # Initialize model
    model = CustomNet().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}
    
    # Training and validation
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):

        train_loss, train_accuracy = train(
            epoch,
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        # Save best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    print(f"Best validation accuracy: {best_acc:.2f}%")

    wandb.finish()
    plot_training_history(history)



if __name__ == "__main__":
    main()
