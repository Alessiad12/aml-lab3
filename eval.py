
import torch


def validate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Calculate metrics
            val_loss += loss.item()

            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total

    print(
        f"Validation Loss: {val_loss:.6f} "
        f"Acc: {val_accuracy:.2f}%"
    )

    return val_loss, val_accuracy
