
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=32):

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = ImageFolder(
        root=f"{data_dir}/train",
        transform=transform
    )

    val_dataset = ImageFolder(
        root=f"{data_dir}/val",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader
