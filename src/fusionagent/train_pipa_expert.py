"""Fine-tune cue-specific ResNet-50 experts on PIPA train identities only."""

import argparse
import csv
from pathlib import Path

import h5py
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


class PIPACueDataset(Dataset):
    def __init__(self, manifest, h5_path, train=True):
        with open(manifest, "r", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
        pids = sorted({int(r["pid"]) for r in rows})
        self.pid2label = {pid: i for i, pid in enumerate(pids)}
        self.rows = rows
        self.h5_path = str(h5_path)
        self.h5 = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def _file(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        return self.h5

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.fromarray(self._file()[row["key"]][:]).convert("RGB")
        return self.transform(image), self.pid2label[int(row["pid"])]


class CueClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = torch.flatten(self.backbone(x), 1)
        return self.classifier(feat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cue", choices=["head", "upper"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data_dir = Path(args.data_root) / "PIPA_FusionAgent"
    manifest = data_dir / "manifest.csv"
    h5_path = data_dir / ("pipa_head.h5" if args.cue == "head" else "pipa_upper.h5")
    dataset = PIPACueDataset(manifest, h5_path, train=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CueClassifier(len(dataset.pid2label)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
        print(
            f"epoch={epoch + 1:03d} loss={total_loss / max(total, 1):.4f} "
            f"train_acc={correct / max(total, 1):.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone_state_dict": model.backbone.state_dict(),
        "cue": args.cue,
        "num_train_ids": len(dataset.pid2label),
        "seed": args.seed,
    }, output)
    print(f"Saved PIPA {args.cue} expert to {output}")


if __name__ == "__main__":
    main()
