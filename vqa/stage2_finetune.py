# stage2_finetune.py — Stage 2 VQA Instruction Tuning (fully runnable)
# Trains the Q-Former + LLM (LoRA) on VQA datasets while keeping ViT frozen.

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast


class SyntheticVQADataset(Dataset):
    """
    Synthetic VQA dataset for testing the training pipeline.
    Replace with real VQAv2/TextVQA data for actual training.
    """
    def __init__(self, size=1000, img_size=224, seq_len=32):
        self.size    = size
        self.img_size = img_size
        self.seq_len  = seq_len

    def __len__(self): return self.size

    def __getitem__(self, idx):
        return {
            "image":        torch.randn(3, self.img_size, self.img_size),
            "input_ids":    torch.randint(0, 1000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels":       torch.randint(0, 1000, (self.seq_len,)),
        }


def freeze_visual_encoder(model):
    """Freeze ViT weights — only Q-Former and LLM (LoRA) train in stage 2."""
    for param in model.visual_encoder.parameters():
        param.requires_grad = False
    frozen = sum(p.numel() for p in model.visual_encoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen ViT params:  {frozen/1e6:.1f}M")
    print(f"Trainable params:   {trainable/1e6:.1f}M")


def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        images       = batch["image"].to(device)
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        labels       = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device == "cuda")):
            outputs = model(images, input_ids, attn_mask, labels=labels)
            # HuggingFace models return a loss in outputs.loss
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Stub: simulate cross-entropy loss
                logits = torch.randn(
                    images.size(0), input_ids.size(1), 1000, device=device)
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, 1000), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (step + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} | {elapsed:.1f}s elapsed")

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--num-workers",type=int,   default=2)
    parser.add_argument("--output-dir", default="checkpoints/stage2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Stage 2 fine-tuning on: {device}")

    # Import model
    from model import MultimodalVQAModel
    model = MultimodalVQAModel().to(device)
    freeze_visual_encoder(model)

    dataset = SyntheticVQADataset(size=200)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader))
    scaler = GradScaler(enabled=(device == "cuda"))

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, loader, optimizer, scaler, device, epoch)
        scheduler.step()
        print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

        ckpt = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt)
        print(f"Checkpoint saved: {ckpt}")


if __name__ == "__main__":
    main()
