import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .labeling import generate_labels_file, parse_image_exts
except ImportError:
    # Allow running this file directly: python train_dynamic.py ...
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from labeling import generate_labels_file, parse_image_exts


DEFAULT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples_from_labels_file(
    labels_file: Path, data_root: Path
) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    with labels_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            if "\t" in row:
                image_rel, label = row.split("\t", 1)
            else:
                parts = row.rsplit(maxsplit=1)
                if len(parts) != 2:
                    continue
                image_rel, label = parts[0], parts[1]
            image_rel = image_rel.strip()
            label = label.strip()
            if not image_rel or not label:
                continue
            image_path = (data_root / image_rel).resolve()
            if image_path.exists():
                samples.append((image_path, label))
    return samples


def build_charset(
    samples: Sequence[Tuple[Path, str]], charset_file: Path
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    charset: Set[str] = set()
    for _, label in samples:
        charset.update(label)
    sorted_chars = sorted(charset)
    charset_file.parent.mkdir(parents=True, exist_ok=True)
    with charset_file.open("w", encoding="utf-8") as f:
        for c in sorted_chars:
            f.write(c + "\n")

    char_to_idx = {c: i + 1 for i, c in enumerate(sorted_chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(sorted_chars)}
    return sorted_chars, char_to_idx, idx_to_char


def split_samples(
    samples: Sequence[Tuple[Path, str]], train_ratio: float, seed: int
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    n = len(samples)
    if n <= 1:
        return list(samples), []
    train_ratio = min(max(train_ratio, 0.0), 1.0)
    train_size = int(n * train_ratio)
    train_size = min(max(train_size, 1), n - 1)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_ids = set(indices[:train_size])
    train_samples = [samples[i] for i in range(n) if i in train_ids]
    val_samples = [samples[i] for i in range(n) if i not in train_ids]
    return train_samples, val_samples


class ResizeNormalizeDynamic:
    def __init__(self, img_h=32, min_img_w=16, max_img_w=512):
        self.img_h = img_h
        self.min_img_w = min_img_w
        self.max_img_w = max_img_w

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, int]:
        w, h = img.size
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image size: {(w, h)}")
        new_w = int(round(w * (self.img_h / h)))
        new_w = max(new_w, self.min_img_w)
        if self.max_img_w is not None:
            new_w = min(new_w, self.max_img_w)
        new_w = max(new_w, 1)

        img = img.resize((new_w, self.img_h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.expand_dims(arr, axis=0)  # [1, H, W]
        return torch.from_numpy(arr), new_w


def calc_crnn_output_width(input_w: int) -> int:
    w = input_w
    w = (w - 2) // 2 + 1  # maxpool(2,2)
    w = (w - 2) // 2 + 1  # maxpool(2,2)
    w = w - 2 + 1  # conv(2,1,0)
    return w


class CRNNDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, str]],
        char_to_idx: Dict[str, int],
        transform: ResizeNormalizeDynamic,
    ):
        self.samples = list(samples)
        self.char_to_idx = char_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, text = self.samples[idx]
        img = Image.open(image_path).convert("L")
        img_tensor, resized_w = self.transform(img)
        label_ids = torch.tensor(
            [self.char_to_idx[c] for c in text], dtype=torch.long
        )
        input_length = calc_crnn_output_width(resized_w)
        if input_length <= 0:
            raise ValueError(
                f"Input length <= 0 for image: {image_path}, resized_w={resized_w}"
            )
        if len(label_ids) > input_length:
            raise ValueError(
                f"Label too long for CTC: image={image_path}, "
                f"label_len={len(label_ids)}, input_length={input_length}"
            )
        return img_tensor, label_ids, len(label_ids), input_length, text


def crnn_dynamic_collate_fn(batch):
    images = []
    labels = []
    label_lengths = []
    input_lengths = []
    raw_labels = []

    max_w = max(item[0].shape[-1] for item in batch)
    for img, label_ids, label_len, input_len, raw_label in batch:
        pad_w = max_w - img.shape[-1]
        if pad_w > 0:
            # normalized white value
            img = F.pad(img, (0, pad_w, 0, 0), value=1.0)
        images.append(img)
        labels.append(label_ids)
        label_lengths.append(label_len)
        input_lengths.append(input_len)
        raw_labels.append(raw_label)

    images = torch.stack(images, 0)
    labels = torch.cat(labels, 0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    return images, labels, label_lengths, input_lengths, raw_labels


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(inplace=True),
        )
        self.rnn1 = nn.LSTM(
            input_size=512, hidden_size=256, bidirectional=True, batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=512, hidden_size=256, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        _, _, h, _ = x.size()
        if not torch.jit.is_tracing() and h != 1:
            raise RuntimeError(f"CNN output height must be 1, got {h}")
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)  # [T, B, C]
        return x


def ctc_greedy_decode(log_probs: torch.Tensor, idx_to_char: Dict[int, str]):
    preds = log_probs.argmax(2).permute(1, 0)
    results = []
    for seq in preds:
        prev = -1
        text = []
        for idx in seq.tolist():
            if idx != 0 and idx != prev:
                text.append(idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(text))
    return results


def evaluate(model, loader, criterion, device, idx_to_char):
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0

    with torch.no_grad():
        for images, labels, label_lengths, input_lengths, raw_labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            input_lengths = input_lengths.to(device)

            outputs = model(images)
            log_probs = outputs.log_softmax(2)
            actual_t = outputs.size(0)
            if torch.any(input_lengths > actual_t):
                raise ValueError(
                    f"input_lengths > actual_t: {input_lengths.tolist()} > {actual_t}"
                )
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()

            preds = ctc_greedy_decode(log_probs, idx_to_char)
            for pred, gt in zip(preds, raw_labels):
                total_count += 1
                if pred == gt:
                    correct += 1

    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / total_count if total_count > 0 else 0.0
    return avg_loss, acc


def train_one_epoch(
    model, loader, criterion, optimizer, device, epoch, total_epochs, print_every
):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (
        images,
        labels,
        label_lengths,
        input_lengths,
        _,
    ) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        input_lengths = input_lengths.to(device)

        outputs = model(images)
        log_probs = outputs.log_softmax(2)
        actual_t = outputs.size(0)
        if torch.any(input_lengths > actual_t):
            raise ValueError(
                f"input_lengths > actual_t: {input_lengths.tolist()} > {actual_t}"
            )
        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"epoch {epoch}/{total_epochs} "
                f"batch {batch_idx + 1}/{num_batches} "
                f"loss={avg_loss:.4f} max_w={images.shape[-1]} T={actual_t}",
                flush=True,
            )

    return total_loss / max(num_batches, 1)


def safe_torch_load(checkpoint_path: Path):
    try:
        return torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def save_checkpoint(
    path: Path,
    model: CRNN,
    optimizer,
    epoch: int,
    best_acc: float,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
        },
        str(path),
    )


def export_torchscript_from_checkpoint(
    checkpoint_path: Path,
    torchscript_path: Path,
    trace_h: int,
    trace_w: int,
):
    ckpt = safe_torch_load(checkpoint_path)
    state_dict = ckpt["model_state_dict"]
    char_to_idx = ckpt.get("char_to_idx", {})
    num_classes = len(char_to_idx) + 1
    if num_classes <= 1 and "fc.weight" in state_dict:
        num_classes = state_dict["fc.weight"].shape[0]
    model = CRNN(num_classes=num_classes)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy = torch.randn(1, 1, trace_h, trace_w)
    traced = torch.jit.trace(model, dummy)
    torchscript_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(torchscript_path))
    print(f"[export] torchscript saved: {torchscript_path}", flush=True)


def _find_generated_ncnn_files(workdir: Path) -> Tuple[Path, Path]:
    params = sorted(
        workdir.glob("*.ncnn.param"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    bins = sorted(
        workdir.glob("*.ncnn.bin"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not params or not bins:
        raise FileNotFoundError("No ncnn.param/bin generated by pnnx")
    return params[0], bins[0]


def export_ncnn_with_pnnx(
    torchscript_path: Path,
    pnnx_bin: str,
    input_h: int,
    input_w1: int,
    input_w2: int,
    output_param: Path,
    output_bin: Path,
):
    workdir = torchscript_path.parent
    cmd = [
        pnnx_bin,
        str(torchscript_path),
        f"inputshape=[1,1,{input_h},{input_w1}]",
        f"inputshape2=[1,1,{input_h},{input_w2}]",
    ]
    print(f"[export] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(workdir), check=True)

    generated_param, generated_bin = _find_generated_ncnn_files(workdir)
    output_param.parent.mkdir(parents=True, exist_ok=True)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    if generated_param.resolve() != output_param.resolve():
        shutil.copy2(generated_param, output_param)
    if generated_bin.resolve() != output_bin.resolve():
        shutil.copy2(generated_bin, output_bin)
    print(f"[export] ncnn param: {output_param}", flush=True)
    print(f"[export] ncnn bin:   {output_bin}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CRNN dynamic-width training + TorchScript + NCNN export"
    )
    parser.add_argument(
        "--data-root", required=True, help="Dataset image root directory."
    )
    parser.add_argument(
        "--labels-file",
        default="",
        help="labels.txt path. Defaults to <data-root>/labels.txt",
    )
    parser.add_argument(
        "--charset-file",
        default="",
        help="charset.txt path. Defaults to <data-root>/charset.txt",
    )
    parser.add_argument(
        "--auto-labels",
        action="store_true",
        help="Auto-generate labels.txt from filename prefix.",
    )
    parser.add_argument(
        "--auto-labels-refresh",
        action="store_true",
        help="Force regenerate labels.txt before training.",
    )
    parser.add_argument(
        "--label-split-char",
        default="_",
        help="Filename split char used by --auto-labels.",
    )
    parser.add_argument(
        "--label-image-exts",
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Image extensions used by --auto-labels.",
    )

    parser.add_argument("--best-model", default="")
    parser.add_argument("--latest-model", default="")
    parser.add_argument("--resume", default="")

    parser.add_argument("--img-h", type=int, default=32)
    parser.add_argument("--min-img-w", type=int, default=16)
    parser.add_argument("--max-img-w", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto/cpu/cuda/0/1/0,1 ...",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--torchscript-file", default="")
    parser.add_argument("--ncnn-param-file", default="")
    parser.add_argument("--ncnn-bin-file", default="")
    parser.add_argument("--pnnx", default="pnnx")
    parser.add_argument("--trace-width", type=int, default=160)
    parser.add_argument("--trace-width2", type=int, default=320)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    value = str(device_arg or "auto").strip().lower()
    if value in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if "," in value:
        value = value.split(",", 1)[0].strip()
    if value.isdigit():
        gpu_index = int(value)
        if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu_index}")
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[env] device={device}", flush=True)

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    labels_file = (
        Path(args.labels_file).resolve()
        if args.labels_file
        else data_root / "labels.txt"
    )
    charset_file = (
        Path(args.charset_file).resolve()
        if args.charset_file
        else data_root / "charset.txt"
    )
    best_model_path = (
        Path(args.best_model).resolve()
        if args.best_model
        else data_root / "best_crnn_dynamic.pth"
    )
    latest_model_path = (
        Path(args.latest_model).resolve()
        if args.latest_model
        else data_root / "latest_crnn_dynamic.pth"
    )
    torchscript_path = (
        Path(args.torchscript_file).resolve()
        if args.torchscript_file
        else data_root / "crnn.pt"
    )
    ncnn_param_path = (
        Path(args.ncnn_param_file).resolve()
        if args.ncnn_param_file
        else data_root / "crnn.ncnn.param"
    )
    ncnn_bin_path = (
        Path(args.ncnn_bin_file).resolve()
        if args.ncnn_bin_file
        else data_root / "crnn.ncnn.bin"
    )

    if args.auto_labels and (args.auto_labels_refresh or not labels_file.exists()):
        image_exts = parse_image_exts(args.label_image_exts)
        count, skipped = generate_labels_file(
            data_root=data_root,
            labels_file=labels_file,
            split_char=args.label_split_char,
            image_exts=image_exts,
        )
        print(
            f"[labels] generated: {labels_file} (lines={count}, skipped={skipped})",
            flush=True,
        )

    if not labels_file.exists():
        raise FileNotFoundError(f"labels file not found: {labels_file}")

    samples = load_samples_from_labels_file(labels_file, data_root)
    if not samples:
        raise ValueError("No valid samples loaded from labels.txt")
    print(f"[data] loaded samples={len(samples)}", flush=True)

    chars, char_to_idx, idx_to_char = build_charset(samples, charset_file)
    print(f"[data] charset size={len(chars)} -> {charset_file}", flush=True)

    train_samples, val_samples = split_samples(
        samples=samples, train_ratio=args.train_ratio, seed=args.seed
    )
    print(
        f"[data] train={len(train_samples)} val={len(val_samples)}",
        flush=True,
    )

    train_set = CRNNDataset(
        train_samples,
        char_to_idx,
        ResizeNormalizeDynamic(args.img_h, args.min_img_w, args.max_img_w),
    )
    val_set = CRNNDataset(
        val_samples,
        char_to_idx,
        ResizeNormalizeDynamic(args.img_h, args.min_img_w, args.max_img_w),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=crnn_dynamic_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=crnn_dynamic_collate_fn,
    )

    model = CRNN(num_classes=len(chars) + 1).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=max(args.weight_decay, 0.0),
    )

    best_acc = 0.0
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if resume_path.exists():
            ckpt = safe_torch_load(resume_path)
            state_dict = ckpt["model_state_dict"]
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {
                    k.replace("module.", "", 1): v
                    for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=False)
            best_acc = float(ckpt.get("best_acc", 0.0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(
                f"[resume] loaded checkpoint={resume_path}, start_epoch={start_epoch}",
                flush=True,
            )

    no_improve_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            print_every=max(1, args.print_every),
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            idx_to_char=idx_to_char,
        )
        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        improved = val_acc > (best_acc + max(args.early_stop_min_delta, 0.0))
        if improved:
            best_acc = val_acc
            no_improve_epochs = 0
            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
            )
            print(f"[save] best -> {best_model_path}", flush=True)
        else:
            no_improve_epochs += 1

        save_checkpoint(
            path=latest_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_acc=best_acc,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
        )
        print(f"[save] latest -> {latest_model_path}", flush=True)

        if (
            args.early_stop_patience > 0
            and no_improve_epochs >= args.early_stop_patience
        ):
            print(
                f"[early-stop] no improvement for {no_improve_epochs} epochs",
                flush=True,
            )
            break

    print(f"[done] best_val_acc={best_acc:.4f}", flush=True)

    if args.skip_export:
        print("[export] skipped by --skip-export", flush=True)
        return

    export_ckpt = best_model_path if best_model_path.exists() else latest_model_path
    if not export_ckpt.exists():
        raise FileNotFoundError("No checkpoint available for export")

    export_torchscript_from_checkpoint(
        checkpoint_path=export_ckpt,
        torchscript_path=torchscript_path,
        trace_h=args.img_h,
        trace_w=args.trace_width,
    )
    export_ncnn_with_pnnx(
        torchscript_path=torchscript_path,
        pnnx_bin=args.pnnx,
        input_h=args.img_h,
        input_w1=args.trace_width,
        input_w2=args.trace_width2,
        output_param=ncnn_param_path,
        output_bin=ncnn_bin_path,
    )


if __name__ == "__main__":
    main()
