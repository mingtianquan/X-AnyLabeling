import argparse
import sys
from pathlib import Path

try:
    from .train_dynamic import (
        export_ncnn_with_pnnx,
        export_torchscript_from_checkpoint,
    )
except ImportError:
    # Allow running this file directly: python export_dynamic.py ...
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from train_dynamic import (
        export_ncnn_with_pnnx,
        export_torchscript_from_checkpoint,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export CRNN checkpoint to TorchScript and NCNN via pnnx."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--torchscript-file", required=True)
    parser.add_argument("--ncnn-param-file", required=True)
    parser.add_argument("--ncnn-bin-file", required=True)
    parser.add_argument("--img-h", type=int, default=32)
    parser.add_argument("--trace-width", type=int, default=160)
    parser.add_argument("--trace-width2", type=int, default=320)
    parser.add_argument("--pnnx", default="pnnx")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    torchscript_path = Path(args.torchscript_file).resolve()
    ncnn_param_path = Path(args.ncnn_param_file).resolve()
    ncnn_bin_path = Path(args.ncnn_bin_file).resolve()

    export_torchscript_from_checkpoint(
        checkpoint_path=checkpoint_path,
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
    print(f"[done] ncnn param: {ncnn_param_path}", flush=True)
    print(f"[done] ncnn bin:   {ncnn_bin_path}", flush=True)


if __name__ == "__main__":
    main()
