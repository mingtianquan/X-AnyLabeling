# CRNN Training (Dynamic Width + NCNN Export)

This project includes a lightweight CRNN training pipeline:

- Label generation from filename prefix
- Dynamic-width CTC training
- TorchScript export
- NCNN export via `pnnx`

## 1) Generate `labels.txt` from filename prefix

```bash
python -m anylabeling.services.auto_training.crnn.labeling \
  --data-root D:/yolov8/datasets/crnn \
  --labels-file D:/yolov8/datasets/crnn/labels.txt \
  --split-char _
```

Expected label line format:

```text
tx/22225_d375c089db599bafdddaceaa286fc43a.jpg\t22225
```

## 2) Train and export NCNN

```bash
python -m anylabeling.services.auto_training.crnn.train_dynamic \
  --data-root D:/yolov8/datasets/crnn \
  --auto-labels \
  --epochs 50 \
  --batch-size 32 \
  --pnnx pnnx
```

Outputs (default under `--data-root`):

- `best_crnn_dynamic.pth`
- `latest_crnn_dynamic.pth`
- `crnn.pt`
- `crnn.ncnn.param`
- `crnn.ncnn.bin`

## 3) Labeling UI integration

In X-AnyLabeling GUI:

- `Upload -> CRNN` imports `labels.txt` to json labels (`description` field).
- `Export -> CRNN` exports current json annotations to `labels.txt`.

When exporting CRNN labels, you can enable fallback to filename prefix for images without text annotation.
