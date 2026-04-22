import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import yaml

LogCallback = Callable[[str], None]

DEFAULT_NCNN_EXPORT_SCRIPT = str(
    Path(__file__).resolve().with_name("export_yolo_to_ncnn.py")
)
SCRIPT_PATH_ENV = "XANYLABELING_NCNN_EXPORT_SCRIPT"

_TASK_MAP: Dict[str, str] = {
    "detect": "det",
    "segment": "seg",
    "pose": "pose",
    "obb": "obb",
    "classify": "cls",
}


def _emit(log_callback: Optional[LogCallback], message: str) -> None:
    if log_callback:
        log_callback(message)


def _resolve_script_path() -> str:
    return os.environ.get(SCRIPT_PATH_ENV, "").strip() or DEFAULT_NCNN_EXPORT_SCRIPT


def _detect_model_version(text: str) -> Optional[str]:
    probe = (text or "").lower()
    if "yolo26" in probe or "yolov26" in probe:
        return "yolo26"
    if "yolo11" in probe or "yolov11" in probe:
        return "yolo11"
    if "yolov8" in probe or "yolo8" in probe:
        return "yolov8"
    return None


def _safe_read_args_yaml(project_path: str) -> Dict:
    args_path = os.path.join(project_path, "args.yaml")
    if not os.path.exists(args_path):
        return {}
    try:
        with open(args_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _detect_export_profile(
    project_path: str, weights_path: str
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    args_data = _safe_read_args_yaml(project_path)

    task_raw = str(args_data.get("task", "")).lower()
    model_type = _TASK_MAP.get(task_raw)

    model_hint = str(args_data.get("model", ""))
    model_version = _detect_model_version(model_hint)
    if not model_version:
        model_version = _detect_model_version(weights_path)

    imgsz = args_data.get("imgsz")
    imgsz_value = None
    if isinstance(imgsz, int) and imgsz > 0:
        imgsz_value = imgsz

    return model_version, model_type, imgsz_value


def _guess_output_path(weights_path: str, model_type: str) -> str:
    model_dir = os.path.dirname(weights_path)
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    output_name = stem.replace("-", "_")
    if model_type == "cls":
        output_name = stem
    return os.path.join(model_dir, f"{output_name}.ncnn.param")


def run_custom_ncnn_export(
    runtime_python: str,
    project_path: str,
    weights_path: str,
    log_callback: Optional[LogCallback] = None,
) -> Tuple[bool, str]:
    script_path = _resolve_script_path()
    if not os.path.exists(script_path):
        return False, f"Custom NCNN export script not found: {script_path}"

    model_version, model_type, imgsz = _detect_export_profile(
        project_path, weights_path
    )
    if not model_version or not model_type:
        return (
            False,
            "Failed to detect model profile for custom NCNN export (version/task).",
        )

    command = [
        runtime_python,
        script_path,
        model_version,
        model_type,
        weights_path,
    ]
    if imgsz:
        command.extend(["--imgsz", str(imgsz)])

    _emit(log_callback, f"[NCNN] Using custom exporter: {script_path}")
    _emit(log_callback, f"[NCNN] $ {' '.join(command)}")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as e:
        return False, f"Failed to start custom NCNN export process: {e}"

    assert process.stdout is not None
    for line in process.stdout:
        cleaned = line.rstrip()
        if cleaned:
            _emit(log_callback, cleaned)

    return_code = process.wait()
    if return_code != 0:
        return False, f"Custom NCNN export failed with exit code {return_code}"

    exported_param = _guess_output_path(weights_path, model_type)
    if os.path.exists(exported_param):
        return True, exported_param

    return False, f"Custom NCNN export completed but file not found: {exported_param}"
