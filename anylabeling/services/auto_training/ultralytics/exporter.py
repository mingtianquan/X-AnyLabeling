import os
import json
import tempfile
import subprocess
import threading
from pathlib import Path
from io import StringIO
from typing import Dict, List, Tuple

from PyQt6.QtCore import QObject, pyqtSignal


from .runtime_env import ensure_ultralytics_runtime
from .custom_ncnn_export import run_custom_ncnn_export

CRNN_EXPORT_SCRIPT_ENV = "XANYLABELING_CRNN_EXPORT_SCRIPT"
DEFAULT_CRNN_EXPORT_SCRIPT = str(
    Path(__file__).resolve().parents[1] / "crnn" / "export_dynamic.py"
)


class ExportEventRedirector(QObject):
    """Thread-safe export event redirector"""

    export_event_signal = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

    def emit_export_event(self, event_type, data):
        """Safely emit export events from child thread to main thread"""
        self.export_event_signal.emit(event_type, data)


class ExportLogRedirector(QObject):
    """Thread-safe export log redirector"""

    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.log_stream = StringIO()

    def write(self, text):
        """Write text to log stream and emit signal if not empty"""
        if text.strip():
            self.log_signal.emit(text)

    def flush(self):
        """Flush the log stream"""
        pass


EXPORT_RESULT_PREFIX = "XANYLABELING_EXPORT_RESULT="

EXPORT_RUNTIME_PACKAGES: Dict[str, List[str]] = {
    "onnx": ["onnx>=1.12.0,<1.18.0", "onnxslim>=0.1.59", "onnxruntime"],
    "openvino": ["openvino>=2024.0.0"],
    "engine": ["tensorrt>7.0.0,!=10.1.0"],
    "coreml": ["coremltools>=8.0"],
    "saved_model": ["tensorflow>=2.0.0"],
    "pb": ["tensorflow>=2.0.0"],
    "tflite": ["tensorflow>=2.0.0"],
    "edgetpu": ["tensorflow>=2.0.0"],
    "tfjs": ["tensorflow>=2.0.0"],
    "paddle": ["paddlepaddle-gpu", "x2paddle"],
    "mnn": ["MNN>=2.9.6"],
    "ncnn": ["ncnn", "pnnx"],
    "imx": ["imx500-converter[pt]>=3.16.1", "mct-quantizers>=1.6.0"],
    "rknn": ["rknn-toolkit2"],
}


class ExportManager:
    def __init__(self):
        self.is_exporting = False
        self.callbacks = []
        self.export_thread = None

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Error in export callback: {e}")

    def _emit_export_log(self, message: str):
        self.notify_callbacks("export_log", {"message": message})

    def _run_export_subprocess(
        self, runtime_python: str, weights_path: str, export_format: str
    ) -> Tuple[bool, str]:
        script_content = f"""# -*- coding: utf-8 -*-
import io
import json
import os
import sys

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from ultralytics import YOLO

RESULT_PREFIX = {repr(EXPORT_RESULT_PREFIX)}
weights_path = {repr(weights_path)}
export_format = {repr(export_format)}

try:
    model = YOLO(weights_path)
    results = model.export(format=export_format)
    exported_path = results if isinstance(results, str) else str(results)
    print(RESULT_PREFIX + json.dumps({{"ok": True, "exported_path": exported_path}}, ensure_ascii=True), flush=True)
except Exception as e:
    print(RESULT_PREFIX + json.dumps({{"ok": False, "error": str(e)}}, ensure_ascii=True), flush=True)
    raise
"""

        script_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix="_export_script.py",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(script_content)
                script_path = f.name

            self._emit_export_log(
                f"Starting export subprocess with runtime: {runtime_python}"
            )
            process = subprocess.Popen(
                [runtime_python, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            payload = None
            assert process.stdout is not None
            for line in process.stdout:
                cleaned = line.rstrip()
                if not cleaned:
                    continue
                if cleaned.startswith(EXPORT_RESULT_PREFIX):
                    raw_json = cleaned[len(EXPORT_RESULT_PREFIX) :]
                    try:
                        payload = json.loads(raw_json)
                    except json.JSONDecodeError:
                        pass
                    continue
                self._emit_export_log(cleaned)

            return_code = process.wait()
            if payload is None:
                return (
                    False,
                    f"Export subprocess exited with code {return_code} without a result payload.",
                )
            if not payload.get("ok"):
                return False, payload.get("error", "Export failed")
            return True, str(payload.get("exported_path", ""))
        except Exception as e:
            return False, str(e)
        finally:
            if script_path:
                try:
                    os.remove(script_path)
                except Exception:
                    pass

    @staticmethod
    def _resolve_crnn_export_script_path() -> str:
        return (
            os.environ.get(CRNN_EXPORT_SCRIPT_ENV, "").strip()
            or DEFAULT_CRNN_EXPORT_SCRIPT
        )

    @staticmethod
    def _find_crnn_checkpoint(project_path: str) -> str:
        candidates = [
            os.path.join(project_path, "weights", "best_crnn_dynamic.pth"),
            os.path.join(project_path, "weights", "latest_crnn_dynamic.pth"),
            os.path.join(project_path, "best_crnn_dynamic.pth"),
            os.path.join(project_path, "latest_crnn_dynamic.pth"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    def _run_crnn_export_subprocess(
        self,
        runtime_python: str,
        project_path: str,
        checkpoint_path: str,
    ) -> Tuple[bool, str]:
        script_path = self._resolve_crnn_export_script_path()
        if not os.path.exists(script_path):
            return False, f"CRNN export script not found: {script_path}"

        weights_dir = os.path.join(project_path, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        torchscript_path = os.path.join(weights_dir, "crnn.pt")
        ncnn_param_path = os.path.join(weights_dir, "crnn.ncnn.param")
        ncnn_bin_path = os.path.join(weights_dir, "crnn.ncnn.bin")

        command = [
            runtime_python,
            script_path,
            "--checkpoint",
            checkpoint_path,
            "--torchscript-file",
            torchscript_path,
            "--ncnn-param-file",
            ncnn_param_path,
            "--ncnn-bin-file",
            ncnn_bin_path,
            "--pnnx",
            "pnnx",
        ]
        self._emit_export_log(f"[CRNN] $ {' '.join(command)}")

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
            return False, f"Failed to start CRNN export process: {e}"

        assert process.stdout is not None
        for line in process.stdout:
            cleaned = line.rstrip()
            if cleaned:
                self._emit_export_log(cleaned)

        return_code = process.wait()
        if return_code != 0:
            return False, f"CRNN export failed with exit code {return_code}"
        if not os.path.exists(ncnn_param_path):
            return False, f"CRNN export finished but output not found: {ncnn_param_path}"
        return True, ncnn_param_path

    def _resolve_exported_path(
        self, weights_path: str, export_format: str, exported_path: str
    ) -> str:
        weights_dir = os.path.dirname(weights_path)
        model_name = os.path.splitext(os.path.basename(weights_path))[0]
        candidates = []

        if exported_path:
            candidates.append(exported_path)
            if not os.path.isabs(exported_path):
                candidates.append(os.path.join(weights_dir, exported_path))

        candidates.append(weights_path.replace(".pt", f".{export_format}"))
        if export_format == "ncnn":
            candidates.append(os.path.join(weights_dir, f"{model_name}_ncnn_model"))

        for path in candidates:
            if path and os.path.exists(path):
                return path
        return exported_path

    def start_export(
        self,
        project_path: str,
        export_format: str = "onnx",
        task_type: str = "",
    ) -> Tuple[bool, str]:
        if self.is_exporting:
            if self.export_thread and self.export_thread.is_alive():
                return False, "Export already in progress"
            # Recover stale exporting state when thread is no longer running.
            self.is_exporting = False
            self.export_thread = None

        if self.export_thread and not self.export_thread.is_alive():
            self.export_thread = None

        if self.is_exporting:
            return False, "Export already in progress"

        normalized_task_type = (task_type or "").strip().lower()
        self.is_exporting = True
        if normalized_task_type == "crnn":
            checkpoint_path = self._find_crnn_checkpoint(project_path)
            if not checkpoint_path:
                self.is_exporting = False
                return (
                    False,
                    "CRNN checkpoint not found, expected best_crnn_dynamic.pth or latest_crnn_dynamic.pth.",
                )
            self.export_thread = threading.Thread(
                target=self._export_crnn_worker,
                args=(project_path, checkpoint_path, export_format),
            )
        else:
            weights_path = os.path.join(project_path, "weights", "best.pt")
            if not os.path.exists(weights_path):
                self.is_exporting = False
                return False, f"Model weights not found at: {weights_path}"
            self.export_thread = threading.Thread(
                target=self._export_worker,
                args=(project_path, weights_path, export_format),
            )
        self.export_thread.start()
        return True, "Export started successfully"

    def _export_crnn_worker(
        self, project_path: str, checkpoint_path: str, export_format: str
    ):
        normalized_format = (export_format or "").strip().lower() or "ncnn"
        if normalized_format not in {"ncnn"}:
            self.notify_callbacks(
                "export_error",
                {
                    "error": f"CRNN only supports NCNN export in this UI. Got: {export_format}"
                },
            )
            self.is_exporting = False
            self.export_thread = None
            return

        try:
            self.notify_callbacks(
                "export_started",
                {"weights_path": checkpoint_path, "format": "ncnn"},
            )
            self._emit_export_log("Checking export runtime environment...")
            runtime_ok, runtime_message, runtime_python = (
                ensure_ultralytics_runtime(
                    log_callback=self._emit_export_log,
                    extra_packages=["ncnn", "pnnx"],
                )
            )
            if not runtime_ok:
                self.notify_callbacks(
                    "export_error",
                    {"error": runtime_message},
                )
                return

            success, result = self._run_crnn_export_subprocess(
                runtime_python=runtime_python,
                project_path=project_path,
                checkpoint_path=checkpoint_path,
            )
            if not success:
                self.notify_callbacks(
                    "export_error", {"error": f"CRNN export failed: {result}"}
                )
                return

            self.notify_callbacks(
                "export_completed",
                {"exported_path": result, "format": "ncnn"},
            )
        except Exception as e:
            self.notify_callbacks(
                "export_error",
                {"error": f"Unexpected error during CRNN export: {str(e)}"},
            )
        finally:
            self.is_exporting = False
            self.export_thread = None

    def _export_worker(
        self, project_path: str, weights_path: str, export_format: str
    ):
        try:
            self.notify_callbacks(
                "export_started",
                {"weights_path": weights_path, "format": export_format},
            )
            self._emit_export_log("Checking export runtime environment...")
            runtime_ok, runtime_message, runtime_python = (
                ensure_ultralytics_runtime(
                    log_callback=self._emit_export_log,
                    extra_packages=EXPORT_RUNTIME_PACKAGES.get(
                        export_format, []
                    ),
                )
            )
            if not runtime_ok:
                self.notify_callbacks(
                    "export_error",
                    {"error": runtime_message},
                )
                return

            if export_format == "ncnn":
                custom_ok, custom_result = run_custom_ncnn_export(
                    runtime_python=runtime_python,
                    project_path=project_path,
                    weights_path=weights_path,
                    log_callback=self._emit_export_log,
                )
                if custom_ok:
                    self.notify_callbacks(
                        "export_completed",
                        {
                            "exported_path": custom_result,
                            "format": export_format,
                        },
                    )
                    return
                self._emit_export_log(
                    f"[NCNN] Custom exporter failed, fallback to default path: {custom_result}"
                )

            self._emit_export_log(f"Loading model from {weights_path}")
            self._emit_export_log(
                f"Starting export to {export_format} format..."
            )

            success, result = self._run_export_subprocess(
                runtime_python, weights_path, export_format
            )
            if not success:
                self.notify_callbacks(
                    "export_error", {"error": f"Export failed: {result}"}
                )
                return

            exported_path = self._resolve_exported_path(
                weights_path, export_format, result
            )
            if exported_path and os.path.exists(exported_path):
                self.notify_callbacks(
                    "export_completed",
                    {
                        "exported_path": exported_path,
                        "format": export_format,
                    },
                )
                return

            self.notify_callbacks(
                "export_error",
                {"error": "Export completed but output file not found"},
            )

        except Exception as e:
            self.notify_callbacks(
                "export_error",
                {"error": f"Unexpected error during export: {str(e)}"},
            )
        finally:
            self.is_exporting = False
            self.export_thread = None

    def stop_export(self) -> bool:
        if not self.is_exporting and not (
            self.export_thread and self.export_thread.is_alive()
        ):
            return False

        self.is_exporting = False
        if self.export_thread and self.export_thread.is_alive():
            self.export_thread.join(timeout=5)
        if self.export_thread and not self.export_thread.is_alive():
            self.export_thread = None

        self.notify_callbacks("export_stopped", {})
        return True


_export_manager = None


def get_export_manager() -> ExportManager:
    global _export_manager
    if _export_manager is None:
        _export_manager = ExportManager()
    return _export_manager


def export_model(
    project_path: str, export_format: str = "onnx", task_type: str = ""
) -> Tuple[bool, str]:
    manager = get_export_manager()
    return manager.start_export(project_path, export_format, task_type)
