import os
import signal
import shutil
import subprocess
import time
import threading
from pathlib import Path
from io import StringIO
from typing import Dict, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from .config import get_settings_config_path
from .runtime_env import ensure_ultralytics_runtime

CRNN_TRAIN_SCRIPT_ENV = "XANYLABELING_CRNN_TRAIN_SCRIPT"
DEFAULT_CRNN_TRAIN_SCRIPT = str(
    Path(__file__).resolve().parents[1] / "crnn" / "train_dynamic.py"
)


class TrainingEventRedirector(QObject):
    """Thread-safe training event redirector"""

    training_event_signal = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

    def emit_training_event(self, event_type, data):
        """Safely emit training events from child thread to main thread"""
        self.training_event_signal.emit(event_type, data)


class TrainingLogRedirector(QObject):
    """Thread-safe training log redirector"""

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


class TrainingManager:
    def __init__(self):
        self.training_process = None
        self.is_training = False
        self.callbacks = []
        self.total_epochs = 100
        self.stop_event = threading.Event()

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception:
                pass

    def _emit_training_log(self, message: str):
        self.notify_callbacks("training_log", {"message": message})

    @staticmethod
    def _resolve_torch_backend_from_device(device_value) -> str:
        if isinstance(device_value, str) and device_value.lower() == "cpu":
            return "cpu"
        return "auto"

    @staticmethod
    def _resolve_crnn_train_script_path() -> str:
        return (
            os.environ.get(CRNN_TRAIN_SCRIPT_ENV, "").strip()
            or DEFAULT_CRNN_TRAIN_SCRIPT
        )

    @staticmethod
    def _normalize_task_type(task_type: str, train_args: Dict) -> str:
        direct = (task_type or "").strip().lower()
        if direct:
            return direct
        return str(train_args.get("__task_type__", "")).strip().lower()

    @staticmethod
    def _to_device_string(device_value) -> str:
        if isinstance(device_value, list):
            if not device_value:
                return "cpu"
            return ",".join(str(v) for v in device_value)
        if device_value is None:
            return "auto"
        return str(device_value)

    def _build_yolo_train_script(
        self, train_args: Dict, project_dir: str
    ) -> Tuple[str, str]:
        model_path = train_args.pop("model")
        script_content = f"""# -*- coding: utf-8 -*-
import io
import os
import signal
import sys
import multiprocessing

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO

def signal_handler(signum, frame):
    print("Training interrupted by signal", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        multiprocessing.set_start_method('spawn', force=True)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        model = YOLO({repr(model_path)})
        train_args = {repr(train_args)}
        train_args['verbose'] = False
        train_args['show'] = False
        _ = model.train(**train_args)
    except KeyboardInterrupt:
        print("Training interrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"Training error: {{e}}", flush=True)
        sys.exit(1)
"""

        script_path = os.path.join(project_dir, "train_script.py")
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        return script_path, "yolo"

    def _build_crnn_train_command(
        self, train_args: Dict
    ) -> Tuple[Tuple[str, ...], str]:
        script_path = self._resolve_crnn_train_script_path()
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"CRNN training script not found: {script_path}"
            )

        project = train_args.get("project", "")
        name = train_args.get("name", "exp")
        project_path = os.path.join(project, name)
        weights_dir = os.path.join(project_path, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        data_root = train_args.get("data_root", "")
        labels_file = train_args.get("labels_file", "")
        if not data_root or not labels_file:
            raise ValueError("CRNN training requires data_root and labels_file")

        charset_file = train_args.get(
            "charset_file", os.path.join(project_path, "charset.txt")
        )
        best_model = train_args.get(
            "best_model", os.path.join(weights_dir, "best_crnn_dynamic.pth")
        )
        latest_model = train_args.get(
            "latest_model", os.path.join(weights_dir, "latest_crnn_dynamic.pth")
        )

        command = [
            "__RUNTIME_PYTHON__",
            script_path,
            "--data-root",
            str(data_root),
            "--labels-file",
            str(labels_file),
            "--charset-file",
            str(charset_file),
            "--best-model",
            str(best_model),
            "--latest-model",
            str(latest_model),
            "--epochs",
            str(int(train_args.get("epochs", 50))),
            "--batch-size",
            str(int(train_args.get("batch_size", train_args.get("batch", 32)))),
            "--train-ratio",
            str(float(train_args.get("train_ratio", 0.9))),
            "--num-workers",
            str(int(train_args.get("num_workers", train_args.get("workers", 0)))),
            "--img-h",
            str(int(train_args.get("img_h", 32))),
            "--lr",
            str(float(train_args.get("lr", 0.001))),
            "--weight-decay",
            str(float(train_args.get("weight_decay", 1e-4))),
            "--device",
            self._to_device_string(train_args.get("device", "auto")),
            "--skip-export",
        ]

        resume = str(train_args.get("resume", "")).strip()
        if resume and os.path.exists(resume):
            command.extend(["--resume", resume])

        return tuple(command), "crnn"

    def start_training(
        self, train_args: Dict, task_type: str = ""
    ) -> Tuple[bool, str]:
        if self.is_training:
            return False, "Training is already in progress"

        try:
            runtime_train_args = dict(train_args or {})
            normalized_task_type = self._normalize_task_type(
                task_type, runtime_train_args
            )
            runtime_train_args.pop("__task_type__", None)

            self.total_epochs = runtime_train_args.get("epochs", 100)
            self.stop_event.clear()
            self.is_training = True

            project_dir = runtime_train_args.get("project", "/tmp")
            is_crnn_task = normalized_task_type == "crnn"
            script_path = ""
            process_command: Tuple[str, ...] = tuple()
            script_owner = ""
            if is_crnn_task:
                process_command, script_owner = self._build_crnn_train_command(
                    train_args=runtime_train_args
                )
            else:
                script_path, script_owner = self._build_yolo_train_script(
                    train_args=runtime_train_args,
                    project_dir=project_dir,
                )

            def run_training():
                try:
                    torch_backend = self._resolve_torch_backend_from_device(
                        runtime_train_args.get("device")
                    )
                    runtime_ok, runtime_message, runtime_python = (
                        ensure_ultralytics_runtime(
                            log_callback=self._emit_training_log,
                            torch_backend=torch_backend,
                        )
                    )
                    if not runtime_ok:
                        self.is_training = False
                        self.notify_callbacks(
                            "training_error", {"error": runtime_message}
                        )
                        return
                    if self.stop_event.is_set():
                        self.is_training = False
                        self.notify_callbacks("training_stopped", {})
                        return

                    self.notify_callbacks(
                        "training_started", {"total_epochs": self.total_epochs}
                    )

                    if is_crnn_task:
                        runtime_process_command = list(process_command)
                        runtime_process_command[0] = runtime_python
                    else:
                        runtime_process_command = [runtime_python, script_path]

                    self.training_process = subprocess.Popen(
                        runtime_process_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        bufsize=1,
                        preexec_fn=os.setsid if os.name != "nt" else None,
                    )

                    while True:
                        if self.stop_event.is_set():
                            self.training_process.terminate()
                            try:
                                self.training_process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                if os.name == "nt":
                                    self.training_process.kill()
                                else:
                                    os.killpg(
                                        os.getpgid(self.training_process.pid),
                                        signal.SIGKILL,
                                    )
                            self.is_training = False
                            self.notify_callbacks("training_stopped", {})
                            return

                        output = self.training_process.stdout.readline()
                        if (
                            output == ""
                            and self.training_process.poll() is not None
                        ):
                            break
                        if output:
                            cleaned_output = output.strip()
                            if cleaned_output:
                                self.notify_callbacks(
                                    "training_log", {"message": cleaned_output}
                                )

                    return_code = self.training_process.poll()
                    self.is_training = False

                    if return_code == 0:
                        self.notify_callbacks(
                            "training_completed",
                            {"results": "Training completed successfully"},
                        )
                    else:
                        self.notify_callbacks(
                            "training_error",
                            {
                                "error": f"Training process exited with code {return_code}"
                            },
                        )

                except Exception as e:
                    self.is_training = False
                    self.notify_callbacks("training_error", {"error": str(e)})
                finally:
                    try:
                        if script_owner == "yolo" and script_path:
                            os.remove(script_path)
                    except Exception:
                        pass

            def save_settings_config():
                save_path = os.path.join(
                    runtime_train_args["project"], runtime_train_args["name"]
                )
                save_file = os.path.join(save_path, "settings.json")

                while not os.path.exists(save_path):
                    time.sleep(1)

                shutil.copy2(get_settings_config_path(), save_file)

            training_thread = threading.Thread(target=run_training)
            training_thread.daemon = True
            training_thread.start()

            config_thread = threading.Thread(target=save_settings_config)
            config_thread.daemon = True
            config_thread.start()

            return True, "Training started successfully"

        except Exception as e:
            self.is_training = False
            return False, f"Failed to start training: {str(e)}"

    def stop_training(self) -> bool:
        if not self.is_training:
            return False

        try:
            self.stop_event.set()
            return True
        except Exception:
            return False


_training_manager = TrainingManager()


def get_training_manager() -> TrainingManager:
    return _training_manager
