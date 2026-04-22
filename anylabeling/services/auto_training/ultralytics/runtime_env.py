import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from .config import get_trainer_root_dir

LogCallback = Callable[[str], None]

RUNTIME_ROOT_DIRNAME = "runtime_env"
RUNTIME_VENV_DIRNAME = "venv"
RUNTIME_STATE_FILENAME = "runtime_state.json"
DEFAULT_PYPI_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
ENV_PYPI_INDEX_URL = "XANYLABELING_PYPI_INDEX_URL"
ENV_PYPI_EXTRA_INDEX_URL = "XANYLABELING_PYPI_EXTRA_INDEX_URL"
ENV_PYPI_TRUSTED_HOST = "XANYLABELING_PYPI_TRUSTED_HOST"
ENV_DISABLE_PYPI_MIRROR = "XANYLABELING_DISABLE_PYPI_MIRROR"
ENV_TORCH_BACKEND_FALLBACKS = "XANYLABELING_TORCH_BACKEND_FALLBACKS"
DEFAULT_TORCH_BACKEND_FALLBACKS = ["cu126", "cu121", "cu118"]

_VALIDATION_SNIPPET = """
import json
import torch
import ultralytics

info = {
    "ultralytics_version": getattr(ultralytics, "__version__", "unknown"),
    "torch_version": getattr(torch, "__version__", "unknown"),
    "cuda_available": bool(torch.cuda.is_available()),
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "gpu_compat_ok": True,
    "gpu_compat_error": "",
    "gpu_capability": None,
    "supported_arches": [],
}

if info["cuda_available"]:
    capability = None
    try:
        capability = torch.cuda.get_device_capability(0)
        info["gpu_capability"] = f"sm_{capability[0]}{capability[1]}"
    except Exception as e:
        info["gpu_compat_ok"] = False
        info["gpu_compat_error"] = f"get_device_capability failed: {e}"

    try:
        arch_list = torch.cuda.get_arch_list()
        info["supported_arches"] = arch_list if isinstance(arch_list, list) else []
    except Exception:
        info["supported_arches"] = []

    if info["gpu_compat_ok"] and info["gpu_capability"] and info["supported_arches"]:
        if info["gpu_capability"] not in info["supported_arches"]:
            info["gpu_compat_ok"] = False
            info["gpu_compat_error"] = (
                f"GPU arch {info['gpu_capability']} not in torch arches "
                f"{','.join(info['supported_arches'])}"
            )

    if info["gpu_compat_ok"]:
        try:
            x = torch.randn((16,), device="cuda")
            y = x * 2
            _ = y.sum().item()
            torch.cuda.synchronize()
        except Exception as e:
            info["gpu_compat_ok"] = False
            info["gpu_compat_error"] = str(e)

print(json.dumps(info, ensure_ascii=True))
""".strip()


def _emit_log(log_callback: Optional[LogCallback], message: str) -> None:
    if log_callback:
        log_callback(message)


def _get_runtime_root_dir() -> str:
    return os.path.join(get_trainer_root_dir(), RUNTIME_ROOT_DIRNAME)


def _get_runtime_venv_dir() -> str:
    return os.path.join(_get_runtime_root_dir(), RUNTIME_VENV_DIRNAME)


def _get_runtime_state_path() -> str:
    return os.path.join(_get_runtime_root_dir(), RUNTIME_STATE_FILENAME)


def _get_venv_python_path(venv_dir: str) -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _candidate_python_commands() -> List[List[str]]:
    commands: List[List[str]] = []
    if sys.executable:
        commands.append([sys.executable])
    if os.name == "nt":
        commands.append(["py", "-3"])
    commands.append(["python"])
    commands.append(["python3"])
    return commands


def _resolve_host_python() -> Optional[List[str]]:
    seen = set()
    for cmd in _candidate_python_commands():
        key = tuple(cmd)
        if key in seen:
            continue
        seen.add(key)
        try:
            result = subprocess.run(
                cmd + ["-c", "import sys; print(sys.version_info[0])"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip() == "3":
                return cmd
        except Exception:
            continue
    return None


def _run_command(
    command: List[str],
    log_callback: Optional[LogCallback],
    description: str,
) -> Tuple[bool, str]:
    _emit_log(log_callback, f"[Runtime] {description}")
    _emit_log(log_callback, f"[Runtime] $ {' '.join(command)}")

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
        return False, f"{description} failed to start: {e}"

    assert process.stdout is not None
    for line in process.stdout:
        cleaned = line.rstrip()
        if cleaned:
            _emit_log(log_callback, cleaned)

    return_code = process.wait()
    if return_code != 0:
        return False, f"{description} failed with exit code {return_code}"
    return True, ""


def _bool_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_index_settings() -> Dict[str, str]:
    if _bool_env(ENV_DISABLE_PYPI_MIRROR):
        return {}

    index_url = os.environ.get(ENV_PYPI_INDEX_URL, "").strip()
    if not index_url:
        index_url = DEFAULT_PYPI_INDEX_URL

    settings: Dict[str, str] = {}
    if index_url:
        settings["index_url"] = index_url

    extra_index_url = os.environ.get(ENV_PYPI_EXTRA_INDEX_URL, "").strip()
    if extra_index_url:
        settings["extra_index_url"] = extra_index_url

    trusted_host = os.environ.get(ENV_PYPI_TRUSTED_HOST, "").strip()
    if trusted_host:
        settings["trusted_host"] = trusted_host

    return settings


def _build_index_args(settings: Dict[str, str]) -> List[str]:
    args: List[str] = []
    if settings.get("index_url"):
        args.extend(["--index-url", settings["index_url"]])
    if settings.get("extra_index_url"):
        args.extend(["--extra-index-url", settings["extra_index_url"]])
    if settings.get("trusted_host"):
        args.extend(["--trusted-host", settings["trusted_host"]])
    return args


def _run_install_command(
    base_command: List[str],
    index_settings: Dict[str, str],
    log_callback: Optional[LogCallback],
    description: str,
) -> Tuple[bool, str]:
    index_args = _build_index_args(index_settings)
    if not index_args:
        return _run_command(base_command, log_callback, description)

    ok, error_message = _run_command(
        base_command + index_args,
        log_callback,
        f"{description} (mirror)",
    )
    if ok:
        return True, ""

    _emit_log(
        log_callback,
        f"[Runtime] Mirror install failed, retrying official source: {error_message}",
    )
    return _run_command(base_command, log_callback, f"{description} (official)")


def _parse_torch_backend_fallbacks() -> List[str]:
    raw = os.environ.get(ENV_TORCH_BACKEND_FALLBACKS, "").strip()
    if not raw:
        return list(DEFAULT_TORCH_BACKEND_FALLBACKS)

    parsed: List[str] = []
    for token in raw.split(","):
        item = token.strip().lower()
        if item and item not in parsed:
            parsed.append(item)
    return parsed


def _detect_compute_capability() -> Optional[Tuple[int, int]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        parts = value.split(".", 1)
        if len(parts) != 2:
            continue
        if not parts[0].isdigit() or not parts[1].isdigit():
            continue
        return int(parts[0]), int(parts[1])
    return None


def _build_torch_backend_candidates(requested_backend: str) -> List[str]:
    backend = (requested_backend or "auto").strip().lower() or "auto"
    if backend == "cpu":
        return ["cpu"]
    if backend != "auto":
        return [backend]

    fallback_backends = _parse_torch_backend_fallbacks()
    capability = _detect_compute_capability()
    if capability is not None:
        major, minor = capability
        if major < 7:
            preferred = fallback_backends + ["auto"]
            return list(dict.fromkeys(preferred))

    return list(dict.fromkeys(["auto"] + fallback_backends))


def _is_runtime_gpu_compatible(runtime_info: Dict) -> bool:
    if not bool(runtime_info.get("cuda_available")):
        return False
    return bool(runtime_info.get("gpu_compat_ok", True))


def _is_runtime_compatible_for_backend(runtime_info: Dict, backend: str) -> bool:
    normalized = (backend or "auto").strip().lower() or "auto"
    if normalized == "cpu":
        return True
    return _is_runtime_gpu_compatible(runtime_info)


def _recreate_runtime_venv(log_callback: Optional[LogCallback]) -> Tuple[bool, str]:
    runtime_venv_dir = _get_runtime_venv_dir()
    if os.path.exists(runtime_venv_dir):
        _emit_log(
            log_callback,
            "[Runtime] Recreating runtime virtual environment for backend retry",
        )
        try:
            shutil.rmtree(runtime_venv_dir)
        except Exception as e:
            return False, f"Failed to remove runtime venv: {e}"
    return _ensure_venv(log_callback)


def _probe_runtime(runtime_python: str) -> Tuple[bool, Dict, str]:
    try:
        result = subprocess.run(
            [runtime_python, "-c", _VALIDATION_SNIPPET],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except Exception as e:
        return False, {}, str(e)

    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip()
        return False, {}, error_text or "runtime check failed"

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return True, json.loads(line), ""
        except json.JSONDecodeError:
            continue
    return False, {}, "failed to parse runtime validation output"


def _write_runtime_state(state: Dict) -> None:
    state_path = _get_runtime_state_path()
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=True, indent=2)


def _format_runtime_summary(prefix: str, info: Dict) -> str:
    ultralytics_version = info.get("ultralytics_version", "unknown")
    torch_version = info.get("torch_version", "unknown")
    cuda_available = bool(info.get("cuda_available"))
    cuda_version = info.get("torch_cuda_version") or "none"
    gpu_capability = info.get("gpu_capability") or "unknown"
    gpu_compat_ok = bool(info.get("gpu_compat_ok", True))
    device_label = "cuda" if cuda_available else "cpu"
    base = (
        f"{prefix}: ultralytics={ultralytics_version}, "
        f"torch={torch_version}, torch_cuda={cuda_version}, device={device_label}, "
        f"gpu_capability={gpu_capability}, gpu_compat={gpu_compat_ok}"
    )
    if not gpu_compat_ok:
        gpu_compat_error = info.get("gpu_compat_error") or "unknown"
        return f"{base}, gpu_error={gpu_compat_error}"
    return base


def _ensure_venv(log_callback: Optional[LogCallback]) -> Tuple[bool, str]:
    runtime_venv_dir = _get_runtime_venv_dir()
    runtime_python = _get_venv_python_path(runtime_venv_dir)
    if os.path.exists(runtime_python):
        return True, ""

    os.makedirs(os.path.dirname(runtime_venv_dir), exist_ok=True)

    host_python = _resolve_host_python()
    if not host_python:
        return (
            False,
            "No Python 3 interpreter found to create training runtime venv.",
        )

    return _run_command(
        host_python + ["-m", "venv", runtime_venv_dir],
        log_callback,
        "Creating training runtime virtual environment",
    )


def _ensure_uv(
    runtime_python: str,
    log_callback: Optional[LogCallback],
    index_settings: Dict[str, str],
) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            [runtime_python, "-m", "uv", "--version"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            return True, ""
    except Exception:
        pass

    return _run_install_command(
        [runtime_python, "-m", "pip", "install", "uv"],
        index_settings,
        log_callback,
        "Installing uv in training runtime",
    )


def _install_ultralytics(
    runtime_python: str,
    log_callback: Optional[LogCallback],
    index_settings: Dict[str, str],
    torch_backend: str = "auto",
) -> Tuple[bool, str]:
    backend = (torch_backend or "auto").strip().lower()
    if not backend:
        backend = "auto"

    return _run_install_command(
        [
            runtime_python,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            runtime_python,
            f"--torch-backend={backend}",
            "ultralytics",
        ],
        index_settings,
        log_callback,
        f"Installing ultralytics with torch backend: {backend}",
    )


def _install_additional_packages(
    runtime_python: str,
    packages: List[str],
    log_callback: Optional[LogCallback],
    index_settings: Dict[str, str],
) -> Tuple[bool, str]:
    if not packages:
        return True, ""

    return _run_install_command(
        [
            runtime_python,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            runtime_python,
            *packages,
        ],
        index_settings,
        log_callback,
        "Installing additional runtime packages",
    )


def ensure_ultralytics_runtime(
    log_callback: Optional[LogCallback] = None,
    extra_packages: Optional[List[str]] = None,
    torch_backend: str = "auto",
) -> Tuple[bool, str, str]:
    os.makedirs(_get_runtime_root_dir(), exist_ok=True)
    unique_extra_packages = (
        list(dict.fromkeys(pkg for pkg in (extra_packages or []) if pkg))
        if extra_packages
        else []
    )

    ok, error_message = _ensure_venv(log_callback)
    if not ok:
        return False, error_message, ""

    index_settings = _resolve_index_settings()
    backend_candidates = _build_torch_backend_candidates(torch_backend)
    _emit_log(
        log_callback,
        f"[Runtime] Torch backend candidates: {', '.join(backend_candidates)}",
    )
    if index_settings.get("index_url"):
        _emit_log(
            log_callback,
            f"[Runtime] Package index: {index_settings['index_url']}",
        )
    else:
        _emit_log(log_callback, "[Runtime] Package index: official default")

    runtime_python = _get_venv_python_path(_get_runtime_venv_dir())
    is_ready, info, probe_error = _probe_runtime(runtime_python)
    runtime_compatible = (
        is_ready and _is_runtime_compatible_for_backend(info, torch_backend)
    )
    if runtime_compatible:
        if unique_extra_packages:
            ok, error_message = _ensure_uv(
                runtime_python, log_callback, index_settings
            )
            if not ok:
                return False, error_message, ""
            ok, error_message = _install_additional_packages(
                runtime_python,
                unique_extra_packages,
                log_callback,
                index_settings,
            )
            if not ok:
                return False, error_message, ""

        summary = _format_runtime_summary("[Runtime] Reusing training runtime", info)
        _emit_log(log_callback, summary)
        _write_runtime_state(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "python": runtime_python,
                "runtime_info": info,
            }
        )
        return True, summary, runtime_python

    if probe_error:
        _emit_log(log_callback, f"[Runtime] Existing runtime check failed: {probe_error}")
    elif is_ready and not runtime_compatible:
        _emit_log(
            log_callback,
            "[Runtime] Existing runtime is incompatible with requested GPU backend, retrying with fallbacks",
        )

    last_error = "Unknown runtime installation error"
    for attempt, backend in enumerate(backend_candidates):
        if attempt > 0 or (is_ready and not runtime_compatible):
            ok, error_message = _recreate_runtime_venv(log_callback)
            if not ok:
                last_error = error_message
                continue

        runtime_python = _get_venv_python_path(_get_runtime_venv_dir())
        ok, error_message = _ensure_uv(runtime_python, log_callback, index_settings)
        if not ok:
            last_error = error_message
            continue

        ok, error_message = _install_ultralytics(
            runtime_python,
            log_callback,
            index_settings,
            torch_backend=backend,
        )
        if not ok:
            last_error = error_message
            continue

        if unique_extra_packages:
            ok, error_message = _install_additional_packages(
                runtime_python,
                unique_extra_packages,
                log_callback,
                index_settings,
            )
            if not ok:
                last_error = error_message
                continue

        is_ready, info, probe_error = _probe_runtime(runtime_python)
        if not is_ready:
            last_error = (
                f"Training runtime validation failed after installation: {probe_error}"
            )
            continue

        if not _is_runtime_compatible_for_backend(info, torch_backend):
            gpu_error = info.get("gpu_compat_error") or "unknown"
            last_error = (
                f"GPU runtime incompatibility with backend {backend}: {gpu_error}"
            )
            _emit_log(log_callback, f"[Runtime] {last_error}")
            continue

        summary = _format_runtime_summary("[Runtime] Training runtime prepared", info)
        _emit_log(log_callback, summary)
        _write_runtime_state(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "python": runtime_python,
                "runtime_info": info,
                "selected_backend": backend,
            }
        )
        return True, summary, runtime_python

    return (
        False,
        f"Failed to prepare compatible training runtime after trying backends {backend_candidates}: {last_error}",
        "",
    )
