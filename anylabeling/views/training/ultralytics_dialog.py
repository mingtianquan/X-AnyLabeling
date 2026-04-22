import csv
import copy
import datetime
import glob
import json
import os
import platform
import re
import shutil
import subprocess

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QPushButton,
    QLabel,
    QMessageBox,
    QScrollArea,
    QGroupBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QProgressBar,
    QTextEdit,
    QApplication,
    QSizePolicy,
)

from anylabeling.config import get_config
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.training.widgets.ultralytics_widgets import *
from anylabeling.services.auto_training.ultralytics._io import *
from anylabeling.services.auto_training.ultralytics.config import *
from anylabeling.services.auto_training.ultralytics.exporter import (
    ExportEventRedirector,
    ExportLogRedirector,
    get_export_manager,
)
from anylabeling.services.auto_training.ultralytics.general import (
    create_yolo_dataset,
    format_classes_display,
    parse_string_to_digit_list,
)
from anylabeling.services.auto_training.ultralytics.style import *
from anylabeling.services.auto_training.ultralytics.trainer import (
    TrainingEventRedirector,
    TrainingLogRedirector,
    get_training_manager,
)
from anylabeling.services.auto_training.ultralytics.utils import *
from anylabeling.services.auto_training.ultralytics.validators import (
    validate_basic_config,
    validate_data_file,
)


class CRNNPrepareThread(QThread):
    prepare_log = pyqtSignal(str)
    prepare_success = pyqtSignal(object)
    prepare_failed = pyqtSignal(str)

    def __init__(self, prepare_fn, config_snapshot, parent=None):
        super().__init__(parent)
        self._prepare_fn = prepare_fn
        self._config_snapshot = config_snapshot

    def run(self):
        try:
            train_args = self._prepare_fn(
                self._config_snapshot,
                log_callback=self.prepare_log.emit,
            )
            self.prepare_success.emit(train_args)
        except Exception as e:
            self.prepare_failed.emit(str(e))


class DatasetSummaryThread(QThread):
    summary_completed = pyqtSignal(int, str, object)
    summary_failed = pyqtSignal(int, str, str)

    def __init__(
        self,
        request_id,
        task_type,
        image_list,
        supported_shape,
        output_dir=None,
        parent=None,
    ):
        super().__init__(parent)
        self._request_id = int(request_id)
        self._task_type = str(task_type or "")
        self._image_list = list(image_list or [])
        self._supported_shape = list(supported_shape or [])
        self._output_dir = output_dir

    def run(self):
        try:
            if self._task_type == "Classify":
                table_data = self._compute_classification_data()
            elif self._task_type == "CRNN":
                table_data = self._compute_crnn_data()
            else:
                table_data = get_statistics_table_data(
                    self._image_list,
                    self._supported_shape,
                    self._output_dir,
                )

            if self.isInterruptionRequested():
                return
            self.summary_completed.emit(
                self._request_id, self._task_type, table_data
            )
        except Exception as e:
            self.summary_failed.emit(
                self._request_id, self._task_type, str(e)
            )

    def _compute_classification_data(self):
        headers = ["Label"] + self._supported_shape + ["Total"]
        classify_shapes = TASK_SHAPE_MAPPINGS.get("Classify", ["flags"])
        label_infos = get_label_infos(
            self._image_list, classify_shapes, self._output_dir
        )
        if not label_infos:
            return [headers]

        table_data = [headers]
        total_counts = [0] * len(self._supported_shape)
        total_images = 0

        for label, infos in sorted(label_infos.items()):
            shape_counts = [0] * len(self._supported_shape)
            image_count = infos.get("_total", 0)
            total_images += image_count
            row = [label] + [str(c) for c in shape_counts] + [str(image_count)]
            table_data.append(row)

        total_row = (
            ["Total"] + [str(c) for c in total_counts] + [str(total_images)]
        )
        table_data.append(total_row)
        return table_data

    def _compute_crnn_data(self):
        from anylabeling.views.labeling.label_converter import LabelConverter

        headers = ["Label", "Total"]
        converter = LabelConverter()
        label_counter = {}
        total_count = 0

        for image_file in self._image_list:
            if self.isInterruptionRequested():
                return [headers]

            label_dir, filename = os.path.split(image_file)
            if self._output_dir:
                label_dir = self._output_dir
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )
            label_text = converter.custom_to_crnn(
                image_file=image_file,
                label_file=label_file,
                split_char="_",
                fallback_from_filename=True,
            )
            if not label_text:
                continue
            label_counter[label_text] = label_counter.get(label_text, 0) + 1
            total_count += 1

        table_data = [headers]
        for label in sorted(label_counter.keys()):
            table_data.append([label, str(label_counter[label])])
        table_data.append(["Total", str(total_count)])
        return table_data


class UltralyticsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(*DEFAULT_WINDOW_SIZE)

        self.image_list = parent.image_list
        self.output_dir = parent.output_dir
        self.supported_shape = parent.supported_shape
        self.selected_task_type = None
        self.config_widgets = {}
        self._classification_cache = None
        self._detection_cache = None
        self._crnn_cache = None
        self.task_type_buttons = {}
        self.names = []

        # Training related attributes
        self.log_redirector = TrainingLogRedirector()
        self.log_redirector.log_signal.connect(
            self.append_training_log, Qt.ConnectionType.QueuedConnection
        )
        self.event_redirector = TrainingEventRedirector()
        self.event_redirector.training_event_signal.connect(
            self.on_training_event, Qt.ConnectionType.QueuedConnection
        )
        self.training_manager = get_training_manager()
        self.training_manager.callbacks = [
            self.event_redirector.emit_training_event
        ]

        # Export related attributes
        self.export_log_redirector = ExportLogRedirector()
        self.export_log_redirector.log_signal.connect(
            self.append_training_log, Qt.ConnectionType.QueuedConnection
        )
        self.export_event_redirector = ExportEventRedirector()
        self.export_event_redirector.export_event_signal.connect(
            self.on_export_event, Qt.ConnectionType.QueuedConnection
        )
        self.export_manager = get_export_manager()
        self.export_manager.callbacks = [
            self.export_event_redirector.emit_export_event
        ]

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_training_progress)
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_training_images)
        self.crnn_prepare_thread = None
        self.dataset_summary_thread = None
        self._summary_request_id = 0
        self.current_project_path = None
        self.training_status = "idle"  # idle, training, completed, error
        self.current_epochs = 0
        self.total_epochs = 0

        app_config = get_config()
        self.project_readonly = (
            app_config.get("training", {})
            .get("ultralytics", {})
            .get("project_readonly", True)
        )

        self.init_ui()
        self.setStyleSheet(get_ultralytics_dialog_style())
        self.refresh_dataset_summary()

    def init_ui(self):
        self.data_tab = QWidget()
        self.config_tab = QWidget()
        self.train_tab = QWidget()

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.data_tab, self.tr("Data"))
        self.tab_widget.addTab(self.config_tab, self.tr("Config"))
        self.tab_widget.addTab(self.train_tab, self.tr("Train"))
        self.tab_widget.tabBar().setEnabled(False)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)

        self.init_data_tab()
        self.init_config_tab()
        self.init_train_tab()

    def save_training_logs_to_file(self):
        """Save training logs to a local file with timestamp"""
        if (
            not hasattr(self, "log_display")
            or not self.log_display.toPlainText().strip()
        ):
            return

        if not os.path.exists(self.current_project_path):
            return
        log_dir_path = os.path.join(self.current_project_path, "logs")
        os.makedirs(log_dir_path, exist_ok=True)

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{self.training_status}_{timestamp}.txt"
            log_file_path = os.path.join(log_dir_path, filename)

            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(self.log_display.toPlainText())

            logger.info(f"Training logs saved to: {log_file_path}")

        except Exception as e:
            logger.error(f"Failed to save training logs: {str(e)}")

    def go_to_specific_tab(self, index):
        """Go to specific tab by index"""
        self.tab_widget.setCurrentIndex(index)

    # Data Tab
    def show_pose_config(self):
        """Show the pose config field"""
        if hasattr(self, "pose_config_label"):
            self.pose_config_label.setVisible(True)
            self.config_widgets["pose_config"].setVisible(True)

            for i in range(self.pose_config_layout.count()):
                widget = self.pose_config_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)

    def hide_pose_config(self):
        """Hide the pose config field"""
        if hasattr(self, "pose_config_label"):
            self.pose_config_label.setVisible(False)
            self.config_widgets["pose_config"].setVisible(False)

            for i in range(self.pose_config_layout.count()):
                widget = self.pose_config_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)

    def on_task_type_selected(self, task_type):
        normalized_task_type = None
        for task in TASK_TYPES:
            if task.lower() == task_type.lower():
                normalized_task_type = task
                break

        if normalized_task_type is None:
            logger.warning(f"Unknown task type: {task_type}")
            return

        task_type = normalized_task_type

        if task_type not in self.task_type_buttons:
            logger.warning(f"Task type button not found: {task_type}")
            return

        if self.selected_task_type == task_type:
            self.selected_task_type = None
            self.task_type_buttons[task_type].set_selected(False)
            self.hide_pose_config()
        else:
            if self.selected_task_type:
                self.task_type_buttons[self.selected_task_type].set_selected(
                    False
                )
            self.selected_task_type = task_type
            self.task_type_buttons[task_type].set_selected(True)

            if task_type.lower() == "pose":
                self.show_pose_config()
            else:
                self.hide_pose_config()

        self._apply_task_type_ui_hints()
        self.refresh_dataset_summary()

    def create_task_handler(self, task_type):
        def handler():
            self.on_task_type_selected(task_type)

        return handler

    def _apply_task_type_ui_hints(self):
        model_widget = self.config_widgets.get("model")
        data_widget = self.config_widgets.get("data")
        imgsz_widget = self.config_widgets.get("imgsz")

        if self.selected_task_type == "CRNN":
            if model_widget is not None:
                model_widget.setPlaceholderText(
                    self.tr(
                        "Optional resume checkpoint (.pth). Leave empty to train from scratch."
                    )
                )
            if data_widget is not None:
                data_widget.setPlaceholderText(
                    self.tr(
                        "Optional dataset root containing labels.txt; leave empty to auto-build from current annotations"
                    )
                )
            if imgsz_widget is not None and imgsz_widget.value() == 640:
                imgsz_widget.setValue(32)
        else:
            if model_widget is not None:
                model_widget.setPlaceholderText("")
            if data_widget is not None:
                data_widget.setPlaceholderText("")
            if (
                imgsz_widget is not None
                and imgsz_widget.value() == 32
                and self.selected_task_type in {"Detect", "OBB", "Segment", "Pose"}
            ):
                imgsz_widget.setValue(640)

    def init_task_configuration(self, parent_layout):
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        task_type_layout = QHBoxLayout()
        task_type_layout.addWidget(QLabel(self.tr("Task Type:")))
        for task_type in TASK_TYPES:
            button = CustomQPushButton(task_type)
            button.clicked.connect(self.create_task_handler(task_type))
            task_type_layout.addWidget(button)
            self.task_type_buttons[task_type] = button

        task_type_layout.addStretch()
        config_layout.addLayout(task_type_layout)
        parent_layout.addWidget(config_widget)

    def refresh_dataset_summary(self):
        self._stop_dataset_summary_thread_if_running()

        if not self.image_list:
            self.summary_table.clear()
            return

        if self.selected_task_type is None:
            table_data = [
                ["Metric", "Value"],
                ["Total Images", str(len(self.image_list))],
                [
                    "Task",
                    self.tr("Select task type to load detailed statistics"),
                ],
            ]
            self.summary_table.load_data(table_data)
            return

        if self.selected_task_type == "Classify" and self._classification_cache:
            self.summary_table.load_data(self._classification_cache)
            return
        if self.selected_task_type == "CRNN" and self._crnn_cache:
            self.summary_table.load_data(self._crnn_cache)
            return
        if (
            self.selected_task_type not in {"Classify", "CRNN"}
            and self._detection_cache
        ):
            self.summary_table.load_data(self._detection_cache)
            return

        self._summary_request_id += 1
        request_id = self._summary_request_id

        self.summary_table.load_data(
            [
                ["Metric", "Value"],
                ["Task", self.selected_task_type],
                ["Status", self.tr("Loading summary...")],
            ]
        )
        self.dataset_summary_thread = DatasetSummaryThread(
            request_id=request_id,
            task_type=self.selected_task_type,
            image_list=list(self.image_list),
            supported_shape=list(self.supported_shape),
            output_dir=self.output_dir,
            parent=self,
        )
        self.dataset_summary_thread.summary_completed.connect(
            self._on_dataset_summary_completed,
            Qt.ConnectionType.QueuedConnection,
        )
        self.dataset_summary_thread.summary_failed.connect(
            self._on_dataset_summary_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self.dataset_summary_thread.finished.connect(
            self._on_dataset_summary_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self.dataset_summary_thread.start()

    def _stop_dataset_summary_thread_if_running(self):
        if (
            self.dataset_summary_thread
            and self.dataset_summary_thread.isRunning()
        ):
            self.dataset_summary_thread.requestInterruption()
            self.dataset_summary_thread.wait(2000)

    def _on_dataset_summary_completed(self, request_id, task_type, table_data):
        if request_id != self._summary_request_id:
            return
        if task_type != self.selected_task_type:
            return

        if task_type == "Classify":
            self._classification_cache = table_data
        elif task_type == "CRNN":
            self._crnn_cache = table_data
        else:
            self._detection_cache = table_data

        self.summary_table.load_data(table_data or [])

    def _on_dataset_summary_failed(self, request_id, task_type, error_message):
        if request_id != self._summary_request_id:
            return
        if task_type != self.selected_task_type:
            return
        self.summary_table.load_data(
            [
                ["Metric", "Value"],
                ["Task", task_type],
                ["Status", self.tr("Failed to load summary")],
                ["Error", str(error_message)],
            ]
        )

    def _on_dataset_summary_finished(self):
        self.dataset_summary_thread = None

    def _get_classification_table_data(self):
        if self._classification_cache is None:
            self._classification_cache = self._compute_classification_data()
        return self._classification_cache

    def _get_detection_table_data(self):
        if self._detection_cache is None:
            self._detection_cache = self._compute_detection_data()
        return self._detection_cache

    def _get_crnn_table_data(self):
        if self._crnn_cache is None:
            self._crnn_cache = self._compute_crnn_data()
        return self._crnn_cache

    def _compute_classification_data(self):
        headers = ["Label"] + self.supported_shape + ["Total"]

        # Get classification statistics
        classify_shapes = TASK_SHAPE_MAPPINGS.get("Classify", ["flags"])
        label_infos = get_label_infos(
            self.image_list, classify_shapes, self.output_dir
        )
        if not label_infos:
            return [headers]

        table_data = [headers]
        total_counts = [0] * len(self.supported_shape)
        total_images = 0

        for label, infos in sorted(label_infos.items()):
            # All shape columns are 0 for classification
            shape_counts = [0] * len(self.supported_shape)
            image_count = infos.get("_total", 0)
            total_images += image_count

            row = [label] + [str(c) for c in shape_counts] + [str(image_count)]
            table_data.append(row)

        total_row = (
            ["Total"] + [str(c) for c in total_counts] + [str(total_images)]
        )
        table_data.append(total_row)

        return table_data

    def _compute_detection_data(self):
        return get_statistics_table_data(
            self.image_list, self.supported_shape, self.output_dir
        )

    def _compute_crnn_data(self):
        from anylabeling.views.labeling.label_converter import LabelConverter

        headers = ["Label", "Total"]
        converter = LabelConverter()
        label_counter = {}
        total_count = 0

        for image_file in self.image_list:
            label_dir, filename = os.path.split(image_file)
            if self.output_dir:
                label_dir = self.output_dir
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )
            label_text = converter.custom_to_crnn(
                image_file=image_file,
                label_file=label_file,
                split_char="_",
                fallback_from_filename=True,
            )
            if not label_text:
                continue
            label_counter[label_text] = label_counter.get(label_text, 0) + 1
            total_count += 1

        table_data = [headers]
        for label in sorted(label_counter.keys()):
            table_data.append([label, str(label_counter[label])])
        table_data.append(["Total", str(total_count)])
        return table_data

    def clear_cache(self):
        self._classification_cache = None
        self._detection_cache = None
        self._crnn_cache = None

    def closeEvent(self, event):
        if self.crnn_prepare_thread and self.crnn_prepare_thread.isRunning():
            QMessageBox.warning(
                self,
                self.tr("Preparing Training"),
                self.tr(
                    "Cannot close window while training is being prepared. Please wait for preparation to finish."
                ),
            )
            event.ignore()
            return

        if self.training_status == "training":
            QMessageBox.warning(
                self,
                self.tr("Training in Progress"),
                self.tr(
                    "Cannot close window while training is in progress. Please stop training first."
                ),
            )
            event.ignore()
            return

        try:
            if (
                hasattr(self, "export_manager")
                and self.export_manager
                and self.export_manager.is_exporting
            ):
                self.export_manager.stop_export()
        except Exception:
            pass

        if self.training_status in ["completed", "error", "stop"]:
            self.save_training_logs_to_file()

        self._stop_dataset_summary_thread_if_running()
        self.clear_cache()
        super().closeEvent(event)

    def load_images(self):
        self.parent().open_folder_dialog()
        self.image_list = self.parent().image_list
        self.clear_cache()
        self.refresh_dataset_summary()

    def init_dataset_summary(self, parent_layout):
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.addWidget(QLabel(self.tr("Dataset Summary:")))

        self.summary_table = CustomTable()
        summary_layout.addWidget(self.summary_table)
        parent_layout.addWidget(summary_widget, 1)

    def proceed_to_config(self):
        if not self.selected_task_type:
            QMessageBox.warning(
                self,
                self.tr("Validation Error"),
                self.tr("Please select a task type"),
            )
            return

        if not self.image_list:
            QMessageBox.warning(
                self,
                self.tr("Validation Error"),
                self.tr("Please load images first"),
            )
            return

        project = os.path.join(
            get_default_project_dir(), self.selected_task_type.lower()
        )
        self.config_widgets["project"].setText(project)
        self.config_widgets["project"].setReadOnly(self.project_readonly)

        self.go_to_specific_tab(1)

    def init_actions(self, parent_layout):
        actions_layout = QHBoxLayout()

        self.load_images_button = SecondaryButton(self.tr("Load Images"))
        self.load_images_button.clicked.connect(self.load_images)
        actions_layout.addWidget(self.load_images_button)
        actions_layout.addStretch()

        self.next_button = PrimaryButton(self.tr("Next"))
        self.next_button.clicked.connect(self.proceed_to_config)
        actions_layout.addWidget(self.next_button)
        parent_layout.addLayout(actions_layout)

    def init_data_tab(self):
        layout = QVBoxLayout(self.data_tab)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_task_configuration(scroll_layout)
        self.init_dataset_summary(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.init_actions(layout)

    # Config Tab
    def browse_model_file(self):
        file_filter = "Model Files (*.pt);;All Files (*)"
        if self.selected_task_type == "CRNN":
            file_filter = "Model Files (*.pth *.pt);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Model File"),
            "",
            file_filter,
        )
        if file_path:
            self.config_widgets["model"].setText(file_path)

    def browse_data_file(self):
        if self.selected_task_type == "Classify":
            dir_path = QFileDialog.getExistingDirectory(
                self, self.tr("Select Classification Dataset Directory"), ""
            )
            if dir_path:
                self.config_widgets["data"].setText(dir_path)
        elif self.selected_task_type == "CRNN":
            dir_path = QFileDialog.getExistingDirectory(
                self,
                self.tr("Select CRNN Dataset Root (optional)"),
                "",
            )
            if dir_path:
                self.config_widgets["data"].setText(dir_path)
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                self.tr("Select Data File"),
                "",
                "Text Files (*.yaml);;All Files (*)",
            )
            if file_path:
                is_valid, result = validate_data_file(file_path)
                if is_valid:
                    self.config_widgets["data"].setText(file_path)
                    self.names = result
                    logger.info(f"Data file loaded successfully: {file_path}")
                else:
                    QMessageBox.warning(
                        self, self.tr("Invalid Data File"), result
                    )
                    self.config_widgets["data"].clear()
                    self.names = []

    def browse_pose_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Pose Config File"),
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if file_path:
            self.config_widgets["pose_config"].setText(file_path)

    def setup_cuda_checkboxes(self, device_count):
        if not hasattr(self, "_cuda_layout") or not self._cuda_layout:
            if self.device_checkboxes.layout() is None:
                self._cuda_layout = QHBoxLayout(self.device_checkboxes)
            else:
                self._cuda_layout = self.device_checkboxes.layout()
            self._cuda_layout.setContentsMargins(0, 0, 0, 0)
            self._cuda_layout.setSpacing(5)
        else:
            while self._cuda_layout.count():
                child = self._cuda_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)

        for i in range(device_count):
            checkbox = CustomCheckBox(f"GPU {i}")
            checkbox.setMaximumHeight(20)
            checkbox.setChecked(True)  # Default check all GPUs
            self._cuda_layout.addWidget(checkbox)

    def on_device_changed(self, device_text):
        if device_text == "cuda":
            try:
                import torch

                if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
                    cuda_visible_devices_backup = os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    )
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                    torch.cuda.empty_cache()
                    device_count = torch.cuda.device_count()
                    if cuda_visible_devices_backup != "-1":
                        os.environ["CUDA_VISIBLE_DEVICES"] = (
                            cuda_visible_devices_backup
                        )
                else:
                    device_count = torch.cuda.device_count()

                self.setup_cuda_checkboxes(device_count)
                self.device_checkboxes.setVisible(True)
            except ImportError:
                self.device_checkboxes.setVisible(False)
        else:
            self.device_checkboxes.setVisible(False)

    def init_basic_settings(self, parent_layout):
        group = QGroupBox(self.tr("Basic Settings"))
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        self.config_widgets["project"] = CustomLineEdit()
        selected_task_type = (
            self.selected_task_type.lower()
            if self.selected_task_type
            else "detect"
        )
        text_project = os.path.join(
            get_default_project_dir(), selected_task_type
        )
        self.config_widgets["project"].setText(text_project)
        layout.addRow("Project:", self.config_widgets["project"])

        self.config_widgets["name"] = CustomLineEdit()
        self.config_widgets["name"].setText("exp")
        layout.addRow("Name:", self.config_widgets["name"])

        model_layout = QHBoxLayout()
        self.config_widgets["model"] = CustomLineEdit()
        model_browse_btn = SecondaryButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.config_widgets["model"])
        model_layout.addWidget(model_browse_btn)
        layout.addRow("Model:", model_layout)

        data_layout = QHBoxLayout()
        self.config_widgets["data"] = CustomLineEdit()
        data_browse_btn = SecondaryButton("Browse")
        data_browse_btn.clicked.connect(self.browse_data_file)
        data_layout.addWidget(self.config_widgets["data"])
        data_layout.addWidget(data_browse_btn)
        layout.addRow("Data:", data_layout)

        pose_config_layout = QHBoxLayout()
        self.config_widgets["pose_config"] = CustomLineEdit()
        pose_config_browse_btn = SecondaryButton("Browse")
        pose_config_browse_btn.clicked.connect(self.browse_pose_config_file)
        pose_config_layout.addWidget(self.config_widgets["pose_config"])
        pose_config_layout.addWidget(pose_config_browse_btn)

        self.pose_config_label = QLabel("Pose Config:")
        layout.addRow(self.pose_config_label, pose_config_layout)
        self.pose_config_layout = pose_config_layout

        self.pose_config_label.setVisible(False)
        self.config_widgets["pose_config"].setVisible(False)
        pose_config_browse_btn.setVisible(False)

        device_layout = QHBoxLayout()
        self.config_widgets["device"] = CustomComboBox()
        self.config_widgets["device"].addItems(DEVICE_OPTIONS)
        self.config_widgets["device"].setCurrentText("cpu")
        self.device_checkboxes = QWidget()
        self.device_checkboxes.setVisible(False)
        self.config_widgets["device"].currentTextChanged.connect(
            self.on_device_changed
        )
        device_layout.addWidget(self.config_widgets["device"])
        device_layout.addWidget(self.device_checkboxes)
        layout.addRow("Device:", device_layout)
        self.on_device_changed(self.config_widgets["device"].currentText())

        dataset_layout = QHBoxLayout()
        self.config_widgets["dataset_ratio"] = CustomSlider(
            Qt.Orientation.Horizontal
        )
        self.config_widgets["dataset_ratio"].setRange(5, 95)
        self.config_widgets["dataset_ratio"].setValue(80)
        self.dataset_ratio_label = QLabel("0.8")
        self.config_widgets["dataset_ratio"].valueChanged.connect(
            lambda v: self.dataset_ratio_label.setText(str(v / 100.0))
        )
        dataset_layout.addWidget(self.config_widgets["dataset_ratio"])
        dataset_layout.addWidget(self.dataset_ratio_label)
        layout.addRow("Dataset Ratio:", dataset_layout)

        parent_layout.addWidget(group)

    def toggle_advanced_settings(self):
        """Toggle the visibility of advanced settings"""
        if self.advanced_content_widget.isVisible():
            self.advanced_content_widget.setVisible(False)
            self.advanced_toggle_btn.setIcon(
                QIcon(new_icon("caret-down", "svg"))
            )
        else:
            self.advanced_content_widget.setVisible(True)
            self.advanced_toggle_btn.setIcon(
                QIcon(new_icon("caret-up", "svg"))
            )

    def init_train_settings(self, parent_layout):
        group = QGroupBox(self.tr("Train Settings"))
        layout = QVBoxLayout(group)

        # Basic settings
        basic_group = QGroupBox(self.tr("Basic"))
        basic_layout = QHBoxLayout(basic_group)
        basic_layout.addWidget(QLabel("Epochs:"))
        self.config_widgets["epochs"] = CustomSpinBox()
        self.config_widgets["epochs"].setRange(1, 10000)
        self.config_widgets["epochs"].setValue(
            DEFAULT_TRAINING_CONFIG["epochs"]
        )
        basic_layout.addWidget(self.config_widgets["epochs"])

        basic_layout.addWidget(QLabel("Batch:"))
        self.config_widgets["batch"] = CustomSpinBox()
        self.config_widgets["batch"].setRange(-1, 8192)
        self.config_widgets["batch"].setValue(DEFAULT_TRAINING_CONFIG["batch"])
        basic_layout.addWidget(self.config_widgets["batch"])

        basic_layout.addWidget(QLabel("Image Size:"))
        self.config_widgets["imgsz"] = CustomSpinBox()
        self.config_widgets["imgsz"].setRange(32, 8192)
        self.config_widgets["imgsz"].setValue(DEFAULT_TRAINING_CONFIG["imgsz"])
        basic_layout.addWidget(self.config_widgets["imgsz"])

        basic_layout.addWidget(QLabel("Workers:"))
        self.config_widgets["workers"] = CustomSpinBox()
        self.config_widgets["workers"].setRange(0, NUM_WORKERS)
        self.config_widgets["workers"].setValue(
            DEFAULT_TRAINING_CONFIG["workers"]
        )
        basic_layout.addWidget(self.config_widgets["workers"])

        basic_layout.addWidget(QLabel("Classes:"))
        self.config_widgets["classes"] = CustomLineEdit()
        self.config_widgets["classes"].setText(
            DEFAULT_TRAINING_CONFIG["classes"]
        )
        self.config_widgets["classes"].setPlaceholderText(
            self.tr("Class indices (e.g., 0,1,2) or leave empty for all")
        )
        basic_layout.addWidget(self.config_widgets["classes"])

        self.config_widgets["single_cls"] = CustomCheckBox("Single Class")
        self.config_widgets["single_cls"].setChecked(
            DEFAULT_TRAINING_CONFIG["single_cls"]
        )
        basic_layout.addWidget(self.config_widgets["single_cls"])

        basic_layout.addStretch()
        layout.addWidget(basic_group)

        # Advanced settings
        advanced_container = QWidget()
        advanced_container_layout = QVBoxLayout(advanced_container)
        advanced_container_layout.setContentsMargins(0, 0, 0, 0)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)

        advanced_label = QLabel(self.tr("Advanced Settings"))
        advanced_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(advanced_label)

        # Collapse/Expand button
        self.advanced_toggle_btn = QPushButton()
        self.advanced_toggle_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.advanced_toggle_btn.setStyleSheet(get_advanced_toggle_btn_style())
        self.advanced_toggle_btn.setIcon(QIcon(new_icon("caret-down", "svg")))
        self.advanced_toggle_btn.clicked.connect(self.toggle_advanced_settings)
        header_layout.addWidget(self.advanced_toggle_btn)
        header_layout.addStretch()
        advanced_container_layout.addWidget(header_widget)

        self.advanced_content_widget = QWidget()
        self.advanced_content_widget.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_content_widget)

        # 1. Training Strategy
        strategy_group = QGroupBox("Training Strategy")
        strat_layout = QHBoxLayout(strategy_group)
        strat_layout.addWidget(QLabel("Time (h):"))
        self.config_widgets["time"] = CustomDoubleSpinBox()
        self.config_widgets["time"].setValue(DEFAULT_TRAINING_CONFIG["time"])
        self.config_widgets["time"].setSpecialValueText("None")
        strat_layout.addWidget(self.config_widgets["time"])

        strat_layout.addWidget(QLabel("Patience:"))
        self.config_widgets["patience"] = CustomSpinBox()
        self.config_widgets["patience"].setRange(1, 10000)
        self.config_widgets["patience"].setValue(
            DEFAULT_TRAINING_CONFIG["patience"]
        )
        strat_layout.addWidget(self.config_widgets["patience"])

        strat_layout.addWidget(QLabel("Close Mosaic:"))
        self.config_widgets["close_mosaic"] = CustomSpinBox()
        self.config_widgets["close_mosaic"].setRange(0, 1000)
        self.config_widgets["close_mosaic"].setValue(
            DEFAULT_TRAINING_CONFIG["close_mosaic"]
        )
        strat_layout.addWidget(self.config_widgets["close_mosaic"])

        strat_layout.addWidget(QLabel("Optimizer:"))
        self.config_widgets["optimizer"] = CustomComboBox()
        self.config_widgets["optimizer"].addItems(OPTIMIZER_OPTIONS)
        strat_layout.addWidget(self.config_widgets["optimizer"])

        self.config_widgets["cos_lr"] = CustomCheckBox("Cosine LR")
        self.config_widgets["cos_lr"].setChecked(
            DEFAULT_TRAINING_CONFIG["cos_lr"]
        )
        strat_layout.addWidget(self.config_widgets["cos_lr"])
        self.config_widgets["amp"] = CustomCheckBox("AMP")
        self.config_widgets["amp"].setChecked(DEFAULT_TRAINING_CONFIG["amp"])
        strat_layout.addWidget(self.config_widgets["amp"])
        self.config_widgets["multi_scale"] = CustomCheckBox("Multi Scale")
        self.config_widgets["multi_scale"].setChecked(
            DEFAULT_TRAINING_CONFIG["multi_scale"]
        )
        strat_layout.addWidget(self.config_widgets["multi_scale"])
        strat_layout.addStretch()
        advanced_layout.addWidget(strategy_group)

        # 2. Learning Rate
        lr_group = QGroupBox("Learning Rate")
        lr_layout = QHBoxLayout(lr_group)
        lr_layout.addWidget(QLabel("LR0:"))
        self.config_widgets["lr0"] = CustomDoubleSpinBox()
        self.config_widgets["lr0"].setDecimals(6)
        self.config_widgets["lr0"].setValue(DEFAULT_TRAINING_CONFIG["lr0"])
        lr_layout.addWidget(self.config_widgets["lr0"])

        lr_layout.addWidget(QLabel("LRF:"))
        self.config_widgets["lrf"] = CustomDoubleSpinBox()
        self.config_widgets["lrf"].setDecimals(6)
        self.config_widgets["lrf"].setValue(DEFAULT_TRAINING_CONFIG["lrf"])
        lr_layout.addWidget(self.config_widgets["lrf"])

        lr_layout.addWidget(QLabel("Momentum:"))
        self.config_widgets["momentum"] = CustomDoubleSpinBox()
        self.config_widgets["momentum"].setDecimals(3)
        self.config_widgets["momentum"].setValue(
            DEFAULT_TRAINING_CONFIG["momentum"]
        )
        lr_layout.addWidget(self.config_widgets["momentum"])

        lr_layout.addWidget(QLabel("Weight Decay:"))
        self.config_widgets["weight_decay"] = CustomDoubleSpinBox()
        self.config_widgets["weight_decay"].setDecimals(6)
        self.config_widgets["weight_decay"].setValue(
            DEFAULT_TRAINING_CONFIG["weight_decay"]
        )
        lr_layout.addWidget(self.config_widgets["weight_decay"])
        lr_layout.addStretch()
        advanced_layout.addWidget(lr_group)

        # 3. Warmup Parameters
        warmup_group = QGroupBox("Warmup Parameters")
        warmup_layout = QHBoxLayout(warmup_group)
        warmup_layout.addWidget(QLabel("Warmup Epochs:"))
        self.config_widgets["warmup_epochs"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_epochs"].setDecimals(1)
        self.config_widgets["warmup_epochs"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_epochs"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_epochs"])

        warmup_layout.addWidget(QLabel("Warmup Momentum:"))
        self.config_widgets["warmup_momentum"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_momentum"].setDecimals(3)
        self.config_widgets["warmup_momentum"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_momentum"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_momentum"])

        warmup_layout.addWidget(QLabel("Warmup Bias LR:"))
        self.config_widgets["warmup_bias_lr"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_bias_lr"].setDecimals(3)
        self.config_widgets["warmup_bias_lr"].setValue(
            DEFAULT_TRAINING_CONFIG["warmup_bias_lr"]
        )
        warmup_layout.addWidget(self.config_widgets["warmup_bias_lr"])
        warmup_layout.addStretch()
        advanced_layout.addWidget(warmup_group)

        # 4. Augmentation Settings
        augment_group = QGroupBox("Augmentation Settings")
        augment_layout = QVBoxLayout(augment_group)
        augment_params = [
            (
                "hsv_h",
                "HSV Hue:",
                DEFAULT_TRAINING_CONFIG["hsv_h"],
                0.0,
                1.0,
                3,
            ),
            (
                "hsv_s",
                "HSV Saturation:",
                DEFAULT_TRAINING_CONFIG["hsv_s"],
                0.0,
                1.0,
                3,
            ),
            (
                "hsv_v",
                "HSV Value:",
                DEFAULT_TRAINING_CONFIG["hsv_v"],
                0.0,
                1.0,
                3,
            ),
            (
                "degrees",
                "Rotation Degrees:",
                DEFAULT_TRAINING_CONFIG["degrees"],
                -180.0,
                180.0,
                1,
            ),
            (
                "translate",
                "Translate:",
                DEFAULT_TRAINING_CONFIG["translate"],
                0.0,
                1.0,
                3,
            ),
            ("scale", "Scale:", DEFAULT_TRAINING_CONFIG["scale"], 0.0, 2.0, 3),
            (
                "shear",
                "Shear:",
                DEFAULT_TRAINING_CONFIG["shear"],
                -45.0,
                45.0,
                1,
            ),
            (
                "perspective",
                "Perspective:",
                DEFAULT_TRAINING_CONFIG["perspective"],
                0.0,
                0.001,
                6,
            ),
        ]

        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setVerticalSpacing(5)
        for i, (
            param,
            label,
            default,
            min_val,
            max_val,
            decimals,
        ) in enumerate(augment_params):
            row = i // 4
            col = (i % 4) * 2

            label_widget = QLabel(label)
            label_widget.setMinimumWidth(80)
            grid_layout.addWidget(label_widget, row, col)

            widget = CustomDoubleSpinBox()
            widget.setRange(min_val, max_val)
            widget.setDecimals(decimals)
            widget.setValue(default)
            widget.setMinimumWidth(80)
            self.config_widgets[param] = widget
            grid_layout.addWidget(widget, row, col + 1)

        for col in range(8, 10):
            grid_layout.setColumnStretch(col, 1)
        augment_layout.addLayout(grid_layout)
        advanced_layout.addWidget(augment_group)

        # 5. Regularization
        reg_group = QGroupBox("Regularization")
        reg_layout = QHBoxLayout(reg_group)
        reg_layout.addWidget(QLabel("Dropout:"))
        self.config_widgets["dropout"] = CustomDoubleSpinBox()
        self.config_widgets["dropout"].setDecimals(3)
        self.config_widgets["dropout"].setValue(
            DEFAULT_TRAINING_CONFIG["dropout"]
        )
        reg_layout.addWidget(self.config_widgets["dropout"])

        reg_layout.addWidget(QLabel("Fraction:"))
        self.config_widgets["fraction"] = CustomDoubleSpinBox()
        self.config_widgets["fraction"].setDecimals(3)
        self.config_widgets["fraction"].setValue(
            DEFAULT_TRAINING_CONFIG["fraction"]
        )
        reg_layout.addWidget(self.config_widgets["fraction"])

        self.config_widgets["rect"] = CustomCheckBox("Rectangular")
        self.config_widgets["rect"].setChecked(DEFAULT_TRAINING_CONFIG["rect"])
        reg_layout.addWidget(self.config_widgets["rect"])
        reg_layout.addStretch()
        advanced_layout.addWidget(reg_group)

        # 6. Loss Weights
        loss_group = QGroupBox("Loss Weights")
        loss_layout = QHBoxLayout(loss_group)
        loss_layout.addWidget(QLabel("Box:"))
        self.config_widgets["box"] = CustomDoubleSpinBox()
        self.config_widgets["box"].setDecimals(2)
        self.config_widgets["box"].setValue(DEFAULT_TRAINING_CONFIG["box"])
        loss_layout.addWidget(self.config_widgets["box"])

        loss_layout.addWidget(QLabel("Cls:"))
        self.config_widgets["cls"] = CustomDoubleSpinBox()
        self.config_widgets["cls"].setDecimals(2)
        self.config_widgets["cls"].setValue(DEFAULT_TRAINING_CONFIG["cls"])
        loss_layout.addWidget(self.config_widgets["cls"])

        loss_layout.addWidget(QLabel("DFL:"))
        self.config_widgets["dfl"] = CustomDoubleSpinBox()
        self.config_widgets["dfl"].setDecimals(2)
        self.config_widgets["dfl"].setValue(DEFAULT_TRAINING_CONFIG["dfl"])
        loss_layout.addWidget(self.config_widgets["dfl"])

        loss_layout.addWidget(QLabel("Pose:"))
        self.config_widgets["pose"] = CustomDoubleSpinBox()
        self.config_widgets["pose"].setDecimals(2)
        self.config_widgets["pose"].setValue(DEFAULT_TRAINING_CONFIG["pose"])
        loss_layout.addWidget(self.config_widgets["pose"])

        loss_layout.addWidget(QLabel("Kobj:"))
        self.config_widgets["kobj"] = CustomDoubleSpinBox()
        self.config_widgets["kobj"].setDecimals(2)
        self.config_widgets["kobj"].setValue(DEFAULT_TRAINING_CONFIG["kobj"])
        loss_layout.addWidget(self.config_widgets["kobj"])
        loss_layout.addStretch()
        advanced_layout.addWidget(loss_group)

        # 7. Checkpoint and Validation
        ckpt_group = QGroupBox("Checkpoint and Validation")
        ckpt_layout = QHBoxLayout(ckpt_group)
        ckpt_layout.addWidget(QLabel("Save Period:"))
        self.config_widgets["save_period"] = CustomSpinBox()
        self.config_widgets["save_period"].setRange(-1, 1000)
        self.config_widgets["save_period"].setValue(
            DEFAULT_TRAINING_CONFIG["save_period"]
        )
        self.config_widgets["save_period"].setSpecialValueText("Disabled")
        ckpt_layout.addWidget(self.config_widgets["save_period"])

        self.config_widgets["val"] = CustomCheckBox("Validation")
        self.config_widgets["val"].setChecked(DEFAULT_TRAINING_CONFIG["val"])
        ckpt_layout.addWidget(self.config_widgets["val"])
        self.config_widgets["plots"] = CustomCheckBox("Plots")
        self.config_widgets["plots"].setChecked(
            DEFAULT_TRAINING_CONFIG["plots"]
        )
        ckpt_layout.addWidget(self.config_widgets["plots"])
        self.config_widgets["save"] = CustomCheckBox("Save")
        self.config_widgets["save"].setChecked(DEFAULT_TRAINING_CONFIG["save"])
        ckpt_layout.addWidget(self.config_widgets["save"])
        self.config_widgets["resume"] = CustomCheckBox("Resume")
        self.config_widgets["resume"].setChecked(
            DEFAULT_TRAINING_CONFIG["resume"]
        )
        ckpt_layout.addWidget(self.config_widgets["resume"])
        self.config_widgets["cache"] = CustomCheckBox("Cache")
        self.config_widgets["cache"].setChecked(
            DEFAULT_TRAINING_CONFIG["cache"]
        )
        ckpt_layout.addWidget(self.config_widgets["cache"])
        self.config_widgets["skip_empty_files"] = CustomCheckBox(
            "Skip Empty Files"
        )
        self.config_widgets["skip_empty_files"].setChecked(False)
        ckpt_layout.addWidget(self.config_widgets["skip_empty_files"])
        self.config_widgets["only_checked_files"] = CustomCheckBox(
            "Only Checked Files"
        )
        self.config_widgets["only_checked_files"].setChecked(False)
        ckpt_layout.addWidget(self.config_widgets["only_checked_files"])
        ckpt_layout.addStretch()
        advanced_layout.addWidget(ckpt_group)

        advanced_container_layout.addWidget(self.advanced_content_widget)
        layout.addWidget(advanced_container)
        parent_layout.addWidget(group)

    def load_config_to_ui(self, config):
        def set_widget_value(key, value):
            if key not in self.config_widgets:
                return

            widget = self.config_widgets[key]
            widget_type = type(widget).__name__

            try:
                if widget_type == "CustomLineEdit":
                    if key == "classes":
                        widget.setText(format_classes_display(value))
                    else:
                        widget.setText(str(value) if value is not None else "")
                elif widget_type in ["CustomSpinBox", "CustomDoubleSpinBox"]:
                    widget.setValue(value)
                elif widget_type == "CustomComboBox":
                    if isinstance(value, str):
                        index = widget.findText(value)
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    else:
                        widget.setCurrentIndex(value)
                elif widget_type == "CustomCheckBox":
                    widget.setChecked(bool(value))
                elif widget_type == "CustomSlider":
                    widget.setValue(value)
            except Exception as e:
                logger.warning(f"Failed to set value for widget {key}: {e}")

        sections_to_process = [
            "basic",
            "train",
            "augment",
            "strategy",
            "learning_rate",
            "warmup",
            "regularization",
            "loss_weights",
            "checkpoint",
        ]
        for section in sections_to_process:
            if section in config:
                for key, value in config[section].items():
                    if key == "dataset_ratio":
                        if 0 <= value <= 1:
                            self.config_widgets[key].setValue(int(value * 100))
                            self.dataset_ratio_label.setText(str(value))
                        else:
                            self.config_widgets[key].setValue(int(value))
                            self.dataset_ratio_label.setText(
                                str(value / 100.0)
                            )
                    elif key == "device":
                        index = self.config_widgets[key].findText(str(value))
                        if index >= 0:
                            self.config_widgets[key].setCurrentIndex(index)
                            self.on_device_changed(str(value))
                    elif key == "optimizer":
                        index = self.config_widgets[key].findText(str(value))
                        if index >= 0:
                            self.config_widgets[key].setCurrentIndex(index)
                    elif key == "pose_config":
                        if value:
                            self.config_widgets[key].setText(value)
                    elif key in (
                        "skip_empty_files",
                        "only_checked_files",
                    ):
                        set_widget_value(key, value)
                    else:
                        set_widget_value(key, value)

        for key, value in config.items():
            if key not in sections_to_process and key in self.config_widgets:
                set_widget_value(key, value)

    def import_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Import Config"),
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            config = load_config_from_file(file_path)
            if config:
                self.load_config_to_ui(config)
                QMessageBox.information(
                    self,
                    self.tr("Success"),
                    self.tr("Config imported successfully"),
                )
            else:
                QMessageBox.warning(
                    self, self.tr("Error"), self.tr("Failed to import config")
                )

    def get_current_config(self):
        def get_widget_value(key):
            if key not in self.config_widgets:
                return None
            widget = self.config_widgets[key]
            widget_type = type(widget).__name__
            try:
                if widget_type == "CustomLineEdit":
                    return widget.text()
                elif widget_type in ["CustomSpinBox", "CustomDoubleSpinBox"]:
                    return widget.value()
                elif widget_type == "CustomComboBox":
                    return widget.currentText()
                elif widget_type == "CustomCheckBox":
                    return widget.isChecked()
                elif widget_type == "CustomSlider":
                    return widget.value()
            except Exception:
                return None
            return None

        config = {
            "basic": {
                "project": get_widget_value("project"),
                "name": get_widget_value("name"),
                "model": get_widget_value("model").strip('"'),
                "data": get_widget_value("data").strip('"'),
                "device": get_widget_value("device"),
                "dataset_ratio": (
                    get_widget_value("dataset_ratio") / 100.0
                    if get_widget_value("dataset_ratio") is not None
                    else 0.8
                ),
                "pose_config": get_widget_value("pose_config"),
            },
            "train": {
                "epochs": get_widget_value("epochs"),
                "batch": get_widget_value("batch"),
                "imgsz": get_widget_value("imgsz"),
                "workers": get_widget_value("workers"),
                "single_cls": get_widget_value("single_cls"),
                "classes": parse_string_to_digit_list(
                    get_widget_value("classes")
                ),
            },
            "strategy": {
                "time": get_widget_value("time"),
                "patience": get_widget_value("patience"),
                "close_mosaic": get_widget_value("close_mosaic"),
                "optimizer": get_widget_value("optimizer"),
                "cos_lr": get_widget_value("cos_lr"),
                "amp": get_widget_value("amp"),
                "multi_scale": get_widget_value("multi_scale"),
            },
            "learning_rate": {
                "lr0": get_widget_value("lr0"),
                "lrf": get_widget_value("lrf"),
                "momentum": get_widget_value("momentum"),
                "weight_decay": get_widget_value("weight_decay"),
            },
            "warmup": {
                "warmup_epochs": get_widget_value("warmup_epochs"),
                "warmup_momentum": get_widget_value("warmup_momentum"),
                "warmup_bias_lr": get_widget_value("warmup_bias_lr"),
            },
            "augment": {
                "hsv_h": get_widget_value("hsv_h"),
                "hsv_s": get_widget_value("hsv_s"),
                "hsv_v": get_widget_value("hsv_v"),
                "degrees": get_widget_value("degrees"),
                "translate": get_widget_value("translate"),
                "scale": get_widget_value("scale"),
                "shear": get_widget_value("shear"),
                "perspective": get_widget_value("perspective"),
            },
            "regularization": {
                "dropout": get_widget_value("dropout"),
                "fraction": get_widget_value("fraction"),
                "rect": get_widget_value("rect"),
            },
            "loss_weights": {
                "box": get_widget_value("box"),
                "cls": get_widget_value("cls"),
                "dfl": get_widget_value("dfl"),
                "pose": get_widget_value("pose"),
                "kobj": get_widget_value("kobj"),
            },
            "checkpoint": {
                "save_period": get_widget_value("save_period"),
                "val": get_widget_value("val"),
                "plots": get_widget_value("plots"),
                "save": get_widget_value("save"),
                "resume": get_widget_value("resume"),
                "cache": get_widget_value("cache"),
                "skip_empty_files": get_widget_value("skip_empty_files"),
                "only_checked_files": get_widget_value("only_checked_files"),
            },
        }

        return config

    def save_current_config(self):
        try:
            save_config(self.get_current_config())
            template = self.tr("Configuration saved successfully to %s")
            msg_test = template % get_settings_config_path()
            QMessageBox.information(self, self.tr("Success"), msg_test)
        except Exception as e:
            QMessageBox.warning(
                self, self.tr("Error"), f"Failed to save config: {str(e)}"
            )

    def start_training(self):
        if self.training_status == "training":
            QMessageBox.warning(
                self,
                self.tr("Training in Progress"),
                self.tr(
                    "Training is currently in progress. Please stop the training first if you need to reconfigure."
                ),
            )
            return

        config = self.get_current_config()
        is_valid, error_message = validate_basic_config(
            config, self.selected_task_type
        )
        if is_valid == "directory_exists":
            project_dir = error_message
            potential_model_path = self._get_existing_model_path(project_dir)

            if os.path.exists(potential_model_path):
                reply = QMessageBox.question(
                    self,
                    self.tr("Existing Model Detected"),
                    self.tr(
                        "A trained model already exists at this location.\n\n"
                        "Do you want to:\n"
                        "Yes - Export the existing model directly\n"
                        "No - Continue to retrain (will overwrite)"
                    ),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.current_project_path = project_dir
                    self.training_status = "completed"
                    save_config(config)
                    self.go_to_specific_tab(2)
                    self.update_training_status_display()
                    self.start_training_button.setVisible(False)
                    self.export_button.setVisible(True)
                    self.previous_button.setVisible(True)
                    self.update_training_images()
                    self.append_training_log(
                        f"Loaded existing model from: {potential_model_path}"
                    )
                    return

            reply = QMessageBox.question(
                self,
                self.tr("Directory Exists"),
                self.tr(
                    "Project directory already exists! Do you want to overwrite it?\nIf not, please manually modify the `Name` field value."
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                try:
                    shutil.rmtree(error_message)
                    self.append_training_log(
                        f"Removed existing directory: {error_message}"
                    )
                except Exception as e:
                    error_msg = f"Failed to remove directory: {str(e)}"
                    logger.error(error_msg)
                    return
            else:
                return
        elif not is_valid:
            QMessageBox.warning(
                self, self.tr("Validation Error"), error_message
            )
            self.append_training_log(f"Validation Error: {error_message}")
            return

        if not self.selected_task_type:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Please select a task type first"),
            )
            return

        if self.selected_task_type.lower() == "pose":
            pose_config = config["basic"].get("pose_config", "")
            if not pose_config or not os.path.exists(pose_config):
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    self.tr(
                        "Please select a valid pose configuration file for pose detection tasks"
                    ),
                )
                return

        if self.training_status in ["completed", "error"]:
            reply = QMessageBox.question(
                self,
                self.tr("Reset Training"),
                self.tr(
                    "Training traces detected. Do you want to reset the training tab?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.reset_train_tab()
        elif self.training_status == "stop":
            self.start_training_button.setVisible(True)
            self.training_status == "idle"

        save_config(config)
        self.go_to_specific_tab(2)

    def init_config_buttons(self, parent_layout):
        button_layout = QHBoxLayout()

        import_btn = SecondaryButton(self.tr("Import Config"))
        import_btn.clicked.connect(self.import_config)
        button_layout.addWidget(import_btn)

        save_btn = SecondaryButton(self.tr("Save Config"))
        save_btn.clicked.connect(self.save_current_config)
        button_layout.addWidget(save_btn)
        button_layout.addStretch()

        previous_btn = SecondaryButton(self.tr("Previous"))
        previous_btn.clicked.connect(lambda: self.go_to_specific_tab(0))
        button_layout.addWidget(previous_btn)

        train_btn = PrimaryButton(self.tr("Next"))
        train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(train_btn)

        parent_layout.addLayout(button_layout)

    def load_default_config(self):
        config = load_config()
        self.load_config_to_ui(config)

    def init_config_tab(self):
        layout = QVBoxLayout(self.config_tab)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_basic_settings(scroll_layout)
        self.init_train_settings(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.init_config_buttons(layout)
        self.load_default_config()

    # Train tab
    def update_training_status_display(self):
        color = TRAINING_STATUS_COLORS.get(self.training_status, "#6c757d")
        text = self.tr(
            TRAINING_STATUS_TEXTS.get(self.training_status, "Unknown status")
        )
        self.status_label.setText(text)
        self.status_label.setStyleSheet(get_status_label_style(color))

    def update_training_progress(self):
        if not self.current_project_path:
            return

        results_file = os.path.join(self.current_project_path, "results.csv")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) > 1:  # Skip header
                        self.current_epochs = len(rows) - 1
                        progress = min(
                            100,
                            int(
                                (self.current_epochs / self.total_epochs) * 100
                            ),
                        )
                        self.progress_bar.setValue(progress)
                        self.progress_bar.setFormat(
                            f"{self.current_epochs}/{self.total_epochs}"
                        )
            except Exception as e:
                logger.warning(f"Failed to read results.csv: {e}")

    def update_training_images(self):
        if not self.current_project_path:
            return

        def find_images_by_pattern(patterns, max_count=3):
            found_files = []
            for pattern in patterns:
                matches = glob.glob(
                    os.path.join(self.current_project_path, pattern)
                )
                matches.sort()
                found_files.extend(matches)
                if len(found_files) >= max_count:
                    break
            return found_files[:max_count]

        if self.selected_task_type == "Classify":
            image_configs = [
                {"patterns": ["train_batch*.jpg"], "max_count": 3},
                {
                    "patterns": [
                        "val_batch0_labels.jpg",
                        "val_batch0_pred.jpg",
                        "results.png",
                    ],
                    "max_count": 3,
                },
            ]
        elif self.selected_task_type == "CRNN":
            image_configs = []
        else:
            image_configs = [
                {"patterns": ["train_batch*.jpg"], "max_count": 3},
                {
                    "patterns": [
                        "*PR_curve.png",
                        "*F1_curve.png",
                        "results.png",
                    ],
                    "max_count": 3,
                },
            ]

        all_images = []
        for config in image_configs:
            all_images.extend(
                find_images_by_pattern(config["patterns"], config["max_count"])
            )

        for i, image_label in enumerate(self.image_labels):
            if i < len(all_images):
                image_path = all_images[i]
                try:
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            150,
                            150,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        image_label.setPixmap(scaled_pixmap)
                        image_label.setText("")
                        image_label.setToolTip(os.path.basename(image_path))
                        self.image_paths[i] = image_path
                    else:
                        image_label.clear()
                        image_label.setText(self.tr("No image"))
                        image_label.setToolTip("")
                        self.image_paths[i] = None
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
                    image_label.clear()
                    image_label.setText(self.tr("No image"))
                    image_label.setToolTip("")
                    self.image_paths[i] = None
            else:
                image_label.clear()
                image_label.setText(self.tr("No image"))
                image_label.setToolTip("")
                self.image_paths[i] = None

    def on_training_event(self, event_type, data):
        if event_type == "training_started":
            self.training_status = "training"
            self.total_epochs = data["total_epochs"]
            self.current_epochs = 0
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"0/{self.total_epochs}")
            self.update_training_status_display()
            self.start_training_button.setVisible(False)
            self.stop_training_button.setVisible(True)
            self.export_button.setVisible(False)
            self.previous_button.setVisible(False)
            self.progress_timer.start(1000)
            self.image_timer.start(5000)
            self.append_training_log(self.tr("Training is about to start..."))
        elif event_type == "training_completed":
            self.training_status = "completed"
            self.update_training_status_display()
            self.stop_training_button.setVisible(False)
            self.start_training_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.export_button.setVisible(True)
            self.progress_timer.stop()
            self.image_timer.stop()
            self.update_training_progress()
            self.update_training_images()
            self.append_training_log(
                self.tr("Training completed successfully!")
            )
        elif event_type == "training_error":
            self.training_status = "error"
            self.update_training_status_display()
            self.start_training_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.stop_training_button.setVisible(False)
            self.export_button.setVisible(False)
            self.progress_timer.stop()
            self.image_timer.stop()
            error_msg = data.get("error", "Unknown error occurred")
            self.append_training_log(f"ERROR: {error_msg}")
        elif event_type == "training_stopped":
            self.training_status = "stop"
            self.update_training_status_display()
            self.start_training_button.setVisible(False)
            self.previous_button.setVisible(True)
            self.stop_training_button.setVisible(False)
            self.export_button.setVisible(False)
            self.progress_timer.stop()
            self.image_timer.stop()
            self.append_training_log(self.tr("Training stopped by user"))
        elif event_type == "training_log":
            log_message = data.get("message", "")
            if log_message:
                self._update_crnn_progress_from_log(log_message)
                self.append_training_log(log_message)

    def _update_crnn_progress_from_log(self, log_message: str):
        if self.selected_task_type != "CRNN":
            return

        match = re.search(
            r"\[epoch\s+(\d+)\s*/\s*(\d+)\]",
            log_message,
            flags=re.IGNORECASE,
        )
        if not match:
            return

        current_epoch = int(match.group(1))
        total_epoch = int(match.group(2))
        if total_epoch <= 0:
            return

        self.total_epochs = total_epoch
        self.current_epochs = current_epoch
        progress = min(100, int((current_epoch / total_epoch) * 100))
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{current_epoch}/{total_epoch}")

    def append_training_log(self, text):
        def clean_ansi_codes(text: str) -> str:
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape.sub("", text)

        if hasattr(self, "log_display"):
            text = clean_ansi_codes(text)
            self.log_display.append(text.strip())

    def init_training_status(self, parent_layout):
        status_group = QGroupBox(self.tr("Training Status"))
        status_layout = QVBoxLayout(status_group)

        self.status_label = QLabel(self.tr("Ready to train"))
        self.status_label.setStyleSheet(get_status_label_style())
        status_layout.addWidget(self.status_label)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel(self.tr("Progress:")))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/0")
        self.progress_bar.setStyleSheet(get_progress_bar_style())
        progress_layout.addWidget(self.progress_bar)
        status_layout.addLayout(progress_layout)
        parent_layout.addWidget(status_group)

    def clear_training_logs(self):
        if hasattr(self, "log_display"):
            reply = QMessageBox.question(
                self,
                self.tr("Clear Logs"),
                self.tr("Are you sure you want to clear all training logs?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.log_display.clear()

    def copy_training_logs(self):
        if hasattr(self, "log_display"):
            text = self.log_display.toPlainText()
            if text:
                clipboard = QApplication.clipboard()
                clipboard.setText(text)

    def init_training_logs(self, parent_layout):
        logs_group = QGroupBox(self.tr("Training Logs"))
        logs_layout = QVBoxLayout(logs_group)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(250)
        self.log_display.setStyleSheet(get_log_display_style())
        logs_layout.addWidget(self.log_display)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.clear_logs_button = SecondaryButton(self.tr("Clear"))
        self.clear_logs_button.clicked.connect(self.clear_training_logs)
        button_layout.addWidget(self.clear_logs_button)

        self.copy_logs_button = SecondaryButton(self.tr("Copy"))
        self.copy_logs_button.clicked.connect(self.copy_training_logs)
        button_layout.addWidget(self.copy_logs_button)

        logs_layout.addLayout(button_layout)
        parent_layout.addWidget(logs_group)

    def init_training_images(self, parent_layout):
        images_group = QGroupBox(self.tr("Training Images"))
        images_layout = QVBoxLayout(images_group)
        images_layout.setContentsMargins(5, 5, 5, 5)

        self.image_labels = []
        self.image_paths = [None] * 6
        self.images_widget = QWidget()
        images_row_layout = QHBoxLayout(self.images_widget)
        images_row_layout.setSpacing(10)
        images_row_layout.setContentsMargins(0, 0, 0, 0)

        for i in range(6):
            image_label = QLabel()
            image_label.setMinimumSize(150, 150)
            image_label.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            image_label.setStyleSheet(get_image_label_style())
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setText(self.tr("No image"))
            image_label.setScaledContents(False)
            image_label.mousePressEvent = (
                lambda event, idx=i: self.on_image_clicked(idx)
            )
            self.image_labels.append(image_label)
            images_row_layout.addWidget(image_label, 1)

        images_layout.addWidget(self.images_widget, 1)
        parent_layout.addWidget(images_group, 1)

    def on_image_clicked(self, index):
        if self.image_paths[index]:
            self.open_image_file(self.image_paths[index])

    def open_image_file(self, image_path):
        try:
            is_wsl2 = False
            try:
                if (
                    hasattr(os, "uname")
                    and "microsoft" in os.uname().release.lower()
                ):
                    is_wsl2 = True
            except (AttributeError, OSError):
                pass

            if is_wsl2:  # WSL2
                windows_path = (
                    subprocess.check_output(["wslpath", "-w", image_path])
                    .decode()
                    .strip()
                )
                subprocess.run(
                    [
                        "powershell.exe",
                        "-c",
                        f'Start-Process "{windows_path}"',
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif os.name == "nt":  # Windows
                os.startfile(image_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", image_path])
            elif os.name == "posix":  # Linux
                subprocess.run(["xdg-open", image_path])
        except Exception as e:
            logger.warning(f"Failed to open image {image_path}: {e}")

    def open_training_directory(self):
        if self.current_project_path and os.path.exists(
            self.current_project_path
        ):
            try:
                is_wsl2 = False
                try:
                    if (
                        hasattr(os, "uname")
                        and "microsoft" in os.uname().release.lower()
                    ):
                        is_wsl2 = True
                except (AttributeError, OSError):
                    pass

                if is_wsl2:  # WSL2
                    wsl_path = self.current_project_path
                    windows_path = (
                        subprocess.check_output(["wslpath", "-w", wsl_path])
                        .decode()
                        .strip()
                    )
                    subprocess.run(["explorer.exe", windows_path])
                elif os.name == "nt":  # Windows
                    os.startfile(self.current_project_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", self.current_project_path])
                elif os.name == "posix":  # Linux
                    subprocess.run(["xdg-open", self.current_project_path])
            except Exception as e:
                self.append_training_log(f"Failed to open directory: {str(e)}")
                QMessageBox.information(
                    self,
                    self.tr("Info"),
                    f"Directory path: {self.current_project_path}",
                )
        else:
            QMessageBox.information(
                self,
                self.tr("Info"),
                self.tr("No training directory available"),
            )

    def stop_training(self):
        reply = QMessageBox.question(
            self,
            self.tr("Confirm Stop"),
            self.tr("Are you sure you want to stop the training?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            success = self.training_manager.stop_training()
            if success:
                self.append_training_log(self.tr("Stopping training..."))
            else:
                self.append_training_log(self.tr("Cancel to stop training"))

    def _build_auto_data_yaml(self) -> str:
        return self._build_auto_data_yaml_with_log()

    def _log_training_message(self, text, log_callback=None):
        if log_callback:
            log_callback(str(text))
        else:
            self.append_training_log(str(text))

    def _build_auto_data_yaml_with_log(
        self, log_callback=None, task_type=None, image_files=None
    ) -> str:
        selected_task = task_type if task_type is not None else self.selected_task_type
        images = image_files if image_files is not None else self.image_list

        if selected_task in [None, "Classify"]:
            return ""

        valid_shapes = TASK_SHAPE_MAPPINGS.get(selected_task, [])
        label_infos = get_label_infos(images, valid_shapes, self.output_dir)
        class_names = sorted(label_infos.keys())

        if not class_names:
            raise ValueError(
                "Data is empty and no labeled classes were found to auto-generate data.yaml"
            )

        auto_data = {
            "names": {i: name for i, name in enumerate(class_names)},
            "nc": len(class_names),
        }
        auto_data_path = os.path.join(get_trainer_root_dir(), "auto_data.yaml")
        os.makedirs(os.path.dirname(auto_data_path), exist_ok=True)

        if not save_yaml_config(auto_data, auto_data_path):
            raise RuntimeError(
                f"Failed to save auto-generated data file: {auto_data_path}"
            )

        self.names = class_names
        self._log_training_message(
            f"Auto-generated data.yaml: {auto_data_path} (classes: {len(class_names)})",
            log_callback,
        )
        return auto_data_path

    def _get_existing_model_path(self, project_dir: str) -> str:
        if self.selected_task_type == "CRNN":
            candidates = [
                os.path.join(project_dir, "weights", "best_crnn_dynamic.pth"),
                os.path.join(project_dir, "weights", "latest_crnn_dynamic.pth"),
                os.path.join(project_dir, "best_crnn_dynamic.pth"),
                os.path.join(project_dir, "latest_crnn_dynamic.pth"),
            ]
        else:
            candidates = [os.path.join(project_dir, "weights", "best.pt")]

        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _resolve_device_value(self, config_device_value):
        device_value = config_device_value
        if device_value == "cuda" and hasattr(self, "device_checkboxes"):
            selected_gpus = []
            has_cuda_selector = False
            if hasattr(self, "_cuda_layout") and self._cuda_layout:
                has_cuda_selector = self._cuda_layout.count() > 0
                for i in range(self._cuda_layout.count()):
                    widget = self._cuda_layout.itemAt(i).widget()
                    if (
                        widget
                        and hasattr(widget, "isChecked")
                        and widget.isChecked()
                    ):
                        gpu_text = widget.text()
                        gpu_id = gpu_text.split()[-1]
                        selected_gpus.append(int(gpu_id))
            if has_cuda_selector:
                device_value = selected_gpus if selected_gpus else "cpu"
            else:
                device_value = 0
        return device_value

    def _build_crnn_labels_dataset(
        self, config, log_callback=None, image_files=None
    ) -> tuple[str, str]:
        from anylabeling.views.labeling.label_converter import LabelConverter

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = os.path.join(get_dataset_path(), "crnn", f"auto_data_{timestamp}")
        os.makedirs(dataset_dir, exist_ok=True)
        labels_file = os.path.join(dataset_dir, "labels.txt")

        only_checked_files = bool(
            config.get("checkpoint", {}).get("only_checked_files", False)
        )
        converter = LabelConverter()
        lines = []

        images = image_files if image_files is not None else self.image_list
        for image_file in images:
            label_dir, filename = os.path.split(image_file)
            if self.output_dir:
                label_dir = self.output_dir
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )

            if only_checked_files and os.path.exists(label_file):
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    if not info.get("checked", False):
                        continue
                except Exception:
                    continue

            label_text = converter.custom_to_crnn(
                image_file=image_file,
                label_file=label_file,
                split_char="_",
                fallback_from_filename=True,
            )
            if not label_text:
                continue

            abs_image = os.path.abspath(image_file).replace("\\", "/")
            lines.append(f"{abs_image}\t{label_text}")

        lines.sort()
        with open(labels_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        if not lines:
            raise ValueError(
                "No valid CRNN labels found. Please set json description or use filename prefix like 1234_xxx.jpg"
            )

        self._log_training_message(
            f"Created CRNN labels file: {labels_file} (samples: {len(lines)})",
            log_callback,
        )
        return dataset_dir, labels_file

    def _get_crnn_training_args(self, config, log_callback=None, image_files=None):
        project = config["basic"]["project"]
        name = config["basic"]["name"]
        project_path = os.path.join(project, name)
        weights_dir = os.path.join(project_path, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        data_input = config["basic"].get("data", "").strip()
        if data_input:
            if os.path.isdir(data_input):
                data_root = data_input
                labels_file = os.path.join(data_root, "labels.txt")
                if not os.path.exists(labels_file):
                    raise ValueError(
                        f"labels.txt not found in CRNN dataset root: {data_root}"
                    )
                self._log_training_message(
                    f"Using CRNN dataset root: {data_root}",
                    log_callback,
                )
            elif os.path.isfile(data_input):
                labels_file = data_input
                data_root = os.path.dirname(labels_file) or "."
                self._log_training_message(
                    f"Using CRNN labels file: {labels_file}",
                    log_callback,
                )
            else:
                raise ValueError(
                    "CRNN data must be an existing dataset root directory or labels.txt file"
                )
        else:
            data_root, labels_file = self._build_crnn_labels_dataset(
                config,
                log_callback=log_callback,
                image_files=image_files,
            )

        device_value = config["basic"].get("device_resolved")
        if device_value is None:
            device_value = self._resolve_device_value(config["basic"]["device"])
        model_path = config["basic"].get("model", "").strip()
        train_ratio = config["basic"].get("dataset_ratio", 0.8)
        lr0 = config.get("learning_rate", {}).get("lr0", 0.001)
        weight_decay = config.get("learning_rate", {}).get("weight_decay", 1e-4)

        train_args = {
            "__task_type__": "crnn",
            "project": project,
            "name": name,
            "device": device_value,
            "data_root": data_root,
            "labels_file": labels_file,
            "charset_file": os.path.join(project_path, "charset.txt"),
            "best_model": os.path.join(weights_dir, "best_crnn_dynamic.pth"),
            "latest_model": os.path.join(weights_dir, "latest_crnn_dynamic.pth"),
            "epochs": int(config.get("train", {}).get("epochs", 50)),
            "batch_size": int(config.get("train", {}).get("batch", 32)),
            "num_workers": int(config.get("train", {}).get("workers", 0)),
            "train_ratio": float(train_ratio),
            "img_h": int(config.get("train", {}).get("imgsz", 32)),
            "lr": float(lr0) if float(lr0) > 0 else 0.001,
            "weight_decay": (
                float(weight_decay) if float(weight_decay) >= 0 else 1e-4
            ),
            "resume": model_path if model_path and os.path.exists(model_path) else "",
        }

        cmd_parts = [
            "python",
            "crnn/train_dynamic.py",
            f"--data-root={train_args['data_root']}",
            f"--labels-file={train_args['labels_file']}",
            f"--best-model={train_args['best_model']}",
            f"--latest-model={train_args['latest_model']}",
            f"--epochs={train_args['epochs']}",
            f"--batch-size={train_args['batch_size']}",
            f"--train-ratio={train_args['train_ratio']}",
            f"--device={train_args['device']}",
        ]
        if train_args["resume"]:
            cmd_parts.append(f"--resume={train_args['resume']}")
        self._log_training_message(
            f"Training command: {' '.join(cmd_parts)}", log_callback
        )
        return train_args

    def get_training_args(self, config, log_callback=None):
        try:
            selected_task_type = config.get(
                "__selected_task_type__", self.selected_task_type
            )
            image_files = config.get("__image_list_snapshot__", None)

            if selected_task_type == "CRNN":
                return self._get_crnn_training_args(
                    config,
                    log_callback=log_callback,
                    image_files=image_files,
                )

            data_file = config["basic"].get("data", "").strip()
            if selected_task_type != "Classify" and not data_file:
                data_file = self._build_auto_data_yaml_with_log(
                    log_callback=log_callback,
                    task_type=selected_task_type,
                    image_files=image_files,
                )
                config["basic"]["data"] = data_file

            if selected_task_type == "Classify" and os.path.isdir(
                config["basic"]["data"]
            ):
                data_path = config["basic"]["data"]
                self._log_training_message(
                    f"Using existing dataset: {data_path}",
                    log_callback,
                )
            else:
                temp_dir = create_yolo_dataset(
                    image_files if image_files is not None else self.image_list,
                    selected_task_type,
                    config["basic"]["dataset_ratio"],
                    data_file,
                    self.output_dir,
                    config["basic"].get("pose_config"),
                    config["checkpoint"].get("skip_empty_files", False),
                    config["checkpoint"].get("only_checked_files", False),
                )
                logger.info(f"Successfully created YOLO dataset at {temp_dir}")
                self._log_training_message(
                    f"Created dataset: {temp_dir}", log_callback
                )

                if selected_task_type == "Classify":
                    data_path = temp_dir
                else:
                    data_path = os.path.join(temp_dir, "data.yaml")

            device_value = config["basic"].get("device_resolved")
            if device_value is None:
                device_value = self._resolve_device_value(config["basic"]["device"])

            train_args = {
                "data": data_path,
                "model": config["basic"]["model"],
                "project": config["basic"]["project"],
                "name": config["basic"]["name"],
                "device": device_value,
            }

            # Add advanced parameters
            advanced_params = {}
            for section in [
                "train",
                "strategy",
                "learning_rate",
                "warmup",
                "augment",
                "regularization",
                "loss_weights",
                "checkpoint",
            ]:
                advanced_params.update(config.get(section, {}))
            advanced_params = {
                key: value
                for key, value in advanced_params.items()
                if key in DEFAULT_TRAINING_CONFIG
                and value != DEFAULT_TRAINING_CONFIG[key]
            }
            train_args.update(advanced_params)

            # Log the training command
            cmd_parts = ["yolo", selected_task_type.lower(), "train"]
            for key, value in train_args.items():
                cmd_parts.append(f"{key}={value}")
            self._log_training_message(
                f"Training command: {' '.join(cmd_parts)}",
                log_callback,
            )

            return train_args

        except Exception as e:
            self._log_training_message(
                f"Error preparing training args: {str(e)}",
                log_callback,
            )
            raise

    def start_training_from_train_tab(self):
        if self.crnn_prepare_thread and self.crnn_prepare_thread.isRunning():
            QMessageBox.warning(
                self,
                self.tr("Training Preparation"),
                self.tr("Training preparation is already in progress"),
            )
            return

        config = self.get_current_config()
        project_path = config["basic"]["project"]
        name = config["basic"]["name"]
        self.current_project_path = os.path.join(project_path, name)
        config_snapshot = copy.deepcopy(config)
        config_snapshot["__selected_task_type__"] = self.selected_task_type
        config_snapshot["__image_list_snapshot__"] = list(self.image_list)
        config_snapshot["basic"]["device_resolved"] = self._resolve_device_value(
            config["basic"]["device"]
        )

        try:
            self.start_training_button.setEnabled(False)
            self.append_training_log(self.tr("Preparing training..."))
            self.crnn_prepare_thread = CRNNPrepareThread(
                self.get_training_args,
                config_snapshot,
                self,
            )
            self.crnn_prepare_thread.prepare_log.connect(
                self.append_training_log, Qt.ConnectionType.QueuedConnection
            )
            self.crnn_prepare_thread.prepare_success.connect(
                self._on_training_args_prepared,
                Qt.ConnectionType.QueuedConnection,
            )
            self.crnn_prepare_thread.prepare_failed.connect(
                self._on_training_prepare_failed,
                Qt.ConnectionType.QueuedConnection,
            )
            self.crnn_prepare_thread.finished.connect(
                self._on_training_prepare_finished,
                Qt.ConnectionType.QueuedConnection,
            )
            self.crnn_prepare_thread.start()

        except Exception as e:
            error_msg = f"Failed to start training: {str(e)}"
            self.append_training_log(f"ERROR: {error_msg}")
            QMessageBox.critical(self, self.tr("Training Error"), error_msg)
            self.start_training_button.setEnabled(True)

    def _on_training_args_prepared(self, train_args):
        self.total_epochs = int(train_args.get("epochs", 100))
        success, message = self.training_manager.start_training(
            train_args, self.selected_task_type or ""
        )
        if not success:
            self.append_training_log(f"Failed to start training: {message}")
            QMessageBox.critical(self, self.tr("Training Error"), message)
            self.start_training_button.setEnabled(True)

    def _on_training_prepare_failed(self, error_message):
        self.append_training_log(
            f"ERROR: Failed to prepare training: {error_message}"
        )
        QMessageBox.critical(
            self,
            self.tr("Training Error"),
            f"Failed to prepare training: {error_message}",
        )
        self.start_training_button.setEnabled(True)

    def _on_training_prepare_finished(self):
        self.crnn_prepare_thread = None

    def init_training_actions(self, parent_layout):
        actions_layout = QHBoxLayout()

        self.open_dir_button = SecondaryButton(self.tr("Open Directory"))
        self.open_dir_button.clicked.connect(self.open_training_directory)
        actions_layout.addWidget(self.open_dir_button)
        actions_layout.addStretch()

        self.stop_training_button = SecondaryButton(self.tr("Stop Training"))
        self.stop_training_button.clicked.connect(self.stop_training)
        self.stop_training_button.setVisible(False)
        actions_layout.addWidget(self.stop_training_button)

        self.previous_button = SecondaryButton(self.tr("Previous"))
        self.previous_button.clicked.connect(
            lambda: self.go_to_specific_tab(1)
        )
        self.previous_button.setVisible(True)
        actions_layout.addWidget(self.previous_button)

        self.start_training_button = PrimaryButton(self.tr("Start Training"))
        self.start_training_button.clicked.connect(
            self.start_training_from_train_tab
        )
        actions_layout.addWidget(self.start_training_button)

        self.export_button = PrimaryButton(self.tr("Export"))
        self.export_button.clicked.connect(self.start_export)
        self.export_button.setVisible(False)
        actions_layout.addWidget(self.export_button)

        parent_layout.addLayout(actions_layout)

    def init_train_tab(self):
        layout = QVBoxLayout(self.train_tab)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_training_status(scroll_layout)
        self.init_training_logs(scroll_layout)
        self.init_training_images(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        self.init_training_actions(layout)

    def on_export_event(self, event_type, data):
        if event_type == "export_started":
            self.append_training_log(self.tr("Export started..."))
            self.export_button.setEnabled(False)
        elif event_type == "export_completed":
            exported_path = data.get("exported_path", "")
            export_format = data.get("format", "onnx")
            self.append_training_log(
                self.tr(
                    f"Export completed successfully! File saved to: {exported_path}"
                )
            )
            QMessageBox.information(
                self,
                self.tr("Export Successful"),
                self.tr(
                    f"Model successfully exported to {export_format.upper()} format:\n{exported_path}"
                ),
            )
            self.export_button.setEnabled(True)
        elif event_type == "export_error":
            error_msg = data.get("error", "Unknown error occurred")
            self.append_training_log(f"ERROR: {error_msg}")
            QMessageBox.warning(self, self.tr("Export Error"), error_msg)
            self.export_button.setEnabled(True)
        elif event_type == "export_log":
            log_message = data.get("message", "")
            if log_message:
                self.append_training_log(log_message)

    def start_export(self):
        if not self.current_project_path:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("No training project available for export"),
            )
            return

        if self.selected_task_type != "CRNN":
            weights_path = os.path.join(
                self.current_project_path, "weights", "best.pt"
            )
            if not os.path.exists(weights_path):
                QMessageBox.warning(
                    self,
                    self.tr("Model Not Found"),
                    self.tr(f"Model weights not found at: {weights_path}"),
                )
                return

        if self.selected_task_type == "CRNN":
            export_dialog = ExportFormatDialog(self, allowed_formats=["ncnn"])
        else:
            export_dialog = ExportFormatDialog(self)
        if export_dialog.exec() == QDialog.DialogCode.Accepted:
            export_format = export_dialog.get_selected_format()
            success, message = self.export_manager.start_export(
                self.current_project_path,
                export_format,
                self.selected_task_type or "",
            )
            if not success:
                QMessageBox.critical(self, self.tr("Export Error"), message)
                self.append_training_log(f"Failed to start export: {message}")

    def reset_train_tab(self):
        self.training_status = "idle"
        self.current_project_path = None
        self.current_epochs = 0
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0/0")
        self.update_training_status_display()

        if hasattr(self, "log_display"):
            self.log_display.clear()

        for i, image_label in enumerate(self.image_labels):
            image_label.clear()
            image_label.setText(self.tr("No image"))
            image_label.setToolTip("")
            self.image_paths[i] = None

        self.previous_button.setVisible(True)
        self.start_training_button.setVisible(True)
        self.export_button.setVisible(False)
        self.stop_training_button.setVisible(False)
