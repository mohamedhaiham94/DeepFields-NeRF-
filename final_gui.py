import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
import numpy as np
import torch
import os

from PySide6.QtCore import (
    Qt,
    QObject,
    Signal,
    QRunnable,
    QThreadPool,
    Slot,
)
from PySide6.QtWidgets import (
    QApplication,
    QSpinBox,
    QFormLayout,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QSizePolicy,
    QTabWidget,
    QDoubleSpinBox,
    QLineEdit,
    QGroupBox,
    QScrollArea,
    QTextEdit,
    QMessageBox,
    QProgressDialog,
)
from vispy import scene
from qt_material import apply_stylesheet
import yaml

# --- Worker for Asynchronous Data Loading (Unchanged) ---


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    result = Signal(dict)
    progress = Signal(str)


class VolumeLoaderWorker(QRunnable):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.signals.progress.emit("Loading volume from disk...")
            volume = torch.load(self.file_path, weights_only=False)
            points, rgbs = None, None

            if "points_normalized" in volume:
                self.signals.progress.emit("Processing normalized points...")
                points = volume["points_normalized"]
                rgbs = volume["rgbs"]
            else:
                self.signals.progress.emit("Processing occupancy and RGB volumes...")
                curr_volume = volume["occupancy_volume"]
                rgb_volume = volume["rgb_volume"]
                rgb_volume = rgb_volume / rgb_volume.max()
                self.signals.progress.emit("Finding active points (np.argwhere)...")
                points_indices = np.argwhere(curr_volume.numpy())
                self.signals.progress.emit("Extracting RGB values...")
                rgbs = rgb_volume[
                    points_indices[:, 0], points_indices[:, 1], points_indices[:, 2]
                ].numpy()
                self.signals.progress.emit("Normalizing point coordinates...")
                res = curr_volume.shape[0]
                points = (points_indices / (res - 1.0)) * 2.0 - 1.0

            result_data = {"points": points, "rgbs": rgbs}
            self.signals.result.emit(result_data)
        except Exception as e:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


# --- Main Application Window ---


class VispyViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeRF Pipeline Script Launcher")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")

        # State tracking
        self.current_visual = None
        self.current_points = None
        self.current_colors = None
        self.point_size = 1
        self.aabb_z_min = None
        self.aabb_z_max = None

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab1 = self.create_config_creator_tab()
        self.tab2 = self.create_script_launcher_tab()
        self.tab3 = self.setup_volume_tab()

        self.tabs.addTab(self.tab1, "Configuration")
        self.tabs.addTab(self.tab2, "Script Launcher")
        self.tabs.addTab(self.tab3, "Volume Viewer")
        # self.setup_volume_tab()

    def load_volume(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Volume File", "", "Torch Files (*.pth *.pt)"
        )
        if not file_path:
            return

        self.progress_dialog = QProgressDialog("Loading data...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.show()

        worker = VolumeLoaderWorker(file_path)
        worker.signals.result.connect(self.on_loading_complete)
        worker.signals.finished.connect(self.on_loading_finished)
        worker.signals.error.connect(self.on_loading_error)
        worker.signals.progress.connect(self.update_progress_label)
        self.threadpool.start(worker)

    def update_progress_label(self, message):
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.setLabelText(message)

    def on_loading_error(self, error_tuple):
        print("Loading Error:", error_tuple)
        QMessageBox.critical(
            self, "Loading Error", f"Failed to load volume.\n{error_tuple[1]}"
        )

    def on_loading_finished(self):
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.close()
        print("Loading task finished.")

    def on_loading_complete(self, data):
        print("Data loading complete. Populating viewer.")
        if self.current_visual is not None:
            self.current_visual.parent = None
            self.current_visual = None

        self.current_points = data["points"]
        self.current_colors = data["rgbs"]
        print(f"Loaded {len(self.current_points)} points.")

        self.z_min_input.blockSignals(True)
        self.z_max_input.blockSignals(True)
        self.z_min_input.setValue(-1.0)
        self.z_max_input.setValue(1.0)
        self.z_min_input.blockSignals(False)
        self.z_max_input.blockSignals(False)

        if self.current_visual is None:
            self.current_visual = scene.visuals.Markers()
            self.current_visual.set_gl_state(blend=False, depth_test=True)
            self.view.add(self.current_visual)

        self.update_z_slice()
        self.canvas.show()
        self.tabs.setCurrentWidget(self.tab3)

    def setup_volume_tab(self):
        tab_widget = QWidget()
        main_layout = QHBoxLayout(tab_widget)
        controls_widget = QWidget()
        controls_widget.setFixedWidth(250)
        controls_layout = QVBoxLayout(controls_widget)

        # --- Main Controls ---
        vol_bt = QPushButton("Load Volume", clicked=self.load_volume)
        vol_bt.setToolTip("Load a volume from a .pth or .pt file")
        controls_layout.addWidget(vol_bt)

        # controls_layout.addWidget(
        #     QPushButton("Load Volume", clicked=self.load_volume).setToolTip(
        #         "Load a volume from a .pth or .pt file"
        #     )
        # )
        delete_bt = QPushButton("Delete Volume", clicked=self.delete_volume)
        delete_bt.setToolTip("Delete the currently loaded volume")
        controls_layout.addWidget(delete_bt)

        # controls_layout.addWidget(
        #     QPushButton("Delete Volume", clicked=self.delete_volume)
        # )

        # --- Point Size Controls ---
        point_size_group = QGroupBox("Point Size")
        point_size_layout = QHBoxLayout()
        decrease_button = QPushButton("−", clicked=self.decrease_point_size)
        increase_button = QPushButton("+", clicked=self.increase_point_size)
        self.size_label = QLabel(f"Size: {self.point_size}")
        decrease_button.setFixedSize(45, 45)
        increase_button.setFixedSize(45, 45)
        point_size_layout.addWidget(decrease_button)
        point_size_layout.addWidget(increase_button)
        point_size_layout.addWidget(self.size_label)
        point_size_group.setLayout(point_size_layout)
        controls_layout.addWidget(point_size_group)

        # --- Visibility Controls ---
        visibility_group = QGroupBox("Visibility")
        visibility_layout = QVBoxLayout()
        self.origin_checkbox = QCheckBox(
            "Show Origin Axis", checked=True, stateChanged=self.toggle_origin_visibility
        )
        self.unitcube_checkbox = QCheckBox(
            "Show Unit Cube",
            checked=True,
            stateChanged=self.toggle_unit_cube_visibility,
        )
        visibility_layout.addWidget(self.origin_checkbox)
        visibility_layout.addWidget(self.unitcube_checkbox)
        visibility_group.setLayout(visibility_layout)
        controls_layout.addWidget(visibility_group)

        # --- AABB Controls ---
        aabb_group = QGroupBox("AABB Controls")
        aabb_layout = QFormLayout()
        aabb_layout.setSpacing(10)
        load_json_button = QPushButton("Load AABB from JSON")
        load_json_button.setToolTip("Load AABB data from a transform json file")
        load_json_button.clicked.connect(self.load_aabb_json)
        aabb_layout.addRow(load_json_button)

        # ADDED: Label to show the loaded JSON filename
        self.aabb_file_label = QLabel("None")
        self.aabb_file_label.setStyleSheet("font-style: italic; color: #AAAAAA;")
        aabb_layout.addRow("Loaded File:", self.aabb_file_label)

        self.aabb_z_min_label = QLabel("Not loaded")
        self.aabb_z_max_label = QLabel("Not loaded")
        aabb_layout.addRow("AABB Z-Min:", self.aabb_z_min_label)
        aabb_layout.addRow("AABB Z-Max:", self.aabb_z_max_label)

        self.aabb_clip_below = QCheckBox("Clip Below AABB")
        self.aabb_clip_above = QCheckBox("Clip Above AABB")
        self.aabb_clip_below.stateChanged.connect(self.update_z_slice)
        self.aabb_clip_above.stateChanged.connect(self.update_z_slice)
        aabb_layout.addRow(self.aabb_clip_below)
        aabb_layout.addRow(self.aabb_clip_above)
        aabb_group.setLayout(aabb_layout)
        controls_layout.addWidget(aabb_group)

        # --- Manual Slicing Controls ---
        z_slice_group = QGroupBox("Manual Z-Axis Slicing")
        z_slice_layout = QFormLayout()
        self.z_min_input = QDoubleSpinBox(
            minimum=-1.0, maximum=1.0, singleStep=0.05, value=-1.0
        )
        self.z_max_input = QDoubleSpinBox(
            minimum=-1.0, maximum=1.0, singleStep=0.05, value=1.0
        )
        self.z_min_input.valueChanged.connect(self.update_z_slice)
        self.z_max_input.valueChanged.connect(self.update_z_slice)
        z_slice_layout.addRow("Z Min:", self.z_min_input)
        z_slice_layout.addRow("Z Max:", self.z_max_input)
        z_slice_group.setLayout(z_slice_layout)
        controls_layout.addWidget(z_slice_group)

        controls_layout.addStretch()

        # --- VisPy Canvas Setup ---
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=45, azimuth=120, elevation=30, distance=4.0
        )
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
        cube_lines = np.array(
            [
                [1, -1, -1],
                [1, 1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, 1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
                [-1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [-1, 1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, 1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, 1],
            ]
        )
        self.unit_cube = scene.visuals.Line(
            pos=cube_lines, color="red", method="gl", width=2, parent=self.view.scene
        )
        canvas_widget = self.canvas.native
        canvas_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout.addWidget(canvas_widget, stretch=1)
        main_layout.addWidget(controls_widget, stretch=0)

        return tab_widget

    def load_aabb_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open transform.json", os.getcwd(), "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if (
                "scene_aabb" in data
                and "aabb_min" in data["scene_aabb"]
                and "aabb_max" in data["scene_aabb"]
            ):
                # self.aabb_z_min = data["scene_aabb"]["aabb_min"][2]
                self.aabb_z_min = data["scene_aabb"]["aabb_remove_below"]
                # self.aabb_z_max = data["scene_aabb"]["aabb_max"][2]
                self.aabb_z_max = data["scene_aabb"]["aabb_remove_above"]

                # UPDATE: Set the labels with loaded data
                self.aabb_file_label.setText(Path(file_path).name)
                self.aabb_z_min_label.setText(f"{self.aabb_z_min:.4f}")
                self.aabb_z_max_label.setText(f"{self.aabb_z_max:.4f}")
                print(
                    f"Loaded AABB Z-range: [{self.aabb_z_min}, {self.aabb_z_max}] from {Path(file_path).name}"
                )

                self.update_z_slice()
            else:
                raise KeyError("'scene_aabb' or its keys not found in JSON.")
        except Exception as e:
            QMessageBox.critical(
                self, "JSON Error", f"Failed to load or parse AABB data:\n{e}"
            )
            self.aabb_z_min = None
            self.aabb_z_max = None
            self.aabb_file_label.setText("Error")
            self.aabb_z_min_label.setText("Error")
            self.aabb_z_max_label.setText("Error")

    def update_z_slice(self):
        if self.current_points is None or self.current_visual is None:
            return

        z_min_manual = self.z_min_input.value()
        z_max_manual = self.z_max_input.value()

        if z_min_manual > z_max_manual:
            return

        mask = (self.current_points[:, 2] >= z_min_manual) & (
            self.current_points[:, 2] <= z_max_manual
        )

        if self.aabb_clip_below.isChecked() and self.aabb_z_min is not None:
            mask &= self.current_points[:, 2] >= self.aabb_z_min

        if self.aabb_clip_above.isChecked() and self.aabb_z_max is not None:
            mask &= self.current_points[:, 2] <= self.aabb_z_max

        visible_points = self.current_points[mask]
        visible_colors = self.current_colors[mask]

        self.current_visual.set_data(
            visible_points,
            face_color=visible_colors,
            edge_color=visible_colors,
            size=self.point_size,
            edge_width=0.0
            # size=0.1,
        )
        self.canvas.update()

    def toggle_origin_visibility(self, state):
        if self.axis is not None:
            self.axis.visible = bool(state)
            self.canvas.update()

    def toggle_unit_cube_visibility(self, state):
        if self.unit_cube is not None:
            self.unit_cube.visible = bool(state)
            self.canvas.update()

    def delete_volume(self):
        if self.current_visual is not None:
            self.current_visual.parent = None
            self.current_visual = None
            self.current_points = None
            self.current_colors = None

            # RESET: Clear AABB data and labels
            self.aabb_z_min = None
            self.aabb_z_max = None
            self.aabb_file_label.setText("None")
            self.aabb_z_min_label.setText("Not loaded")
            self.aabb_z_max_label.setText("Not loaded")
            self.aabb_clip_below.setChecked(False)
            self.aabb_clip_above.setChecked(False)

            self.canvas.update()
            print("Volume deleted from viewer.")

    def update_point_size(self):
        self.size_label.setText(f"Size: {self.point_size}")
        if self.current_visual is not None:
            self.current_visual.set_data(size=self.point_size)
            # self.current_visual.set_data(size=0.1)
            self.canvas.update()
        self.update_z_slice()

    def increase_point_size(self):
        if self.point_size < 10:
            self.point_size += 1
            self.update_point_size()

    def decrease_point_size(self):
        if self.point_size > 1:
            self.point_size -= 1
            self.update_point_size()

    def create_script_launcher_tab(self):
        """Create the script launcher tab"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)

        # Scene name input
        scene_group = QGroupBox("Scene Configuration")
        scene_layout = QVBoxLayout()
        scene_layout.addWidget(QLabel("Enter scene name:"))
        self.scene_input = QLineEdit()
        self.scene_input.setPlaceholderText("e.g., my_scene")
        scene_layout.addWidget(self.scene_input)
        scene_group.setLayout(scene_layout)
        layout.addWidget(scene_group)

        # Script selection
        script_group = QGroupBox("Select Scripts to Run")
        script_layout = QVBoxLayout()

        # Define scripts with descriptions
        self.scripts = {
            "resize_images.py": {
                "command": "python scripts/resize_images.py --cfg_path cfg/{scene}.yml",
                "description": "Resize input images or copy original images to tmp folder",
            },
            "run_colmap.py": {
                "command": "python scripts/run_colmap.py --workspace ./tmp/{scene}/",
                "description": "Run COLMAP reconstruction",
            },
            "colmap2nerf.py": {
                # "command": "python scripts/colmap2nerf.py --cfg_path cfg/{scene}.yml",
                "command": "python scripts/colmap2nerf_corrected.py --cfg_path cfg/{scene}.yml",
                "description": "Transform scene alignment and compute AABB",
            },
            "precompute_rays.py": {
                "command": "python scripts/precompute_rays.py --cfg_path cfg/{scene}.yml",
                "description": "Precompute rays and save as NPZ",
            },
            "train.py": {
                "command": "python scripts/train.py --cfg_path cfg/{scene}.yml",
                # "command": "python scripts/train_no_amp.py --cfg_path cfg/{scene}.yml",
                "description": "Train the NeRF model",
            },
            "extract_vol.py": {
                "command": "python scripts/extract_vol.py --cfg_path cfg/{scene}.yml",
                "description": "Extract volume from trained model",
            },
            "post_process_vol.py": {
                "command": "python scripts/post_process_vol.py --cfg_path cfg/{scene}.yml",
                "description": "Post-process volume using AABB",
            },
            "write_format.py": {
                "command": "python scripts/write_format.py --cfg_path cfg/{scene}.yml",
                "description": "Convert volume to VTI and TIFF formats",
            },
        }

        self.checkboxes = {}
        for script_name, script_info in self.scripts.items():
            cb = QCheckBox(f"{script_name} - {script_info['description']}")
            script_layout.addWidget(cb)
            self.checkboxes[script_name] = cb

        script_group.setLayout(script_layout)

        # Make script group scrollable
        scroll = QScrollArea()
        scroll.setWidget(script_group)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(420)
        layout.addWidget(scroll)

        # Buttons
        button_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(clear_all_btn)

        launch_btn = QPushButton("Launch Scripts")
        launch_btn.clicked.connect(self.launch_scripts)
        launch_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        button_layout.addWidget(launch_btn)

        layout.addLayout(button_layout)

        # Configuration display
        config_group = QGroupBox("Configuration Preview")
        config_layout = QVBoxLayout()
        self.config_display = QTextEdit()
        self.config_display.setMaximumHeight(150)
        self.config_display.setReadOnly(True)
        config_layout.addWidget(self.config_display)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Connect scene input to update config display
        # Use a safer connection method
        self.scene_input.textChanged.connect(self.safe_update_config_display)

        # Initial config display update
        self.safe_update_config_display()
        return tab_widget

    def select_all(self):
        """Select all checkboxes"""
        try:
            for cb in self.checkboxes.values():
                if cb and not cb.isHidden():
                    cb.setChecked(True)
            self.safe_update_config_display()
        except Exception as e:
            print(f"Error in select_all: {e}")

    def clear_all(self):
        """Clear all checkboxes"""
        try:
            for cb in self.checkboxes.values():
                if cb and not cb.isHidden():
                    cb.setChecked(False)
            self.safe_update_config_display()
        except Exception as e:
            print(f"Error in clear_all: {e}")

    def launch_scripts(self):
        """Launch selected scripts"""
        try:
            scene_name = (
                self.scene_input.text().strip()
                if hasattr(self, "scene_input") and self.scene_input
                else ""
            )
            if not scene_name:
                QMessageBox.warning(
                    self, "Missing Scene Name", "Please enter a scene name."
                )
                return

            # Get selected scripts
            selected_scripts = []
            for script_name, cb in self.checkboxes.items():
                if cb and cb.isChecked():
                    command = self.scripts[script_name]["command"].format(
                        scene=scene_name
                    )
                    selected_scripts.append((script_name, command))

            if not selected_scripts:
                QMessageBox.warning(
                    self,
                    "No Scripts Selected",
                    "Please select at least one script to run.",
                )
                return

            # Check if config file exists
            cfg_path = Path(f"cfg/{scene_name}.yml")
            if not cfg_path.exists():
                reply = QMessageBox.question(
                    self,
                    "Config File Not Found",
                    f"Configuration file '{cfg_path}' not found. Continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            self.run_scripts_in_terminal(selected_scripts, scene_name)

        except Exception as e:
            QMessageBox.critical(
                self, "Launch Error", f"Error launching scripts: {str(e)}"
            )

    def safe_update_config_display(self):
        """Safely update config display with error handling"""
        try:
            # Check if config_display still exists and is valid
            if not hasattr(self, "config_display") or not self.config_display:
                return

            # Additional check to see if the widget is still valid
            if self.config_display.parent() is None:
                return

            scene_name = (
                self.scene_input.text().strip()
                if hasattr(self, "scene_input") and self.scene_input
                else "your_scene"
            )
            if not scene_name:
                scene_name = "your_scene"

            cfg_path = f"cfg/{scene_name}.yml"
            workspace = f"./data/{scene_name}/"

            config_text = "==== Configuration ====\n"
            config_text += f"Scene Name:     {scene_name}\n"
            config_text += f"CFG Path:       {cfg_path}\n"
            config_text += f"Workspace:      {workspace}\n\n"

            config_text += "Selected Scripts:\n"
            if hasattr(self, "checkboxes"):
                for script_name, cb in self.checkboxes.items():
                    if cb and not cb.isHidden():
                        status = "✓" if cb.isChecked() else "✗"
                        config_text += f"{status} {script_name}\n"

            config_text += "======================"

            # Safely set the text
            self.config_display.setText(config_text)

        except RuntimeError as e:
            # Handle the specific Qt object deletion error
            print(f"Qt object error in update_config_display: {e}")
        except Exception as e:
            print(f"General error in update_config_display: {e}")

    def run_scripts_in_terminal(self, scripts, scene_name):
        """Run selected scripts in terminal with proper virtual environment activation"""
        try:
            system = platform.system()

            if system == "Windows":
                # Windows batch-like execution
                commands = []

                # Add virtual environment activation (adjust path as needed)
                venv_paths = [
                    r"C:\Users\Dell_AOS\Desktop\Nerf-Kultu\NeRF\.venv\Scripts\activate.bat",
                    r".venv\Scripts\activate.bat",
                    r"venv\Scripts\activate.bat",
                ]

                for venv_path in venv_paths:
                    if Path(venv_path).exists():
                        commands.append(f"call {venv_path}")
                        break

                # Add echo commands and script execution
                for script_name, command in scripts:
                    commands.append(f"echo Running {script_name}")
                    commands.append(
                        command.replace("/", "\\")
                    )  # Convert to Windows paths

                commands.append("echo.")
                commands.append("echo All selected scripts executed.")
                commands.append("pause")

                full_command = " && ".join(commands)

                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", full_command], shell=True
                )

            elif system in ["Linux", "Darwin"]:
                # Unix-like systems
                commands = []

                # Try to activate virtual environment
                venv_paths = [
                    ".venv/bin/activate",
                    "venv/bin/activate",
                    ".env/bin/activate",
                ]
                venv_activated = False
                for venv_path in venv_paths:
                    if Path(venv_path).exists():
                        commands.append(f"source {venv_path}")
                        venv_activated = True
                        break

                if not venv_activated:
                    commands.append(
                        "echo 'No virtual environment found, using system Python'"
                    )

                # Add script execution
                for script_name, command in scripts:
                    commands.append(f"echo 'Running {script_name}'")
                    # Convert Windows-style paths to Unix-style
                    unix_command = command.replace("\\", "/").replace(
                        "python", "python3"
                    )
                    commands.append(unix_command)

                commands.append("echo")
                commands.append("echo 'All selected scripts executed.'")
                commands.append("read -p 'Press Enter to close...'")

                full_command = " && ".join(commands)

                if system == "Darwin":  # macOS
                    subprocess.Popen(
                        [
                            "osascript",
                            "-e",
                            f'tell application "Terminal" to do script "{full_command}"',
                        ]
                    )
                else:  # Linux
                    terminals = [
                        ["gnome-terminal", "--", "bash", "-c", full_command],
                        ["xterm", "-e", f"bash -c '{full_command}'"],
                        ["konsole", "-e", "bash", "-c", full_command],
                        ["xfce4-terminal", "-e", f"bash -c '{full_command}'"],
                    ]

                    terminal_opened = False
                    for terminal_cmd in terminals:
                        try:
                            subprocess.Popen(terminal_cmd)
                            terminal_opened = True
                            break
                        except FileNotFoundError:
                            continue

                    if not terminal_opened:
                        QMessageBox.warning(
                            self,
                            "Terminal Not Found",
                            "No supported terminal emulator found. Please install gnome-terminal, xterm, konsole, or xfce4-terminal.",
                        )

            else:
                QMessageBox.warning(
                    self,
                    "Unsupported OS",
                    f"Operating system '{system}' is not supported.",
                )

        except Exception as e:
            QMessageBox.critical(
                self, "Launch Error", f"Failed to launch scripts: {str(e)}"
            )

    def create_config_creator_tab(self):
        # Main widget and layout for this tab
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)

        # Create a scroll area to hold all the settings
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        # Container widget for the form layout inside the scroll area
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        scroll_area.setWidget(form_container)

        # --- General Settings ---
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout()
        self.cfg_scene_name = QLineEdit("my_scene")
        self.cfg_scene_name.setToolTip("Enter the name of the scene for configuration")

        self.cfg_scene_name.setPlaceholderText("Enter scene name")  # Placeholder text
        self.cfg_scene_name.setFixedWidth(300)  # Set fixed width for better alignment
        self.cfg_volume_resolution = QSpinBox(
            minimum=32, maximum=2048, value=512, singleStep=32
        )
        self.cfg_volume_resolution.setFixedWidth(
            80
        )  # Set fixed width for better alignment
        self.cfg_volume_resolution.setToolTip(
            "Set the resolution for the volume grid.\nThis will determine the size of the 3D grid used for the scene."
        )
        self.cfg_remove_upper_aabb = QCheckBox(checked=True)
        self.cfg_remove_upper_aabb.setToolTip(
            "Remove points above the AABB from transform.json during volume slicing."
        )

        self.cfg_remove_below_aabb = QCheckBox(checked=False)
        self.cfg_remove_below_aabb.setToolTip(
            "Remove points below the AABB from transform.json during volume slicing."
        )
        general_layout.addRow("Scene Name:", self.cfg_scene_name)
        general_layout.addRow("Volume Resolution:", self.cfg_volume_resolution)
        general_layout.addRow("Remove Upper AABB:", self.cfg_remove_upper_aabb)
        general_layout.addRow("Remove Below AABB:", self.cfg_remove_below_aabb)
        general_group.setLayout(general_layout)
        form_layout.addWidget(general_group)

        # --- Image Resizing ---
        resize_group = QGroupBox("Image Resizing")
        resize_layout = QFormLayout()

        # Add checkbox for enabling/disabling image resizing
        self.cfg_resize_enabled = QCheckBox("Enable Image Resizing", checked=True)
        self.cfg_resize_enabled.setToolTip(
            "Resize images if checked; otherwise, just copy the original images to the tmp folder."
        )
        resize_layout.addRow(self.cfg_resize_enabled)

        self.cfg_resize_w = QSpinBox(minimum=128, maximum=4096, value=512)
        self.cfg_resize_h = QSpinBox(minimum=128, maximum=4096, value=512)

        resize_hbox = QHBoxLayout()
        resize_hbox.addWidget(QLabel("W"))
        resize_hbox.addWidget(self.cfg_resize_w)
        resize_hbox.addWidget(QLabel("H"))
        resize_hbox.addWidget(self.cfg_resize_h)
        resize_hbox.setAlignment(Qt.AlignLeft)
        resize_layout.addRow("New Size (W x H):", resize_hbox)
        resize_group.setLayout(resize_layout)
        form_layout.addWidget(resize_group)

        # --- Scene Transformation (AABB) ---
        transform_group = QGroupBox("Scene Transformation (AABB)")
        transform_layout = QFormLayout()
        # self.cfg_rotation = QCheckBox(checked=True)
        # self.cfg_angles = [QDoubleSpinBox(minimum=-360, maximum=360, value=0) for _ in range(3)]
        self.cfg_shift = [
            QDoubleSpinBox(minimum=-5, maximum=5, value=0, singleStep=0.1)
            for _ in range(3)
        ]
        self.cfg_scale = QDoubleSpinBox(
            minimum=0.1, maximum=10.0, value=0.9, singleStep=0.05
        )
        self.cfg_scale.setFixedWidth(80)  # Set fixed width for better alignment
        self.cfg_scale.setToolTip(
            "Scale the scene uniformly.\n"
            "This can help fit the scene within the Cube."
        )
        angles_hbox = QHBoxLayout()
        # for spinbox in self.cfg_angles: angles_hbox.addWidget(spinbox)
        shift_hbox = QHBoxLayout()
        for spinbox in self.cfg_shift:
            spinbox.setToolTip(
                "Shift the scene along the x, y, and z axes.\n"
                "This can help center the scene within the Cube."
            )
            spinbox.setFixedWidth(80)  # Set fixed width for better alignment
            shift_hbox.addWidget(spinbox)

        self.aabb_visualize = QCheckBox(checked=False)
        self.aabb_visualize.setToolTip(
            "Visualize poses, AABB, and points using colmap2nerf; display the volume in post_process_vol.py."
        )  # transform_layout.addRow("Enable Rotation:", self.cfg_rotation)
        # transform_layout.addRow("Angles (x, y, z):", angles_hbox)
        transform_layout.addRow("Shift (x, y, z):", shift_hbox)
        transform_layout.addRow("Scale:", self.cfg_scale)
        transform_layout.addRow("Visualize AABB:", self.aabb_visualize)

        # --- Outlier & Bounding Box Settings ---
        outlier_group = QGroupBox("Outlier & Bounding Box Settings")
        outlier_layout = QFormLayout()

        # target_retention
        self.cfg_target_retention = QDoubleSpinBox(
            minimum=0.0, maximum=1.0, singleStep=0.01, value=0.95
        )
        self.cfg_target_retention.setFixedWidth(
            80
        )  # Set fixed width for better alignment
        self.cfg_target_retention.setDecimals(3)
        self.cfg_target_retention.setToolTip(
            "Fraction of points to retain after outlier filtering (e.g., 0.95 keeps 95% of points)."
        )
        outlier_layout.addRow("Target Retention:", self.cfg_target_retention)

        # outlier_nb_neighbors
        self.cfg_outlier_nb_neighbors = QSpinBox(minimum=1, maximum=100, value=20)
        self.cfg_outlier_nb_neighbors.setToolTip(
            "Number of nearest neighbors used in statistical outlier removal."
        )
        outlier_layout.addRow("Outlier Neighbors:", self.cfg_outlier_nb_neighbors)
        self.cfg_outlier_nb_neighbors.setFixedWidth(
            80
        )  # Set fixed width for better alignment

        # outlier_std_ratio
        self.cfg_outlier_std_ratio = QDoubleSpinBox(
            minimum=0.1, maximum=10.0, singleStep=0.1, value=2.0
        )
        self.cfg_outlier_std_ratio.setFixedWidth(
            80
        )  # Set fixed width for better alignment
        self.cfg_outlier_std_ratio.setToolTip(
            "Threshold factor for standard deviation used in outlier filtering. Higher values remove fewer points."
        )
        outlier_layout.addRow("Outlier Std Ratio:", self.cfg_outlier_std_ratio)

        # percentile_bbox
        self.cfg_percentile_lower = QDoubleSpinBox(
            minimum=0.0, maximum=100.0, singleStep=0.5, value=1.0
        )
        self.cfg_percentile_upper = QDoubleSpinBox(
            minimum=0.0, maximum=100.0, singleStep=0.5, value=99.0
        )
        self.cfg_percentile_padding = QDoubleSpinBox(
            minimum=0.0, maximum=0.5, singleStep=0.01, value=0.05
        )

        bbox_hbox = QHBoxLayout()
        bbox_hbox.addWidget(QLabel("Lower"))
        bbox_hbox.addWidget(self.cfg_percentile_lower)
        bbox_hbox.addWidget(QLabel("Upper"))
        bbox_hbox.addWidget(self.cfg_percentile_upper)
        bbox_hbox.addWidget(QLabel("Padding"))
        bbox_hbox.addWidget(self.cfg_percentile_padding)
        bbox_hbox.setAlignment(Qt.AlignLeft)

        outlier_layout.addRow("Percentile BBox:", bbox_hbox)
        outlier_group.setLayout(outlier_layout)

        transform_group.setLayout(transform_layout)
        form_layout.addWidget(transform_group)
        form_layout.addWidget(outlier_group)

        # --- Training Options ---
        train_group = QGroupBox("Training Options")
        train_layout = QFormLayout()
        self.cfg_batch_size = QSpinBox(
            minimum=1024, maximum=65536, value=4096, singleStep=1024
        )
        self.cfg_batch_size.setFixedWidth(100)  # Set fixed width for better alignment

        self.cfg_num_epochs = QSpinBox(minimum=1, maximum=100, value=1)
        self.cfg_num_epochs.setFixedWidth(100)  # Set fixed width for better alignment

        self.cfg_lr = QDoubleSpinBox(
            minimum=1e-6, maximum=1e-2, value=0.0005, singleStep=1e-4, decimals=6
        )  # 0.0005
        self.cfg_lr.setValue(0.0005)  # Set default learning rate
        self.cfg_lr.setFixedWidth(100)  # Set fixed width for better alignment

        train_layout.addRow("Batch Size:", self.cfg_batch_size)
        train_layout.addRow("Number of Epochs:", self.cfg_num_epochs)
        train_layout.addRow("Learning Rate:", self.cfg_lr)
        train_group.setLayout(train_layout)
        form_layout.addWidget(train_group)

        # --- Model Options ---
        model_group = QGroupBox("Model Options")
        model_layout = QFormLayout()
        self.cfg_ngp = QCheckBox(checked=True)
        self.cfg_ngp.setToolTip(
            "Use NGP (Neural Graphics Primitives) for training.\n"
            "If unchecked, will use a standard NeRF model.\n"
            "NGP is recommended for faster training and better performance."
        )
        self.N_samples = QSpinBox(minimum=32, maximum=1028, value=64, singleStep=1)
        self.N_samples.setToolTip(
            "Number of samples per ray during training.\n" "Stratified sampling."
        )
        self.N_samples.setFixedWidth(80)  # Set fixed width for better alignment

        self.N_importance = QSpinBox(minimum=32, maximum=1028, value=128, singleStep=1)
        self.N_importance.setFixedWidth(80)
        self.N_importance.setToolTip(
            "Number of importance samples per ray during training.\n"
            "This is used for adaptive sampling to focus on areas with high detail."
        )

        self.cfg_hidden_dim = QSpinBox(
            minimum=32, maximum=512, value=256, singleStep=32
        )
        self.cfg_hidden_dim.setFixedWidth(80)  # Set fixed width for better alignment

        self.cfg_pos_L = QSpinBox(minimum=4, maximum=16, value=10)
        self.cfg_pos_L.setFixedWidth(80)
        self.cfg_pos_L.setToolTip(
            "Length of positional embeddings for the vanilla nerf.\n"
            "This controls how much detail is captured in the position of rays."
        )

        self.cfg_dir_L = QSpinBox(minimum=2, maximum=10, value=4)
        self.cfg_dir_L.setFixedWidth(80)
        self.cfg_dir_L.setToolTip(
            "Length of directional embeddings for the vanilla nerf.\n"
            "This controls how much detail is captured in the direction of rays."
        )

        model_layout.addRow("Use NGP:", self.cfg_ngp)
        model_layout.addRow("N_samples:", self.N_samples)
        model_layout.addRow("N_importance:", self.N_importance)
        model_layout.addRow("Hidden Dimensions:", self.cfg_hidden_dim)
        model_layout.addRow("Position Embeddings (L):", self.cfg_pos_L)
        model_layout.addRow("Direction Embeddings (L):", self.cfg_dir_L)
        model_group.setLayout(model_layout)
        form_layout.addWidget(model_group)

        form_layout.addStretch()

        # --- Preview and Save ---
        preview_group = QGroupBox("Preview & Save")
        preview_layout = QVBoxLayout()
        self.config_preview = QTextEdit()
        self.config_preview.setReadOnly(True)
        self.config_preview.setFontFamily("Courier")
        self.config_preview.setMinimumHeight(200)

        # Button layout for Load and Save
        button_layout = QHBoxLayout()

        load_button = QPushButton("Load Config from YAML")
        load_button.clicked.connect(self.load_config_yaml)
        load_button.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }"
        )

        save_button = QPushButton("Save Config to YAML")
        save_button.clicked.connect(self.save_config_yaml)
        save_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )

        button_layout.addWidget(load_button)
        button_layout.addWidget(save_button)

        preview_layout.addLayout(button_layout)
        preview_layout.addWidget(self.config_preview)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # Connect all widgets to the update preview function
        self.cfg_scene_name.textChanged.connect(self.update_config_preview)
        self.cfg_volume_resolution.valueChanged.connect(self.update_config_preview)
        self.cfg_remove_upper_aabb.stateChanged.connect(self.update_config_preview)
        self.cfg_remove_below_aabb.stateChanged.connect(self.update_config_preview)
        self.cfg_resize_enabled.stateChanged.connect(self.update_config_preview)
        self.cfg_resize_w.valueChanged.connect(self.update_config_preview)
        self.cfg_resize_h.valueChanged.connect(self.update_config_preview)
        self.aabb_visualize.stateChanged.connect(self.update_config_preview)
        # self.cfg_rotation.stateChanged.connect(self.update_config_preview)
        # for w in self.cfg_angles + self.cfg_shift: w.valueChanged.connect(self.update_config_preview)

        for w in self.cfg_shift:
            w.valueChanged.connect(self.update_config_preview)
        self.cfg_scale.valueChanged.connect(self.update_config_preview)
        self.cfg_batch_size.valueChanged.connect(self.update_config_preview)
        self.cfg_num_epochs.valueChanged.connect(self.update_config_preview)
        self.cfg_lr.valueChanged.connect(self.update_config_preview)
        self.cfg_ngp.stateChanged.connect(self.update_config_preview)
        self.N_samples.valueChanged.connect(self.update_config_preview)
        self.N_importance.valueChanged.connect(self.update_config_preview)
        self.cfg_hidden_dim.valueChanged.connect(self.update_config_preview)
        self.cfg_pos_L.valueChanged.connect(self.update_config_preview)
        self.cfg_dir_L.valueChanged.connect(self.update_config_preview)

        # Connect outlier and bbox settings
        self.cfg_target_retention.valueChanged.connect(self.update_config_preview)
        self.cfg_outlier_nb_neighbors.valueChanged.connect(self.update_config_preview)
        self.cfg_outlier_std_ratio.valueChanged.connect(self.update_config_preview)
        self.cfg_percentile_lower.valueChanged.connect(self.update_config_preview)
        self.cfg_percentile_upper.valueChanged.connect(self.update_config_preview)
        self.cfg_percentile_padding.valueChanged.connect(self.update_config_preview)

        # Initial preview update
        self.update_config_preview()

        return tab_widget

    def generate_config_dict(self):
        scene = self.cfg_scene_name.text()
        config = {
            "scene_name": scene,
            "output_dir": f"./outputs/{scene}",
            "checkpoint_dir": f"${{output_dir}}/checkpoints",
            "transforms_json": f"transforms_{scene}.json",
            "rays_file": f"{scene}_ray_data.npz",
            "volume_resolution": self.cfg_volume_resolution.value(),
            "remove_upper_aabb": self.cfg_remove_upper_aabb.isChecked(),
            "remove_below_aabb": self.cfg_remove_below_aabb.isChecked(),
            # "visualize": False,
            "visualize": self.aabb_visualize.isChecked(),
            "image_dir_resize": f"./data/{scene}/images",
            "workspace": f"tmp/{scene}",
            "image_dir": "${workspace}/images",
            "tmp_image_dir": f"tmp/{scene}/images",
            "resize_images": self.cfg_resize_enabled.isChecked(),
            "newSize": [self.cfg_resize_w.value(), self.cfg_resize_h.value()],
            #'rotation': self.cfg_rotation.isChecked(),
            #'rotation_initial': None,
            #'rot_order': [0, 1, 2],
            #'angles': [w.value() for w in self.cfg_angles],
            "shift": [w.value() for w in self.cfg_shift],
            "scale": self.cfg_scale.value(),
            # 'target_retention': 0.95,
            # 'outlier_nb_neighbors': 20,
            # 'outlier_std_ratio': 2.0,
            # 'percentile_bbox': {'lower': 1.0, 'upper': 99.0, 'padding': 0.05},
            "target_retention": self.cfg_target_retention.value(),
            "outlier_nb_neighbors": self.cfg_outlier_nb_neighbors.value(),
            "outlier_std_ratio": self.cfg_outlier_std_ratio.value(),
            "percentile_bbox": {
                "lower": self.cfg_percentile_lower.value(),
                "upper": self.cfg_percentile_upper.value(),
                "padding": self.cfg_percentile_padding.value(),
            },
            "aabb_adjust": {"aabb_min": [0, 0, 0], "aabb_max": [0, 0, 0]},
            "checkpoint": "${checkpoint_dir}/nerf_final.pth",
            "volume_output_path": "${output_dir}/volume.pth",
            "aabb_slice": True,
            "sliced_vol_path": "${output_dir}/volume_sliced.pth",
            "colmap_dir": f"./data/{scene}",
            "batch_size": self.cfg_batch_size.value(),
            "num_epochs": self.cfg_num_epochs.value(),
            "lr": self.cfg_lr.value(),
            "ngp": self.cfg_ngp.isChecked(),
            "nerf_type": "large",
            "hidden_dim": self.cfg_hidden_dim.value(),
            "pos_L": self.cfg_pos_L.value(),
            "dir_L": self.cfg_dir_L.value(),
            "N_samples": self.N_samples.value(),
            "N_importance": self.N_importance.value(),
            "white_bg": False,
            "chunk_size": 32768,
            "use_memmap": False,
        }
        return config

    def update_config_preview(self):
        config_dict = self.generate_config_dict()
        try:
            # Use sort_keys=False to maintain order
            yaml_str = yaml.dump(config_dict, sort_keys=False, indent=2)
            self.config_preview.setText(yaml_str)
        except Exception as e:
            self.config_preview.setText(f"Error generating YAML: {e}")

    def load_config_yaml(self):
        """Load configuration from a YAML file and update GUI values"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Config File", "cfg/", "YAML Files (*.yml *.yaml)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

            # Block signals to prevent triggering update_config_preview multiple times
            self.block_all_signals(True)

            # Update GUI values from loaded config
            if "scene_name" in config:
                self.cfg_scene_name.setText(str(config["scene_name"]))

            if "volume_resolution" in config:
                self.cfg_volume_resolution.setValue(config["volume_resolution"])

            if "remove_upper_aabb" in config:
                self.cfg_remove_upper_aabb.setChecked(config["remove_upper_aabb"])

            if "remove_below_abb" in config:
                self.cfg_remove_below_aabb.setChecked(config["remove_below_abb"])

            # Image resizing settings
            if "resize_images" in config:
                self.cfg_resize_enabled.setChecked(config["resize_images"])

            if (
                "newSize" in config
                and isinstance(config["newSize"], list)
                and len(config["newSize"]) >= 2
            ):
                self.cfg_resize_w.setValue(config["newSize"][0])
                self.cfg_resize_h.setValue(config["newSize"][1])

            # Scene transformation
            if (
                "shift" in config
                and isinstance(config["shift"], list)
                and len(config["shift"]) >= 3
            ):
                for i, spinbox in enumerate(self.cfg_shift):
                    if i < len(config["shift"]):
                        spinbox.setValue(config["shift"][i])

            if "scale" in config:
                self.cfg_scale.setValue(config["scale"])

            # Outlier and bounding box settings
            if "target_retention" in config:
                self.cfg_target_retention.setValue(config["target_retention"])

            if "outlier_nb_neighbors" in config:
                self.cfg_outlier_nb_neighbors.setValue(config["outlier_nb_neighbors"])

            if "outlier_std_ratio" in config:
                self.cfg_outlier_std_ratio.setValue(config["outlier_std_ratio"])

            if "percentile_bbox" in config and isinstance(
                config["percentile_bbox"], dict
            ):
                bbox = config["percentile_bbox"]
                if "lower" in bbox:
                    self.cfg_percentile_lower.setValue(bbox["lower"])
                if "upper" in bbox:
                    self.cfg_percentile_upper.setValue(bbox["upper"])
                if "padding" in bbox:
                    self.cfg_percentile_padding.setValue(bbox["padding"])

            # Training options
            if "batch_size" in config:
                self.cfg_batch_size.setValue(config["batch_size"])

            if "num_epochs" in config:
                self.cfg_num_epochs.setValue(config["num_epochs"])

            if "lr" in config:
                self.cfg_lr.setValue(config["lr"])

            # Model options
            if "ngp" in config:
                self.cfg_ngp.setChecked(config["ngp"])

            if "N_samples" in config:
                self.N_samples.setValue(config["N_samples"])

            if "N_importance" in config:
                self.N_importance.setValue(config["N_importance"])

            if "hidden_dim" in config:
                self.cfg_hidden_dim.setValue(config["hidden_dim"])

            if "pos_L" in config:
                self.cfg_pos_L.setValue(config["pos_L"])

            if "dir_L" in config:
                self.cfg_dir_L.setValue(config["dir_L"])

            # Re-enable signals and update preview
            self.block_all_signals(False)
            self.update_config_preview()

            QMessageBox.information(
                self, "Success", f"Configuration loaded from:\n{file_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load configuration file:\n{e}"
            )

    def block_all_signals(self, block):
        """Helper method to block/unblock signals from all config widgets"""
        widgets = [
            self.cfg_scene_name,
            self.cfg_volume_resolution,
            self.cfg_remove_upper_aabb,
            self.cfg_remove_below_aabb,
            self.cfg_resize_enabled,
            self.cfg_resize_w,
            self.cfg_resize_h,
            self.cfg_scale,
            self.cfg_target_retention,
            self.cfg_outlier_nb_neighbors,
            self.cfg_outlier_std_ratio,
            self.cfg_percentile_lower,
            self.cfg_percentile_upper,
            self.cfg_percentile_padding,
            self.cfg_batch_size,
            self.cfg_num_epochs,
            self.cfg_lr,
            self.cfg_ngp,
            self.N_samples,
            self.N_importance,
            self.cfg_hidden_dim,
            self.cfg_pos_L,
            self.cfg_dir_L,
        ]

        # Add shift spinboxes
        widgets.extend(self.cfg_shift)

        for widget in widgets:
            widget.blockSignals(block)

    def save_config_yaml(self):
        config_dict = self.generate_config_dict()
        scene_name = config_dict.get("scene_name", "config")

        # Create cfg directory if it doesn't exist
        cfg_dir = Path("cfg")
        cfg_dir.mkdir(exist_ok=True)

        default_path = cfg_dir / f"{scene_name}.yml"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Config File", str(default_path), "YAML Files (*.yml *.yaml)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, sort_keys=False, indent=2)
            QMessageBox.information(
                self, "Success", f"Configuration saved to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save configuration file:\n{e}"
            )

    def closeEvent(self, event):
        self.threadpool.clear()
        self.threadpool.waitForDone(-1)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_cyan.xml")
    viewer = VispyViewer()
    viewer.show()
    sys.exit(app.exec())
