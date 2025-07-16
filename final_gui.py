from pathlib import Path
import sys
import subprocess
import platform
import numpy as np
import torch

import os
import yaml

from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import (
    QMessageBox,
    QApplication,
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
    QSlider,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QGroupBox,
    QGridLayout,
    QScrollArea,
    QTextEdit,
)
from vispy import scene
from PySide6.QtGui import QIcon


class VispyViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeRF Pipeline Script Launcher")
        self.setGeometry(100, 100, 1000, 600)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create three tabs
        self.tab1 = self.create_script_launcher_tab()
        self.tab2 = QWidget()  # Volume visualization will go here

        self.tabs.addTab(self.tab1, "Set Configurtaion File")
        self.tabs.addTab(self.tab2, "Volume Viewer")

        # Optional content for Tab 1 and Tab 2
        # self.setup_config_tab()
        # Set up Tab 3 with visualization UI
        self.setup_volume_tab()

        # State tracking
        self.current_visual = None
        self.current_points = None
        self.current_colors = None
        self.point_size = 1

    def load_volume(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Volume File", "", "Torch Files (*.pth *.pt)"
        )
        if not file_path:
            return

        # Remove previous visual
        if self.current_visual is not None:
            self.current_visual.parent = None
            self.current_visual = None

        # Load volume
        volume = torch.load(file_path, weights_only=False)

        if "points_normalized" in volume:
            self.current_points = volume["points_normalized"]
            self.current_colors = volume["rgbs"]
        else:
            curr_volume = volume["occupancy_volume"]
            rgb_volume = volume["rgb_volume"]
            rgb_volume = rgb_volume / rgb_volume.max()
            points = np.argwhere(curr_volume.numpy())
            rgbs = rgb_volume[points[:, 0], points[:, 1], points[:, 2]].numpy()

            res = curr_volume.shape[0]
            points = (points / (res - 1)) * 2 - 1

            self.current_points = points
            self.current_colors = rgbs

        # Reset sliders to default
        self.z_min_input.blockSignals(True)
        self.z_max_input.blockSignals(True)
        self.z_min_input.setValue(-1.0)
        self.z_max_input.setValue(1.0)
        self.z_min_input.blockSignals(False)
        self.z_max_input.blockSignals(False)

        # Create scatter
        self.current_visual = scene.visuals.Markers()
        self.current_visual.set_data(
            self.current_points,
            face_color=self.current_colors,
            edge_color=self.current_colors,
            size=self.point_size,
        )
        self.current_visual.set_gl_state(blend=False, depth_test=True)
        self.view.add(self.current_visual)

        # Use update_z_slice to apply default range and display
        self.update_z_slice()

        self.canvas.show()

    def setup_volume_tab(self):
        main_layout = QHBoxLayout()  # <-- changed from QVBoxLayout
        self.tab2.setLayout(main_layout)

        # Left control panel
        controls_layout = QVBoxLayout()

        load_button = QPushButton("Load Volume")
        load_button.setFixedWidth(180)
        load_button.clicked.connect(self.load_volume)
        controls_layout.addWidget(load_button, alignment=Qt.AlignLeft)

        delete_button = QPushButton("Delete Volume")
        delete_button.setFixedWidth(180)
        delete_button.clicked.connect(self.delete_volume)
        controls_layout.addWidget(delete_button, alignment=Qt.AlignLeft)

        volume_control_layout = QHBoxLayout()
        decrease_button = QPushButton("−")
        increase_button = QPushButton("+")
        self.size_label = QLabel("Point Size: 1")

        decrease_button.setFixedSize(45, 45)
        increase_button.setFixedSize(45, 45)

        decrease_button.clicked.connect(self.decrease_point_size)
        increase_button.clicked.connect(self.increase_point_size)

        volume_control_layout.addWidget(decrease_button)
        volume_control_layout.addWidget(increase_button)
        volume_control_layout.addWidget(self.size_label)
        volume_control_layout.addStretch()

        self.origin_checkbox = QCheckBox("Show Origin")
        self.origin_checkbox.setChecked(True)
        self.origin_checkbox.stateChanged.connect(self.toggle_origin_visibility)

        self.unitcube_checkbox = QCheckBox("Show Unit Cube")
        self.unitcube_checkbox.setChecked(True)
        self.unitcube_checkbox.stateChanged.connect(self.toggle_unit_cube_visibility)

        z_slice_layout = QFormLayout()
        self.z_min_input = QDoubleSpinBox()
        self.z_max_input = QDoubleSpinBox()

        self.z_min_input.setRange(-1.0, 1.0)
        self.z_max_input.setRange(-1.0, 1.0)

        self.z_min_input.setFixedWidth(80)
        self.z_max_input.setFixedWidth(80)
        self.z_min_input.setSingleStep(0.1)
        self.z_max_input.setSingleStep(0.1)
        self.z_min_input.setValue(-1.0)
        self.z_max_input.setValue(1.0)
        self.z_min_input.valueChanged.connect(self.update_z_slice)
        self.z_max_input.valueChanged.connect(self.update_z_slice)
        z_slice_layout.addRow("Z Min Slice:", self.z_min_input)
        z_slice_layout.addRow("Z Max Slice:", self.z_max_input)

        # Add to control panel
        controls_layout.addLayout(volume_control_layout)
        controls_layout.addWidget(self.unitcube_checkbox)
        controls_layout.addWidget(self.origin_checkbox)
        controls_layout.addLayout(z_slice_layout)
        controls_layout.addStretch()

        # VisPy canvas setup
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=45, azimuth=120, elevation=30, distance=4.5
        )
        self.axis = scene.visuals.XYZAxis(parent=self.view.scene)

        # Unit cube
        cube_lines = [
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
        cube_lines = torch.tensor(cube_lines, dtype=torch.float32)
        self.unit_cube = scene.visuals.Line(
            pos=cube_lines,
            color="red",
            method="gl",
            width=4,
            parent=self.view.scene,
        )

        canvas_widget = self.canvas.native
        canvas_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add to main layout
        main_layout.addWidget(canvas_widget, stretch=1)
        main_layout.addLayout(controls_layout, stretch=0)

    def update_z_slice(self):
        if self.current_points is None:
            return

        z_min = self.z_min_input.value()
        z_max = self.z_max_input.value()

        if z_min > z_max:
            return  # Optional: show a warning or auto-correct

        mask = (self.current_points[:, 2] >= z_min) & (
            self.current_points[:, 2] <= z_max
        )
        filtered_points = self.current_points[mask]
        filtered_colors = self.current_colors[mask]

        self.current_visual.set_data(
            filtered_points,
            face_color=filtered_colors,
            edge_color=filtered_colors,
            size=self.point_size,
        )
        self.canvas.update()

    def toggle_origin_visibility(self, state):
        if self.axis is not None:
            self.axis.visible = bool(state)

    def toggle_unit_cube_visibility(self, state):
        if self.unit_cube is not None:
            self.unit_cube.visible = bool(state)

    def delete_volume(self):
        if self.current_visual is not None:
            self.current_visual.parent = None  # Remove from scene
            self.current_visual = None
            self.current_points = None
            self.current_colors = None
            self.canvas.update()

    def update_point_size(self):
        self.size_label.setText(f"Point Size: {self.point_size}")
        if self.current_visual is not None and self.current_points is not None:
            self.current_visual.set_data(
                self.current_points,
                face_color=self.current_colors,
                edge_color=self.current_colors,
                size=self.point_size,
            )
            self.canvas.update()

    def increase_point_size(self):
        if self.point_size < 6:
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
                "description": "Resize input images",
            },
            "run_colmap.py": {
                "command": "python scripts/run_colmap.py --workspace ./data/{scene}/",
                "description": "Run COLMAP reconstruction",
            },
            "colmap2nerf.py": {
                "command": "python scripts/colmap2nerf.py --cfg_path cfg/{scene}.yml",
                "description": "Transform scene alignment and compute AABB",
            },
            "precompute_rays.py": {
                "command": "python scripts/precompute_rays.py --cfg_path cfg/{scene}.yml",
                "description": "Precompute rays and save as NPZ",
            },
            "train.py": {
                "command": "python scripts/train.py --cfg_path cfg/{scene}.yml",
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



from qt_material import apply_stylesheet
if __name__ == "__main__":

    app = QApplication(sys.argv)

    # https://github.com/UN-GCPDS/qt-material?tab=readme-ov-file#install
    apply_stylesheet(app, theme="dark_cyan.xml")
    viewer = VispyViewer()
    viewer.show()
    sys.exit(app.exec())
