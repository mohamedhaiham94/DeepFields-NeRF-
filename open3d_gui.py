import torch
import numpy as np
from vispy import scene, app
import threading
import time
import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog, messagebox


class PointCloudMinimalGUI:
    def __init__(self):
        self.volume = None
        self.points = None
        self.colors = None
        self.current_file_path = None

        self.current_point_size = 2
        self.canvas = None
        self.view = None
        self.scatter = None
        self.vis_thread = None
        self.should_update = False
        self._stop_visualizer = False

        self.init_gui()

    def init_gui(self):
        self.root = Tk()
        self.root.title("Point Cloud Controller")
        self.root.geometry("350x850")
        self.root.configure(bg="#2b2b2b")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#2b2b2b", foreground="white")
        style.configure("TButton", background="#4CAF50", foreground="white")

        main_frame = Frame(self.root, bg="#2b2b2b", padx=20, pady=20)
        main_frame.pack(fill=BOTH, expand=True)

        title_label = Label(
            main_frame,
            text="Point Cloud Controls",
            font=("Arial", 16, "bold"),
            bg="#2b2b2b",
            fg="white",
        )
        title_label.pack(pady=(0, 20))

        file_frame = Frame(main_frame, bg="#2b2b2b")
        file_frame.pack(fill=X, pady=10)

        file_label = Label(file_frame, text="Volume File:", bg="#2b2b2b", fg="white", font=("Arial", 12, "bold"))
        file_label.pack()

        self.file_path_var = StringVar(value="No file selected")
        self.file_path_label = Label(file_frame, textvariable=self.file_path_var, bg="#2b2b2b", fg="#cccccc", wraplength=300, justify=LEFT)
        self.file_path_label.pack(pady=5)

        file_button_frame = Frame(file_frame, bg="#2b2b2b")
        file_button_frame.pack(fill=X, pady=5)

        select_file_button = Button(file_button_frame, text="Select Volume File", command=self.select_file,
                                    bg="#FF9800", fg="white", font=("Arial", 10, "bold"), relief=FLAT, padx=20, pady=8)
        select_file_button.pack(side=LEFT, padx=(0, 5))

        load_button = Button(file_button_frame, text="Load Volume", command=self.load_volume,
                             bg="#9C27B0", fg="white", font=("Arial", 10, "bold"), relief=FLAT, padx=20, pady=8)
        load_button.pack(side=LEFT)

        Frame(main_frame, height=2, bg="#555555").pack(fill=X, pady=20)

        size_frame = Frame(main_frame, bg="#2b2b2b")
        size_frame.pack(fill=X, pady=10)

        size_label = Label(size_frame, text="Point Size:", bg="#2b2b2b", fg="white", font=("Arial", 12, "bold"))
        size_label.pack()

        self.point_size_var = IntVar(value=self.current_point_size)
        self.point_size_slider = Scale(size_frame, from_=1, to=10, orient=HORIZONTAL, variable=self.point_size_var,
                                       command=self.on_point_size_changed, bg="#2b2b2b", fg="white",
                                       highlightbackground="#2b2b2b", troughcolor="#555555", activebackground="#4CAF50")
        self.point_size_slider.pack(fill=X, pady=5)

        self.size_value_label = Label(size_frame, text=f"Current: {self.current_point_size}", bg="#2b2b2b", fg="#cccccc")
        self.size_value_label.pack()

        button_frame = Frame(main_frame, bg="#2b2b2b")
        button_frame.pack(fill=X, pady=20)

        self.show_button = Button(button_frame, text="Show Point Cloud", command=self.show_pointcloud,
                                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), relief=FLAT, padx=20, pady=10, state=DISABLED)
        self.show_button.pack(fill=X, pady=5)

        self.update_button = Button(button_frame, text="Update Point Cloud", command=self.update_pointcloud,
                                    bg="#2196F3", fg="white", font=("Arial", 10, "bold"), relief=FLAT, padx=20, pady=10, state=DISABLED)
        self.update_button.pack(fill=X, pady=5)

        Button(button_frame, text="Close Point Cloud", command=self.close_visualizer,
               bg="#36d4f4", fg="white", font=("Arial", 10, "bold"), relief=FLAT, padx=20, pady=10).pack(fill=X, pady=5)

        Button(button_frame, text="Close GUI", command=self.on_closing,
               bg="#e53935", fg="white", font=("Arial", 10, "bold"), relief=FLAT, padx=20, pady=10).pack(fill=X, pady=5)

        info_frame = Frame(main_frame, bg="#2b2b2b")
        info_frame.pack(fill=X, pady=20)

        Label(info_frame, text="Instructions:", font=("Arial", 12, "bold"), bg="#2b2b2b", fg="white").pack()
        for instruction in [
            "1. Click 'Select Volume File' to choose a .pth file",
            "2. Click 'Load Volume' to load the selected file",
            "3. Click 'Show Point Cloud' to open visualizer",
            "4. Adjust point size with slider",
            "5. Click 'Update Point Cloud' to apply changes",
            "6. Use mouse in visualizer to rotate/zoom",
        ]:
            Label(info_frame, text=instruction, bg="#2b2b2b", fg="#999999", font=("Arial", 9)).pack(anchor=W)

        self.status_label = Label(main_frame, text="Status: Ready - Please select a volume file", bg="#2b2b2b", fg="#FFD700", font=("Arial", 10, "bold"))
        self.status_label.pack(pady=20)

    def select_file(self):
        try:
            filetypes = [("PyTorch files", "*.pth"), ("All files", "*.*")]
            filename = filedialog.askopenfilename(title="Select Volume File", filetypes=filetypes, initialdir=os.getcwd())
            if filename:
                self.current_file_path = filename
                display_path = filename if len(filename) <= 50 else "..." + filename[-47:]
                self.file_path_var.set(display_path)
                self.status_label.config(text="File selected. Click 'Load Volume' to load it.", fg="#FFD700")
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting file: {str(e)}")

    def load_volume(self):
        try:
            if not self.current_file_path or not os.path.exists(self.current_file_path):
                messagebox.showwarning("Warning", "Please select a valid file first!")
                return

            self.status_label.config(text="Loading volume...", fg="#FFD700")
            self.root.update()

            self.volume = torch.load(self.current_file_path, weights_only=False)

            if "points_normalized" in self.volume:
                self.points = self.volume["points_normalized"]
            elif "points" in self.volume:
                self.points = self.volume["points"]
            elif "occupancy_volume" in self.volume:
                volume = self.volume["occupancy_volume"]
                rgb_volume = self.volume["rgb_volume"] / self.volume["rgb_volume"].max()
                points = np.argwhere(volume.numpy())
                rgbs = rgb_volume[points[:, 0], points[:, 1], points[:, 2]].numpy()
                res = volume.shape[0]
                points = (points / (res - 1)) * 2 - 1
                self.points = points
                self.colors = rgbs
            else:
                raise ValueError("No points data found.")

            if "rgbs" in self.volume:
                self.colors = self.volume["rgbs"]
            elif "colors" in self.volume:
                self.colors = self.volume["colors"]
            elif "rgb_volume" in self.volume:
                pass
            else:
                raise ValueError("No color data found.")

            if self.colors.max() > 1.0:
                self.colors = self.colors / 255.0

            self.show_button.config(state=NORMAL)
            self.update_button.config(state=NORMAL)

            self.status_label.config(text=f"Volume loaded! {len(self.points):,} points", fg="#90EE90")

        except Exception as e:
            messagebox.showerror("Loading Error", str(e))
            self.status_label.config(text=f"Load error: {str(e)}", fg="#FF6B6B")

    def on_point_size_changed(self, value):
        self.current_point_size = int(value)
        self.size_value_label.config(text=f"Current: {self.current_point_size}")
        if self.points is not None:
            self.status_label.config(
                text=f"Point size set to {self.current_point_size} (click Update to apply)", fg="#FFD700"
            )

    def show_pointcloud(self):
        try:
            if self.points is None:
                messagebox.showwarning("Warning", "Please load a volume file first!")
                return

            if self.canvas is not None:
                self.close_visualizer()

            self.vis_thread = threading.Thread(target=self._run_visualizer, daemon=True)
            self.vis_thread.start()

            self.status_label.config(text="Point cloud visualizer opened", fg="#90EE90")

        except Exception as e:
            messagebox.showerror("Visualization Error", str(e))
            self.status_label.config(text=f"Show error: {str(e)}", fg="#FF6B6B")

    def _run_visualizer(self):
        try:
            self._stop_visualizer = False

            self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="black", show=True, title="Point Cloud Viewer", size=(800, 600))
            self.view = self.canvas.central_widget.add_view()

            self.scatter = scene.visuals.Markers()
            self.scatter.set_data(self.points, face_color=self.colors, edge_color=self.colors, size=int(self.current_point_size))
            self.scatter.set_gl_state(blend=False, depth_test=True)
            self.view.add(self.scatter)

            scene.visuals.XYZAxis(parent=self.view.scene)

            self.view.camera = scene.TurntableCamera(fov=45, azimuth=30, elevation=30, distance=2.0, center=(0, 0, 0))

            while not self._stop_visualizer:
                if self.should_update:
                    self.scatter.set_data(
                        self.points,
                        face_color=self.colors,
                        edge_color=self.colors,
                        size=int(self.current_point_size),
                    )
                    # app.call_on_main_thread(self.canvas.update)
                    self.should_update = False

                self.canvas.app.process_events()
                time.sleep(0.01)

            if self.canvas is not None:
                self.canvas.close()
                self.canvas = None
                self.view = None
                self.scatter = None

        except Exception as e:
            print(f"Error in visualizer thread: {e}")

    def update_pointcloud(self):
        try:
            if self.points is None:
                messagebox.showwarning("Warning", "Please load a volume file first!")
                return

            if self.canvas is not None:
                self.should_update = True
                self.status_label.config(
                    text=f"Point size updated to {self.current_point_size}", fg="#90EE90"
                )
            else:
                self.status_label.config(
                    text="No visualizer open. Click 'Show Point Cloud' first.", fg="#FFD700"
                )

        except Exception as e:
            messagebox.showerror("Update Error", str(e))
            self.status_label.config(text=f"Update error: {str(e)}", fg="#FF6B6B")

    def close_visualizer(self):
        try:
            if self.canvas is not None:
                self._stop_visualizer = True
                time.sleep(0.1)
                if self.vis_thread is not None:
                    self.vis_thread.join(timeout=2.0)
                    self.vis_thread = None
            self.status_label.config(text="Visualizer closed", fg="#90EE90")
        except Exception as e:
            messagebox.showerror("Close Error", str(e))
            self.status_label.config(text=f"Close error: {str(e)}", fg="#FF6B6B")

    def on_closing(self):
        self.close_visualizer()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    app = PointCloudMinimalGUI()
    app.run()


if __name__ == "__main__":
    main()
