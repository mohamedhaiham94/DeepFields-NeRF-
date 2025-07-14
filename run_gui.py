import torch
import numpy as np
import open3d as o3d
import threading
import time
import os
from tkinter import *
from tkinter import ttk
from tkinter import filedialog, messagebox

class PointCloudMinimalGUI:
    def __init__(self):
        # Initialize without loading data
        self.volume = None
        self.points = None
        self.colors = None
        self.pcd = None
        self.current_file_path = None
        
        # GUI Parameter
        self.current_point_size = 2
        self.vis = None
        self.vis_thread = None
        self.should_update = False
        self._stop_visualizer = False
        
        # GUI initialisieren
        self.init_gui()
        
    def init_gui(self):
        # Tkinter GUI erstellen
        self.root = Tk()
        self.root.title("Point Cloud Controller")
        self.root.geometry("350x850")
        self.root.configure(bg='#2b2b2b')
        
        # Style konfigurieren
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', background='#4CAF50', foreground='white')
        
        # Hauptframe
        main_frame = Frame(self.root, bg='#2b2b2b', padx=20, pady=20)
        main_frame.pack(fill=BOTH, expand=True)
    
        # Titel
        title_label = Label(main_frame, text="Point Cloud Controls", 
                           font=('Arial', 16, 'bold'), 
                           bg= '#2b2b2b', fg='white')
        title_label.pack(pady=(0, 20))
        
        # File Selection Frame
        file_frame = Frame(main_frame, bg='#2b2b2b')
        file_frame.pack(fill=X, pady=10)
        
        file_label = Label(file_frame, text="Volume File:", 
                          bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        file_label.pack()
        
        # File path display
        self.file_path_var = StringVar(value="No file selected")
        self.file_path_label = Label(file_frame, textvariable=self.file_path_var,
                                    bg='#2b2b2b', fg='#cccccc', 
                                    wraplength=300, justify=LEFT)
        self.file_path_label.pack(pady=5)
        
        # File selection buttons
        file_button_frame = Frame(file_frame, bg='#2b2b2b')
        file_button_frame.pack(fill=X, pady=5)
        
        select_file_button = Button(file_button_frame, text="Select Volume File",
                                   command=self.select_file,
                                   bg='#FF9800', fg='white',
                                   font=('Arial', 10, 'bold'),
                                   relief=FLAT, padx=20, pady=8)
        select_file_button.pack(side=LEFT, padx=(0, 5))
        
        load_button = Button(file_button_frame, text="Load Volume",
                           command=self.load_volume,
                           bg='#9C27B0', fg='white',
                           font=('Arial', 10, 'bold'),
                           relief=FLAT, padx=20, pady=8)
        load_button.pack(side=LEFT)
        
        # Separator
        separator = Frame(main_frame, height=2, bg='#555555')
        separator.pack(fill=X, pady=20)
        
        # Point Size Controls
        size_frame = Frame(main_frame, bg='#2b2b2b')
        size_frame.pack(fill=X, pady=10)
        
        size_label = Label(size_frame, text="Point Size:", 
                          bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        size_label.pack()
        
        # Point Size Slider
        self.point_size_var = IntVar(value=self.current_point_size)
        self.point_size_slider = Scale(size_frame, 
                                      from_=1, to=6, 
                                      orient=HORIZONTAL,
                                      variable=self.point_size_var,
                                      command=self.on_point_size_changed,
                                      bg='#2b2b2b', fg='white',
                                      highlightbackground='#2b2b2b',
                                      troughcolor='#555555',
                                      activebackground='#4CAF50')
        self.point_size_slider.pack(fill=X, pady=5)
        
        # Current Value Label
        self.size_value_label = Label(size_frame, 
                                     text=f"Current: {self.current_point_size}",
                                     bg='#2b2b2b', fg='#cccccc')
        self.size_value_label.pack()
        
        # Buttons Frame
        button_frame = Frame(main_frame, bg='#2b2b2b')
        button_frame.pack(fill=X, pady=20)
        
        # Show Point Cloud Button
        self.show_button = Button(button_frame, text="Show Point Cloud",
                                 command=self.show_pointcloud,
                                 bg='#4CAF50', fg='white',
                                 font=('Arial', 10, 'bold'),
                                 relief=FLAT, padx=20, pady=10,
                                 state=DISABLED)
        self.show_button.pack(fill=X, pady=5)
        
        # Update Point Cloud Button
        self.update_button = Button(button_frame, text="Update Point Cloud",
                                   command=self.update_pointcloud,
                                   bg='#2196F3', fg='white',
                                   font=('Arial', 10, 'bold'),
                                   relief=FLAT, padx=20, pady=10,
                                   state=DISABLED)
        self.update_button.pack(fill=X, pady=5)
        
        # Close Visualizer Button
        close_button = Button(button_frame, text="Close Visualizer",
                            command=self.close_visualizer,
                            bg='#f44336', fg='white',
                            font=('Arial', 10, 'bold'),
                            relief=FLAT, padx=20, pady=10)
        close_button.pack(fill=X, pady=5)
        
        # Info Frame
        info_frame = Frame(main_frame, bg='#2b2b2b')
        info_frame.pack(fill=X, pady=20)
        
        info_label = Label(info_frame, text="Instructions:",
                          font=('Arial', 12, 'bold'),
                          bg='#2b2b2b', fg='white')
        info_label.pack()
        
        instructions = [
            "1. Click 'Select Volume File' to choose a .pth file",
            "2. Click 'Load Volume' to load the selected file",
            "3. Click 'Show Point Cloud' to open visualizer",
            "4. Adjust point size with slider",
            "5. Click 'Update Point Cloud' to apply changes",
            "6. Use mouse in visualizer to rotate/zoom"
        ]
        
        for instruction in instructions:
            inst_label = Label(info_frame, text=instruction,
                             bg='#2b2b2b', fg='#999999',
                             font=('Arial', 9))
            inst_label.pack(anchor=W)
        
        # Status Label
        self.status_label = Label(main_frame, text="Status: Ready - Please select a volume file",
                                 bg='#2b2b2b', fg='#FFD700',
                                 font=('Arial', 10, 'bold'))
        self.status_label.pack(pady=20)
        
    def select_file(self):
        """Open file dialog to select volume file"""
        try:
            # Define file types
            filetypes = [
                ('PyTorch files', '*.pth'),
                ('All files', '*.*')
            ]
            
            # Open file dialog
            filename = filedialog.askopenfilename(
                title="Select Volume File",
                filetypes=filetypes,
                initialdir=os.getcwd()
            )
            
            if filename:
                self.current_file_path = filename
                # Display shortened path
                display_path = filename
                if len(display_path) > 50:
                    display_path = "..." + display_path[-47:]
                self.file_path_var.set(display_path)
                self.status_label.config(text="File selected. Click 'Load Volume' to load it.", 
                                       fg='#FFD700')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting file: {str(e)}")
            self.status_label.config(text=f"File selection error: {str(e)}", fg='#FF6B6B')
    
    def load_volume(self):
        """Load the selected volume file"""
        try:
            if not self.current_file_path:
                messagebox.showwarning("Warning", "Please select a file first!")
                return
            
            if not os.path.exists(self.current_file_path):
                messagebox.showerror("Error", "Selected file does not exist!")
                return
            
            # Update status
            self.status_label.config(text="Loading volume...", fg='#FFD700')
            self.root.update()
            
            # Load the volume
            self.volume = torch.load(self.current_file_path, weights_only=False)
            
            # Extract points and colors
            if "points_normalized" in self.volume:
                self.points = self.volume["points_normalized"]
            elif "points" in self.volume:
                self.points = self.volume["points"]
            elif "occupancy_volume" in self.volume:
                volume = self.volume["occupancy_volume"]  # shape: [512, 512, 512]
                rgb_volume = self.volume["rgb_volume"]  # shape: [512, 512, 512, 3]
                rgb_volume = rgb_volume / rgb_volume.max()
                points = np.argwhere(volume.numpy())
                rgbs = rgb_volume[points[:, 0], points[:, 1], points[:, 2]].numpy()  # [N, 3]

                # Normalize coordinates to [-1, 1]
                res = volume.shape[0]
                points = (points / (res - 1)) * 2 - 1  # [N, 3]
                
                self.points = points
                self.colors = rgbs
            else:
                raise ValueError("No 'points_normalized' or 'points' key found in volume data")
            
            if "rgbs" in self.volume:
                self.colors = self.volume["rgbs"]
            elif "colors" in self.volume:
                self.colors = self.volume["colors"]
            elif "rgb_volume" in self.volume:
                pass
            else:
                raise ValueError("No 'rgbs' or 'colors' key found in volume data")
            
            # Normalisiere Farben (falls in [0, 255])
            if self.colors.max() > 1.0:
                self.colors = self.colors / 255.0
            
            # Punktwolke erstellen
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
            self.pcd.colors = o3d.utility.Vector3dVector(self.colors)
            
            # Enable buttons
            self.show_button.config(state=NORMAL)
            self.update_button.config(state=NORMAL)
            
            # Update status
            num_points = len(self.points)
            self.status_label.config(text=f"Volume loaded successfully! {num_points:,} points", 
                                   fg='#90EE90')
            
        except Exception as e:
            error_msg = f"Error loading volume: {str(e)}"
            messagebox.showerror("Loading Error", error_msg)
            self.status_label.config(text=error_msg, fg='#FF6B6B')
            print(f"Error in load_volume: {e}")
        
    def on_point_size_changed(self, value):
        """Callback für Slider-Änderungen"""
        self.current_point_size = int(value)
        self.size_value_label.config(text=f"Current: {self.current_point_size}")
        if self.pcd is not None:
            self.status_label.config(text=f"Point size set to {self.current_point_size} (click Update to apply)",
                                   fg='#FFD700')
        
    def show_pointcloud(self):
        """Zeige die Punktwolke in einem neuen Fenster"""
        try:
            if self.pcd is None:
                messagebox.showwarning("Warning", "Please load a volume file first!")
                return

            if self.vis is not None:
                print("close previous cloud")
                self.close_visualizer()
            
            # Erstelle neuen Visualizer in separatem Thread
            self.vis_thread = threading.Thread(target=self._run_visualizer)
            self.vis_thread.daemon = True
            self.vis_thread.start()
            
            self.status_label.config(text="Point cloud visualizer opened", fg='#90EE90')
            
        except Exception as e:
            error_msg = f"Error showing point cloud: {str(e)}"
            messagebox.showerror("Visualization Error", error_msg)
            self.status_label.config(text=error_msg, fg='#FF6B6B')
            print(f"Error in show_pointcloud: {e}")
    
    def _run_visualizer(self):
        """Führe den Visualizer in separatem Thread aus"""
        try:
            self._stop_visualizer = False
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("Point Cloud Viewer", width=800, height=600)
            
            # Füge Punktwolke hinzu
            self.vis.add_geometry(self.pcd)
            
            # Setze Render-Optionen
            render_option = self.vis.get_render_option()
            render_option.point_size = float(self.current_point_size)
            render_option.background_color = np.array([0, 0, 0])
            
            # Visualizer-Loop
            # while True:
            while not self._stop_visualizer:
                if not self.vis.poll_events():
                    break
                
                # Prüfe auf Updates
                if self.should_update:
                    render_option.point_size = float(self.current_point_size)
                    self.should_update = False
                
                self.vis.update_renderer()
                time.sleep(0.01)  # Kleine Pause um CPU zu schonen
            
            self.vis.destroy_window()
            self.vis = None
            
        except Exception as e:
            print(f"Error in visualizer thread: {e}")
            self.vis = None
    
    def update_pointcloud(self):
        """Aktualisiere die Punktwolke mit neuen Parametern"""
        try:            
            if self.pcd is None:
                messagebox.showwarning("Warning", "Please load a volume file first!")
                return
            
            if self.vis is not None:
                self.should_update = True
                self.status_label.config(text=f"Point size updated to {self.current_point_size}", 
                                       fg='#90EE90')
            else:
                self.status_label.config(text="No visualizer open. Click 'Show Point Cloud' first.", 
                                       fg='#FFD700')
                
        except Exception as e:
            error_msg = f"Update error: {str(e)}"
            messagebox.showerror("Update Error", error_msg)
            self.status_label.config(text=error_msg, fg='#FF6B6B')
            print(f"Error in update_pointcloud: {e}")
    
    def close_visualizer(self):
        """Schließe den Visualizer"""
        try:
            if self.vis is not None:
                self._stop_visualizer = True  # signal loop to end
                time.sleep(0.05)  # short wait to ensure thread exits render loop
                # self.vis.destroy_window()
                # self.vis = None
            
                if self.vis_thread is not None:
                    self.vis_thread.join(timeout=1.0)
                    self.vis_thread = None
                
            self.status_label.config(text="Visualizer closed", fg='#90EE90')
            
        except Exception as e:
            error_msg = f"Close error: {str(e)}"
            messagebox.showerror("Close Error", error_msg)
            self.status_label.config(text=error_msg, fg='#FF6B6B')
            print(f"Error in close_visualizer: {e}")
    
    def run(self):
        """Starte die GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Cleanup beim Schließen"""
        self.close_visualizer()
        self.root.destroy()

def main():
    app = PointCloudMinimalGUI()
    app.run()

if __name__ == "__main__":
    main()

