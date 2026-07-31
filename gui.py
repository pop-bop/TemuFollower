import customtkinter as ctk
from PIL import Image
import cv2
import collections
import time

class Dashboard(ctk.CTk):
    def __init__(self, globals_dict):
        super().__init__()
        
        self.title("TemuFollower Dashboard")
        self.geometry("1000x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.globals_dict = globals_dict
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Left Side - Camera View
        self.cam_frame = ctk.CTkFrame(self)
        self.cam_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.cam_label = ctk.CTkLabel(self.cam_frame, text="Waiting for camera...", font=("Helvetica", 16))
        self.cam_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Right Side Top - Graph
        self.graph_frame = ctk.CTkFrame(self)
        self.graph_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.graph_label = ctk.CTkLabel(self.graph_frame, text="Motor Outputs", font=("Helvetica", 14, "bold"))
        self.graph_label.pack(pady=(10, 0))

        # Motor Output Canvas
        self.canvas_width = 450
        self.canvas_height = 250
        self.canvas = ctk.CTkCanvas(self.graph_frame, bg="#1e1e1e", highlightthickness=0, height=self.canvas_height)
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Data history for graph
        self.history_len = 200
        self.motor_l_hist = collections.deque([0]*self.history_len, maxlen=self.history_len)
        self.motor_r_hist = collections.deque([0]*self.history_len, maxlen=self.history_len)
        
        # Right Side Bottom - Controls & Stats
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.stats_label = ctk.CTkLabel(self.control_frame, text="Speed: 0.00 m/s | State: INIT", font=("Helvetica", 18, "bold"))
        self.stats_label.pack(pady=15)
        
        # Sliders
        # Speed Slider
        self.speed_val = ctk.DoubleVar(value=self.globals_dict.get("BASE_SPEED", 0.5))
        self.speed_label = ctk.CTkLabel(self.control_frame, text=f"Base Speed: {self.speed_val.get():.2f}")
        self.speed_label.pack()
        self.speed_slider = ctk.CTkSlider(self.control_frame, from_=0.0, to=2.0, variable=self.speed_val, command=self.update_speed)
        self.speed_slider.pack(pady=(0, 10), padx=20, fill="x")
        
        # KP Slider
        self.kp_val = ctk.DoubleVar(value=self.globals_dict.get("KP", 1.5))
        self.kp_label = ctk.CTkLabel(self.control_frame, text=f"KP (Proportional): {self.kp_val.get():.2f}")
        self.kp_label.pack()
        self.kp_slider = ctk.CTkSlider(self.control_frame, from_=0.0, to=5.0, variable=self.kp_val, command=self.update_kp)
        self.kp_slider.pack(pady=(0, 10), padx=20, fill="x")

        # KI Slider
        self.ki_val = ctk.DoubleVar(value=self.globals_dict.get("KI", 0.0))
        self.ki_label = ctk.CTkLabel(self.control_frame, text=f"KI (Integral): {self.ki_val.get():.4f}")
        self.ki_label.pack()
        self.ki_slider = ctk.CTkSlider(self.control_frame, from_=0.0, to=0.1, variable=self.ki_val, command=self.update_ki)
        self.ki_slider.pack(pady=(0, 10), padx=20, fill="x")
        
        # KD Slider
        self.kd_val = ctk.DoubleVar(value=self.globals_dict.get("KD", 0.0))
        self.kd_label = ctk.CTkLabel(self.control_frame, text=f"KD (Derivative): {self.kd_val.get():.2f}")
        self.kd_label.pack()
        self.kd_slider = ctk.CTkSlider(self.control_frame, from_=0.0, to=5.0, variable=self.kd_val, command=self.update_kd)
        self.kd_slider.pack(pady=(0, 10), padx=20, fill="x")

        # Track last update time to maintain FPS in GUI
        self.last_update_time = time.time()
        self.gui_update_interval = 1.0 / 30.0  # 30 FPS for GUI

    def update_speed(self, value):
        self.speed_label.configure(text=f"Base Speed: {value:.2f}")
        self.globals_dict["BASE_SPEED"] = float(value)
        
    def update_kp(self, value):
        self.kp_label.configure(text=f"KP (Proportional): {value:.2f}")
        self.globals_dict["KP"] = float(value)

    def update_ki(self, value):
        self.ki_label.configure(text=f"KI (Integral): {value:.4f}")
        self.globals_dict["KI"] = float(value)

    def update_kd(self, value):
        self.kd_label.configure(text=f"KD (Derivative): {value:.2f}")
        self.globals_dict["KD"] = float(value)

    def pump_events(self):
        # Keep GUI responsive without blocking
        self.update_idletasks()
        self.update()

    def update_state(self, frame, speed, state, motor_l, motor_r):
        current_time = time.time()
        
        # Update logic that is lightweight
        self.motor_l_hist.append(motor_l)
        self.motor_r_hist.append(motor_r)
        
        # Only update heavy GUI parts at 30 FPS
        if current_time - self.last_update_time >= self.gui_update_interval:
            self.last_update_time = current_time
            
            # Update Stats
            self.stats_label.configure(text=f"Speed: {speed:.2f} m/s | State: {state}")
            
            # Update Image
            if frame is not None:
                # Resize to fit the label roughly
                h, w = frame.shape[:2]
                target_w = 480
                target_h = int(h * (target_w / w))
                frame_resized = cv2.resize(frame, (target_w, target_h))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(target_w, target_h))
                self.cam_label.configure(image=imgtk, text="")
                self.cam_label.image = imgtk
                
            # Update Graph
            self.draw_graph()
            
        self.pump_events()
        
    def draw_graph(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return
            
        dx = w / self.history_len
        
        # draw center line
        self.canvas.create_line(0, h/2, w, h/2, fill="#555555", dash=(4,4))
        
        def plot_line(data, color):
            points = []
            for i, val in enumerate(data):
                x = i * dx
                # val is typically -1.0 to 1.0
                y = h/2 - (val * h/2)
                points.extend([x, y])
            if len(points) >= 4:
                self.canvas.create_line(*points, fill=color, width=2)
                
        plot_line(self.motor_l_hist, "#ff4a4a") # Left motor - Red
        plot_line(self.motor_r_hist, "#4a90e2") # Right motor - Blue
        
        # Legend
        self.canvas.create_text(10, 10, text="Left Motor", fill="#ff4a4a", anchor="nw")
        self.canvas.create_text(10, 30, text="Right Motor", fill="#4a90e2", anchor="nw")
