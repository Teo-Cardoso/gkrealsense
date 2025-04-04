import re
import time
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageTk
import numpy as np
import io


def parse_kalman_line(line):
    match = re.match(r"timestamp: (\d+), objects: \[(.*)\]", line)
    if not match:
        return None

    timestamp = int(match.group(1))
    objects_str = match.group(2)

    object_pattern = re.compile(r"\((\d+), ObjectType\.(\w+), \[([^\]]+)\], ([\d\.]+), (True|False)\)")

    parsed_objects = []
    for obj_match in object_pattern.finditer(objects_str):
        obj_id = int(obj_match.group(1))
        obj_type = obj_match.group(2).capitalize()
        state_values = list(map(float, obj_match.group(3).split()))
        confidence = float(obj_match.group(4))
        valid = obj_match.group(5) == "True"

        parsed_objects.append([obj_id, obj_type, state_values, confidence, valid])

    return [timestamp] + parsed_objects


def parse_image_line(line, image_shape=(480, 848, 3)):
    match = re.match(r"timestamp: (\d+), image: \[(.*)\]", line)
    if not match:
        return None
    timestamp = int(match.group(1))
    image_bytes_str = match.group(2)
    byte_data = eval(image_bytes_str)
    image_array = np.frombuffer(byte_data, dtype=np.uint8)
    image_array = image_array.reshape(image_shape)
    return timestamp, image_array


class KalmanPlotter:
    def __init__(self, master):
        self.master = master
        self.master.title("Filtro de Kalman - Visualização")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        self.data = []
        self.image_data = []
        self.current_index = 0
        self.running = False

        controls_frame = tk.Frame(master)
        controls_frame.pack()

        self.load_button = tk.Button(controls_frame, text="Carregar Arquivo Kalman", command=self.load_file)
        self.load_button.grid(row=0, column=0)

        self.load_image_button = tk.Button(controls_frame, text="Carregar Arquivo de Imagens", command=self.load_image_file)
        self.load_image_button.grid(row=0, column=1)

        self.prev_button = tk.Button(controls_frame, text="Anterior", command=self.prev_cycle)
        self.prev_button.grid(row=0, column=2)

        self.next_button = tk.Button(controls_frame, text="Próximo", command=self.next_cycle)
        self.next_button.grid(row=0, column=3)

        self.jump_label = tk.Label(controls_frame, text="Ir para ciclo:")
        self.jump_label.grid(row=0, column=4)

        self.jump_entry = tk.Entry(controls_frame, width=5)
        self.jump_entry.grid(row=0, column=5)

        self.jump_button = tk.Button(controls_frame, text="Ir", command=self.jump_to_cycle)
        self.jump_button.grid(row=0, column=6)

        self.play_button = tk.Button(controls_frame, text="Reproduzir (90 ciclos/s)", command=self.play_cycles)
        self.play_button.grid(row=0, column=7)

        self.stop_button = tk.Button(controls_frame, text="Parar", command=self.stop_playback)
        self.stop_button.grid(row=0, column=8)

        self.display_frame = tk.Frame(master)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.view_init(elev=30, azim=45)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.display_frame)
        self.image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'r') as file:
                self.data = file.readlines()
            self.current_index = 0
            self.plot_cycle()

    def load_image_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Data", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'r') as file:
                self.image_data = file.readlines()

    def plot_cycle(self):
        if not self.data:
            return

        self.ax.clear()
        self.ax.view_init(elev=30, azim=45)

        parsed = parse_kalman_line(self.data[self.current_index])
        if not parsed:
            return

        timestamp = parsed[0]
        objects = parsed[1:]

        for obj in objects:
            obj_id, obj_type, state, _, _ = obj
            x, y, z, vx, vy, vz, *_ = state

            if obj_type.lower() == "ball":
                self.ax.scatter(x, y, z, c='red', s=50, marker='o')
            else:
                self.ax.scatter(x, y, z, c='blue', s=50, marker='^')

            self.ax.text(x, y, z + 0.2, f"ID {obj_id}", fontsize=8)
            self.ax.quiver(x, y, z, vx, vy, vz, length=0.5, normalize=True, color='black')

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(0, 10)
        self.ax.set_title(f'Ciclo {self.current_index + 1} - Timestamp: {timestamp}')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.canvas.draw()

        if self.image_data and self.current_index < len(self.image_data):
            result = parse_image_line(self.image_data[self.current_index])
            if result:
                _, image_array = result
                image = Image.fromarray(image_array)
                image = ImageTk.PhotoImage(image=image)
                self.image_label.config(image=image)
                self.image_label.image = image

    def prev_cycle(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.plot_cycle()

    def next_cycle(self):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.plot_cycle()

    def jump_to_cycle(self):
        try:
            index = int(self.jump_entry.get()) - 1
            if 0 <= index < len(self.data):
                self.current_index = index
                self.plot_cycle()
        except ValueError:
            pass

    def play_cycles(self):
        self.running = True
        self._play_loop()

    def _play_loop(self):
        if not self.running or self.current_index >= len(self.data):
            return
        self.plot_cycle()
        self.current_index += 1
        self.master.after(11, self._play_loop)

    def stop_playback(self):
        self.running = False

    def on_close(self):
        self.running = False
        self.master.destroy()
        exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = KalmanPlotter(root)
    root.mainloop()
