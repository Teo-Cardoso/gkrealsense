import re
import time
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from PIL import Image, ImageTk
import numpy as np
import io


def parse_kalman_line(line):
    match = re.match(r"timestamp: (\d+), objects: \[(.*)\]", line)
    if not match:
        return None

    timestamp = int(match.group(1))
    objects_str = match.group(2)

    object_pattern = re.compile(r"\((\d+), ObjectType\.(\w+), \[([^\]]+)\], (\[\[.*?\]\]), ([\d\.]+), (True|False)\)")

    parsed_objects = []
    for obj_match in object_pattern.finditer(objects_str):
        obj_id = int(obj_match.group(1))
        obj_type = obj_match.group(2).capitalize()
        state_values = list(map(float, obj_match.group(3).split()))
        try:
            covariance = eval(obj_match.group(4))
        except Exception as e:
            print(f"Error parsing covariance matrix: {e}")
            covariance = np.zeros((9, 9))
        confidence = float(obj_match.group(5))
        valid = obj_match.group(6) == "True"

        parsed_objects.append([obj_id, obj_type, state_values, covariance, confidence, valid])

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
        self.master.protocol("WM_DELETE_WINDOW", self.close_app)

        self.data = []
        self.image_data = []
        self.current_index = 0
        self.running = False
        self.selected_object_id = None

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

        self.play_button = tk.Button(controls_frame, text="Reproduzir", command=self.play_cycles)
        self.play_button.grid(row=0, column=7)

        self.stop_button = tk.Button(controls_frame, text="Parar", command=self.stop_playback)
        self.stop_button.grid(row=0, column=8)

        self.id_label = tk.Label(controls_frame, text="ID do objeto:")
        self.id_label.grid(row=1, column=0)

        self.id_entry = tk.Entry(controls_frame, width=5)
        self.id_entry.grid(row=1, column=1)

        self.id_button = tk.Button(controls_frame, text="Exibir Velocidade", command=self.select_object_by_id)
        self.id_button.grid(row=1, column=2)

        self.display_frame = tk.Frame(master)
        self.display_frame.pack()

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.view_init(elev=30, azim=180)

        # self.ax_2d = self.figure.add_subplot(122)
        # self.ax_2d.axis('off')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        # self.canvas.get_tk_widget().pack()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.display_frame)
        self.image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.graph_figure, self.graph_ax = plt.subplots()
        self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, master=master)
        self.graph_canvas.get_tk_widget().pack()

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

        parsed = parse_kalman_line(self.data[self.current_index])
        if not parsed:
            return

        timestamp = parsed[0]
        objects = parsed[1:]

        self.current_objects = []
        for obj in objects:
            obj_id, obj_type, state, covariance, conf, _ = obj
            x, y, z, vx, vy, vz, *_ = state
            marker = 'o' if obj_type.lower() == 'ball' else '^'
            color = 'red' if obj_type.lower() == 'ball' else 'blue'

            point = self.ax.scatter(x, y, z, c=color, s=50, marker=marker, picker=True)
            self.current_objects.append((point, obj_id, x, y, z))
            self.ax.text(x, y, z + 0.2, f"ID {obj_id}, Conf: {conf:.2f}", fontsize=8)
            self.ax.quiver(x, y, z, vx, vy, vz, length=1.0, normalize=False, color='black')

            # --- Elipsoide 3D para representar incerteza da posição ---
            pos_cov = np.array(covariance)[:3, :3]  # 3x3 posição

            # Autovalores e autovetores
            eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)

            # Escala a elipsoide para um intervalo visual razoável (ex: 1 desvio padrão)
            radii = np.sqrt(eigenvalues)

            # Cria esfera
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_ell = np.outer(np.cos(u), np.sin(v))
            y_ell = np.outer(np.sin(u), np.sin(v))
            z_ell = np.outer(np.ones_like(u), np.cos(v))

            # Aplica escala e rotação (transforma esfera em elipsoide)
            ellipsoid = np.array([x_ell, y_ell, z_ell])
            for i in range(3):
                ellipsoid[i] *= radii[i]
            ellipsoid = eigenvectors @ ellipsoid.reshape(3, -1)
            x_ell, y_ell, z_ell = ellipsoid.reshape(3, *x_ell.shape)

            # Translada para o ponto central (x, y, z)
            self.ax.plot_surface(
                x_ell + x,
                y_ell + y,
                z_ell + z,
                color=color,
                alpha=0.2,
                linewidth=0
            )

        self.ax.set_xlim(0, 15)
        self.ax.set_ylim(5, -5)
        self.ax.set_zlim(0, 3)
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

    def close_app(self):
        self.master.destroy()
        exit(0)

    def select_object_by_id(self):
        try:
            obj_id = int(self.id_entry.get())
            self.selected_object_id = obj_id
            self.plot_velocity_graph(obj_id)
        except ValueError:
            pass

    def plot_velocity_graph(self, object_id):
        cycles = []  # Usar o número do ciclo
        velocities_x = []
        velocities_y = []
        velocities_z = []
        confidences = []

        for cycle_index, line in enumerate(self.data):
            parsed = parse_kalman_line(line)
            if not parsed:
                continue
            timestamp = parsed[0]
            for obj in parsed[1:]:
                if obj[0] == object_id:
                    _, _, state, _, confidence, _ = obj
                    vx, vy, vz = state[3], state[4], state[5]
                    cycles.append(cycle_index + 1)  # Usar o número do ciclo (1-indexed)
                    velocities_x.append(vx)
                    velocities_y.append(vy)
                    velocities_z.append(vz)
                    confidences.append(confidence)

        # Clear previous graph
        self.graph_ax.clear()
        
        # Plot each component separately
        self.graph_ax.plot(cycles, velocities_x, label=f'Velocidade X do Objeto {object_id}', color='r')
        self.graph_ax.plot(cycles, velocities_y, label=f'Velocidade Y do Objeto {object_id}', color='g')
        self.graph_ax.plot(cycles, velocities_z, label=f'Velocidade Z do Objeto {object_id}', color='b')
        self.graph_ax.plot(cycles, confidences, label=f'Confiança do Objeto {object_id}', color='y', linestyle='--')
        
        # Title and labels
        self.graph_ax.set_title(f'Componentes da Velocidade do Objeto {object_id} ao longo dos ciclos')
        self.graph_ax.set_xlabel('Ciclo')
        self.graph_ax.set_ylabel('Velocidade')
        self.graph_ax.legend()

        # Redraw canvas
        self.graph_canvas.draw()



if __name__ == "__main__":
    root = tk.Tk()
    app = KalmanPlotter(root)
    root.mainloop()