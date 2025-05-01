import math
import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path + "/../InterProcessCommunication/")
import InterProcessCommunication as ipc

import numpy as np
from object_pose_estimator import ObjectWithPosition
from object_detector import DetectedObject

class ThreeCamerasHandler:
    def __init__(self, angles_file: str = None, distances_file: str = None):
        self.ipc = ipc.ImageTransportReceiver("CamGroup")
        self.last_timestamp: int = 0

        self.angle_shift = math.radians(120)
        self.angles = []
        with open(angles_file, 'r') as file:
            for line in file:
                angulos = [float(angulo) for angulo in line.strip().split('\t')]
                self.angles.append(angulos)

        self.distances = []
        with open(distances_file, 'r') as file:
            for line in file:
                distancias = [float(distancia) for distancia in line.strip().split('\t')]
                self.distances.append(distancias)

    # Adicionar os valores de ângulo à matriz    
    def get_images(self) -> list[np.ndarray]:
        metadata, frames = self.ipc.receive_image()
        # Should we wait or skip the frame?
        while metadata.timestamp == self.last_timestamp:
            metadata, frames = self.ipc.receive_image()

        return metadata.timestamp, (frames[0][:, 0:640], frames[0][:, 640:1280], frames[0][:, 1280:1920])

    def _map_detected_object_to_object_with_position(
        self, camera_to_world: np.ndarray, detected_object: DetectedObject
    ) -> ObjectWithPosition:
        
        x1, _, x2, y2 = detected_object.box
        center_x = int((x1 + x2) / 2)
        pixel_coordinates = [int(y2), center_x]

        angle = math.radians(self.angles[pixel_coordinates[0]][pixel_coordinates[1]])
        if detected_object.source == 1:
            angle = angle + self.angle_shift
        elif detected_object.source == 3:
            angle = angle - self.angle_shift

        distance = self.distances[pixel_coordinates[0]][pixel_coordinates[1]]

        x_distance = math.cos(angle) * distance * 1e-2
        y_distance = math.sin(angle) * distance * 1e-2
        relative_position = np.array([x_distance, y_distance, 0.0])

        # Apply the transformation from camera to world
        world_position = np.dot(camera_to_world, relative_position)
        world_position = world_position[:3].flatten()

        return ObjectWithPosition(
            detected_object.type,
            world_position,
            np.array([[0.4, 0.4, 0.4]])
        )
    
    def get_objects_position(self, robot_to_world: np.ndarray, detected_objects: list[DetectedObject]) -> list[ObjectWithPosition]:
        detected_objects_with_position: list[ObjectWithPosition] = [
            self._map_detected_object_to_object_with_position(
                robot_to_world, detected_object
            )
            for detected_object in detected_objects
        ]

        return detected_objects_with_position
        
    def fillup_detection_tracking_data(self, detected_balls: list, detected_goal_posts: list, detected_robots: list, detected_blue_shirts: list, detected_red_shirts: list, detected_humans: list):
        pass
        
        