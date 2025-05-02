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
        center_x = int(x1 + (x2 - x1) / 2)
        pixel_coordinates = [int(y2), center_x]
        pixel_coordinates[0] = np.clip(pixel_coordinates[0], 0, len(self.angles) - 1)
        pixel_coordinates[1] = np.clip(pixel_coordinates[1], 0, len(self.angles[0]) - 1)
        angle = math.radians(self.angles[pixel_coordinates[0]][pixel_coordinates[1]])

        if detected_object.source == 1:
            angle = angle - self.angle_shift
        elif detected_object.source == 3:
            angle = angle + self.angle_shift

        distance = self.distances[pixel_coordinates[0]][pixel_coordinates[1]]

        x_distance = math.cos(angle) * distance
        y_distance = math.sin(angle) * distance
        relative_position = np.array([x_distance, y_distance, 0.0, 1.0]).transpose()

        # Apply the transformation from camera to world
        world_position = np.dot(camera_to_world, relative_position)
        world_position = world_position[:3].flatten()

        return ObjectWithPosition(
            detected_object.type,
            world_position,
            np.array([3 * [0.4 / detected_object.confidence]])
        )
    
    def get_objects_position(self, robot_to_world: np.ndarray, detected_objects: list[DetectedObject]) -> list[ObjectWithPosition]:
        detected_objects_with_position: list[ObjectWithPosition] = [
            self._map_detected_object_to_object_with_position(
                robot_to_world, detected_object
            )
            for detected_object in detected_objects
        ]

        # Remove overlapping objects
        to_delete = []
        detected_objects_with_position_size = len(detected_objects_with_position)
        for i in range(detected_objects_with_position_size):
            for j in range(i + 1, detected_objects_with_position_size):
                if j in to_delete:
                    continue

                if (detected_objects_with_position[i].object_type != detected_objects_with_position[j].object_type):
                    continue

                distance_between_objects = np.linalg.norm(
                    detected_objects_with_position[i].position
                    - detected_objects_with_position[j].position
                )

                # In the normal case we only have one ball, so we can try to use a higher threshold
                # to remove the overlapping objects
                if distance_between_objects > 0.325:
                    continue
                
                if detected_objects_with_position[i].variance[0][0] > detected_objects_with_position[j].variance[0][0]:
                    to_delete.append(i)
                    break
                else:
                    to_delete.append(j)
        
        to_delete = list(set(to_delete)) # Make sure to delete only once
        to_delete.sort(reverse=True) # Sort in reverse order to avoid index issues
        for i in to_delete:
            detected_objects_with_position.pop(i)
            detected_objects.pop(i)

        return detected_objects_with_position
        
    def fillup_detection_tracking_data(self, detected_balls: list, detected_goal_posts: list, detected_robots: list, detected_blue_shirts: list, detected_red_shirts: list, detected_humans: list):
        pass
        
        