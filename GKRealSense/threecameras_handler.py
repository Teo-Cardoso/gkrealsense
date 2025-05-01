import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path + "/../InterProcessCommunication/")
import InterProcessCommunication as ipc

import numpy as np
from object_pose_estimator import ObjectWithPosition
from object_detector import DetectedObject

class ThreeCamerasHandler:
    def __init__(self):
        self.ipc = ipc.ImageTransportReceiver("CamGroup")
        self.last_timestamp: int = 0
    
    def get_images(self) -> list[np.ndarray]:
        metadata, frames = self.ipc.receive_image()
        if metadata.timestamp == self.last_timestamp:
            return []

        return frames[0][:, 0:640], frames[0][:, 640:1280], frames[0][:, 1280:1920]

    def get_objects_position(self, robot_to_world: np.ndarray, detected_objects: list[DetectedObject]) -> list[ObjectWithPosition]:
        
        