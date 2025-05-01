"""
Object Detector Module
This module is responsible for detecting objects in the frame.
It uses YOLO model to detect objects.
It has two models: one for color and one for infrared.
It can detect objects of the following types: ball, red, blue, robot, person.
It returns a list of DetectedObject.
"""

import ultralytics.engine
import ultralytics.engine.results
from realsense_handler import FramesMix

import ultralytics
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ObjectType(Enum):
    BALL = "ball"
    RED = "red"
    BLUE = "blue"
    ROBOT = "robot"
    PERSON = "person"


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """Dataclass to store object information"""

    type: ObjectType
    confidence: float
    # The Box is a tuple of 4 integers: x1, y1, x2, y2
    box: tuple[int, int, int, int]  # TO THINK: Should we create a type for this?


class ObjectDetector:
    def __init__(self, color_model_name: str, ir_model_name: str, color_full_model_name: str = ""):
        if color_full_model_name == "":
            color_full_model_name = color_model_name
        self.color_full_model = ultralytics.YOLO(color_full_model_name, task="detect")
        self.color_model = ultralytics.YOLO(color_model_name, task="detect")
        self.color_classes = [
            ObjectType.BALL,
            ObjectType.RED,
            ObjectType.BLUE,
            ObjectType.PERSON,
        ]

        self.ir_model = ultralytics.YOLO(ir_model_name, task="detect")
        self.ir_classes = [ObjectType.BALL, ObjectType.PERSON, ObjectType.ROBOT]

    def _get_class(self, using_color: bool, class_id: int) -> ObjectType:
        # TO FIX: Fix the index to match correctly with the correct class
        if using_color:
            return self.color_classes[min(class_id, 2)]
        else:
            return self.ir_classes[min(class_id, 2)]

    def _map_result_to_object(
        self, source_type: FramesMix, result: list[float]
    ) -> DetectedObject:
        confidence = result[4]
        type = self._get_class(source_type == FramesMix.DEPTH_COLOR, int(result[5]))
        box = tuple(map(lambda x: int(x), result[:4]))
        return DetectedObject(type, confidence, box)
    
    def _filter_by_type(
        self, source_type: FramesMix, result: list[float]
    ) -> list[DetectedObject]:
        type = self._get_class(source_type == FramesMix.DEPTH_COLOR, int(result[5]))
        return type == ObjectType.BALL

    def _parse_realsense_results(self, realsense_results: list[ultralytics.engine.results.Results], source_type: FramesMix) -> list[DetectedObject]:
        detected_objects: list[DetectedObject] = []
        result_size: int = len(realsense_results)
        for result_index in range(result_size):
            result_data: list[float] = realsense_results[result_index].boxes.data.tolist()

            if len(result_data) == 0:
                continue

            for result in result_data:
                if not self._filter_by_type(source_type, result):
                    continue

                detected_objects.append(
                    self._map_result_to_object(source_type, result)
                )

        return detected_objects

    def _parse_threecamera_results(self, threecamera_results: list[list[ultralytics.engine.results .Results]]) -> list[DetectedObject]:
        detected_objects: list[list[DetectedObject]] = [[], [], []]
        for index, result in enumerate(threecamera_results):
            for result_index in range(len(result)):
                result_data: list[float] = result[result_index].boxes.data.tolist()

                if len(result_data) == 0:
                    continue

                for result in result_data:
                    if not self._filter_by_type(FramesMix.DEPTH_COLOR, result):
                        continue

                    detected_objects[index].append(
                        self._map_result_to_object(FramesMix.DEPTH_COLOR, result)
                    )

        return detected_objects

    def detect(
        self, source_type: FramesMix, source: list[np.ndarray]
    ) -> list[DetectedObject]:
        """Detect objects in the frame"""

        model = None
        only_realsense = len(source) == 1
        if only_realsense:
            # Only Using RealSense
            model: ultralytics.YOLO = (
                self.color_model if source_type == FramesMix.DEPTH_COLOR else self.ir_model
            )
        else:
            # Using Three Cameras
            model: ultralytics.YOLO = self.color_full_model
        
        results: list[list[ultralytics.engine.results.Results]] = model.predict(
            source=source,
            conf=0.5,
            verbose=False,
            device=0,
            half=False,
            int8=False,
            agnostic_nms=True,
            imgsz=640,
        )

        realsense_index: int = 0
        realsense_results = self._parse_realsense_results(results[realsense_index], source_type)

        if not only_realsense:
            threecamera_results = self._parse_threecamera_results(results[realsense_index + 1 : realsense_index + 4])
            return realsense_results, threecamera_results
        
        return realsense_index, [[], [], []]
        