"""
Object Detector Module
This module is responsible for detecting objects in the frame.
It uses YOLO model to detect objects.
It has two models: one for color and one for infrared.
It can detect objects of the following types: ball, red, blue, robot, person.
It returns a list of DetectedObject.
"""

import time
import ultralytics.engine
import ultralytics.engine.results
from realsense_handler import FramesMix

import ultralytics
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ObjectType(Enum):
    BALL = "ball"
    GOAL_POSTS = "goal"
    LINES = "lines"
    PERSON = "person"
    ROBOT = "robot"
    RED = "red"
    BLUE = "blue"


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """Dataclass to store object information"""

    type: ObjectType
    confidence: float
    # The Box is a tuple of 4 integers: x1, y1, x2, y2
    box: tuple[int, int, int, int]  # TO THINK: Should we create a type for this?
    source: int


class ObjectDetector:
    def __init__(self, color_model_name: str, ir_model_name: str, color_full_model_name: str = ""):
        if color_full_model_name == "":
            color_full_model_name = color_model_name
        self.color_full_model = ultralytics.YOLO(color_full_model_name, task="detect")
        self.color_full_classes = [
            ObjectType.BALL,
            ObjectType.GOAL_POSTS,
            ObjectType.LINES,
            ObjectType.PERSON,
            ObjectType.ROBOT
        ]

        self.color_model = ultralytics.YOLO(color_model_name, task="detect")
        self.color_classes = [
            ObjectType.BALL,
            ObjectType.GOAL_POSTS,
            ObjectType.LINES,
            ObjectType.PERSON,
            ObjectType.ROBOT
        ]

        self.ir_model = ultralytics.YOLO(ir_model_name, task="detect")
        self.ir_classes = [ObjectType.BALL, ObjectType.PERSON, ObjectType.ROBOT]
        self.allowed_types = [
            ObjectType.BALL,
            # ObjectType.PERSON,
            # ObjectType.ROBOT
        ]
        self.source_supress_list = [1, 3]
        self.source_supress_index = 0

    def _get_class(self, using_color: bool, class_id: int) -> ObjectType:
        # TO FIX: Fix the index to match correctly with the correct class
        if using_color:
            return self.color_classes[class_id]
        else:
            return self.ir_classes[class_id]

    def _map_result_to_object(
        self, source_type: FramesMix, result: list[float], source: int
    ) -> DetectedObject:
        confidence = result[4]
        type = self._get_class(source_type == FramesMix.DEPTH_COLOR, int(result[5]))
        box = tuple(map(lambda x: int(x), result[:4]))
        return DetectedObject(type, confidence, box, source)
    
    def _filter_by_type(
        self, source_type: FramesMix, result: list[float]
    ) -> list[DetectedObject]:
        type = self._get_class(source_type == FramesMix.DEPTH_COLOR, int(result[5]))
        return type in self.allowed_types

    def _parse_realsense_results(self, realsense_results: list[ultralytics.engine.results.Results], source_type: FramesMix) -> list[DetectedObject]:
        REALSENSE_INDEX: int = 0
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
                    self._map_result_to_object(source_type, result, source=REALSENSE_INDEX)
                )

        return detected_objects

    def _parse_threecamera_results(self, threecamera_results: list[list[ultralytics.engine.results .Results]]) -> tuple[list[DetectedObject], tuple[list]]:
        detected_objects: list[DetectedObject] = []
        
        detected_balls = []
        detected_goal_posts =[]
        detected_robots = []
        detected_blue_shirts = []
        detected_red_shirts = []
        detected_humans = []
        detected_lines = []
        
        for cam_index, result in enumerate(threecamera_results):
            result_size: int = len(result)
            for result_index in range(result_size):
                x1, y1, x2, y2, conf, cls = result[result_index].boxes.data.tolist()[0]
                confidence = float(conf)
                object_class: ObjectType = self.color_full_classes[int(cls)]
                detected_data = None
                match object_class:
                    case ObjectType.BALL:
                        detected_data = detected_balls
                    case ObjectType.GOAL_POSTS:
                        detected_data = detected_goal_posts
                    case ObjectType.ROBOT:
                        detected_data = detected_robots
                    case ObjectType.LINES:
                        detected_data = detected_lines
                    case ObjectType.PERSON:
                        detected_data = detected_humans

                if detected_data is not None:
                    real_cam_index = 0
                    supressed_source = self.source_supress_list[self.source_supress_index]
                    if supressed_source <= (cam_index + 1):
                        real_cam_index = cam_index + 1
                    else:
                        real_cam_index = cam_index
                    
                    detected_data.append([int(cls), round(confidence, 2), int(x1), int(y1), int(x2), int(y2), real_cam_index])
                    detected_object_element = [x1, y1, x2, y2, conf, cls]
                    if not self._filter_by_type(FramesMix.DEPTH_COLOR, detected_object_element):
                        continue
                    
                    detected_objects.append(
                        self._map_result_to_object(FramesMix.DEPTH_COLOR, detected_object_element, source=real_cam_index + 1)
                    )

        return detected_objects, (detected_balls, detected_goal_posts, detected_robots, detected_blue_shirts, detected_red_shirts, detected_humans)

    def detect(
        self, source_type: FramesMix, source: list[np.ndarray]
    ) -> list[list[DetectedObject]]:
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
        
        source.pop(self.source_supress_list[self.source_supress_index])

        results: list[list[ultralytics.engine.results.Results]] = model.predict(
            source=source,
            conf=0.5,
            verbose=False,
            device=0,
            half=True,
            int8=False,
            agnostic_nms=False,
            imgsz=(480, 640),
            batch=3,
        )

        realsense_index: int = 0
        realsense_results = self._parse_realsense_results(results[realsense_index], source_type)

        if not only_realsense:
            threecamera_results = self._parse_threecamera_results(results[realsense_index + 1 : realsense_index + 3])
        else:
            threecamera_results = [[]]

        self.source_supress_index = (self.source_supress_index + 1) % len(self.source_supress_list)
        return realsense_results, threecamera_results
        