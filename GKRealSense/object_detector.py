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

import ultralytics
import numpy as np
from dataclasses import dataclass
from enum import Enum, IntEnum


class Source(IntEnum):
    REALSENSE = 0
    CAM_1 = 1
    CAM_2 = 2
    CAM_3 = 3


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
    source: Source


class ObjectDetector:
    def __init__(self, color_model_name: str):
        self.color_model = ultralytics.YOLO(color_model_name, task="detect")
        self.color_classes = [
            ObjectType.BALL,
            ObjectType.GOAL_POSTS,
            ObjectType.LINES,
            ObjectType.PERSON,
            ObjectType.ROBOT
        ]

        self.allowed_types = [
            ObjectType.BALL,
            # ObjectType.PERSON,
            # ObjectType.ROBOT
        ]
        self.source_supress_list = [1, 3]
        self.source_supress_index = 0

    def _get_class(self, class_id: int) -> ObjectType:
        return self.color_classes[class_id]

    def _map_result_to_object(self, result: list[float], source: Source) -> DetectedObject:
        confidence: float = result[4]
        type: ObjectType = self._get_class(int(result[5]))
        box: tuple[int, int, int, int] = tuple(map(lambda x: int(x), result[:4]))
        return DetectedObject(type, confidence, box, source)
    
    def _filter_by_type(self, result: list[float]) -> list[DetectedObject]:
        type: ObjectType = self._get_class(int(result[5]))
        return type in self.allowed_types

    def _parse_realsense_results(self, realsense_results: list[ultralytics.engine.results.Results]) -> list[DetectedObject]:
        detected_objects: list[DetectedObject] = []
        result_size: int = len(realsense_results)
        for result_index in range(result_size):
            result_data: list[float] = realsense_results[result_index].boxes.data.tolist()

            if len(result_data) == 0:
                continue

            if not self._filter_by_type(result_data[0]):
                continue

            detected_objects.append(
                self._map_result_to_object(result_data[0], source=Source.REALSENSE)
            )

        return detected_objects

    def _parse_threecamera_results(self, threecamera_results: list[list[ultralytics.engine.results.Results]], sources_id: list[Source]) -> tuple[list[DetectedObject], tuple[list]]:
        detected_objects: list[DetectedObject] = []
        
        detected_balls = []
        detected_goal_posts =[]
        detected_robots = []
        detected_humans = []
        detected_lines = []
        
        for result, cam_source  in zip(threecamera_results, sources_id):
            result_size: int = len(result)
            for result_index in range(result_size):
                x1, y1, x2, y2, conf, cls = result[result_index].boxes.data.tolist()[0]
                confidence = float(conf)
                object_class: ObjectType = self._get_class(int(cls))
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

                if detected_data is None:
                    continue
               
                detected_data.append([int(cls), round(confidence, 2), int(x1), int(y1), int(x2), int(y2), int(cam_source) - 1])
                detected_object_element = [x1, y1, x2, y2, conf, cls]
                if not self._filter_by_type(detected_object_element):
                    continue
                
                detected_objects.append(
                    self._map_result_to_object(detected_object_element, source=cam_source)
                )

        return detected_objects, (detected_balls, detected_goal_posts, detected_robots, detected_humans, detected_lines)

    def detect(
        self, source: list[np.ndarray], sources_id: list[Source] = []
    ) -> list[list[DetectedObject]]:
        """Detect objects in the frame"""
        assert len(source) == len(sources_id), "source and sources_id must have the same length"

        if len(source) == 0:
            return [], ([], ([], [], [], [], []))

        realsense_supressed: bool = sources_id[0] != Source.REALSENSE
        if not realsense_supressed:
            source.pop(self.source_supress_list[self.source_supress_index])
            sources_id.pop(self.source_supress_list[self.source_supress_index])

        results: list[list[ultralytics.engine.results.Results]] = self.color_model.predict(
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

        if not realsense_supressed:
            realsense_results = self._parse_realsense_results(results[int(Source.REALSENSE)])
            threecamera_results = self._parse_threecamera_results(results[1:3], sources_id[1:3])
            self.source_supress_index = (self.source_supress_index + 1) % len(self.source_supress_list)
        else:
            realsense_results = []
            threecamera_results = self._parse_threecamera_results(results, sources_id)

        return realsense_results, threecamera_results
        