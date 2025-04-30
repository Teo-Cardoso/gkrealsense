from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from object_detector import ObjectType
from object_tracker import TrackedObject

@dataclass(slots=True)
class ObjectDynamic:
    """
    Class representing the dynamic state of an object.
    """
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

class BallClass(Enum):
    """
    Enum representing the classification of a ball.
    """
    UNKNOWN = 0
    KICK = 1
    COMING = 4
    GOING = 5
    STOPPED = 6

@dataclass(slots=True)
class BallClassificationProperties:
    """
    Class representing the properties of a ball classification.
    """
    cycle_count: int = 0
    cycles_going: int = 0
    cycles_coming: int = 0
    trustiness: float = 0.0
    time_to_goal: float = float("inf")
    distance_to_goal: float = float("inf")
    crossing_point: np.ndarray = field(default_factory = lambda: np.array([float("inf"), float("inf"), float("inf")]))

@dataclass(slots=True)
class BallClassifiedObject:
    """
    Class representing a classified ball object.
    """
    object_id: int
    dynamics: ObjectDynamic
    classification: BallClass
    properties: BallClassificationProperties = field(default_factory = lambda: BallClassificationProperties())

class BallClassifier:
    VX_THRESHOLD_GOING = 0.3
    VX_THRESHOLD_COMING = 0.3
    VX_THRESHOLD_KICK = 2.0
    TIME_TO_GOAL_THRESHOLD = 1.0
    DISTANCE_TO_GOAL_THRESHOLD = 0.5

    assert VX_THRESHOLD_COMING >= 0.01, "VX_THRESHOLD_COMING must be greater than 0.01"

    """
    Class for classifying ball objects based on their dynamics and time to goal.
    """
    def __init__(self, goal_center: np.ndarray = np.array([0, 0, 0]), goal_shape: tuple[float, float] = (2.4, 1.0)):
        self.goal_center = goal_center
        self.goal_shape = goal_shape

        self.balls: dict[int, BallClassifiedObject] = {}
        self.closest_ball = (None, float("inf"))
        self.closest_ball_by_time = (None, float("inf"))

    def classify(self, tracked_objects: list[TrackedObject]) -> tuple[dict[int, BallClassifiedObject], tuple[int, float], tuple[int, float]]:
        """
        Classify the ball object based on its dynamics and time to goal.
        """
        
        self.closest_ball = (None, float("inf"))
        self.closest_ball_by_time = (None, float("inf"))
        to_delete: list[int] = list(self.balls.keys())
        for tracked_object in tracked_objects:
            if tracked_object.objectStatus.object_type != ObjectType.BALL:
                continue

            ball_id = tracked_object.objectStatus.object_id
            if not (ball_id in self.balls.keys()):
                self._add_ball(tracked_object)
                continue
            else:
                to_delete.remove(ball_id)

            ball_position = tracked_object.kalmanFilter.x[:3]
            ball_velocity = tracked_object.kalmanFilter.x[3:6]
            ball_acceleration = tracked_object.kalmanFilter.x[6:8]

            self.balls[ball_id].dynamics = ObjectDynamic(ball_position, ball_velocity, ball_acceleration)

            self._update_properties(self.balls[ball_id])
            self.balls[ball_id].properties.trustiness = tracked_object.kalmanFilterStatus.trustiness 
            self.balls[ball_id].classification = self._classify_ball(self.balls[ball_id])
        
        for ball_id in to_delete:
            if ball_id in self.balls.keys():
                del self.balls[ball_id]

        return self.balls, self.closest_ball, self.closest_ball_by_time

    
    def _add_ball(self, tracked_object: TrackedObject):
        """
        Add a new ball to the classifier.
        """

        ball_id = tracked_object.objectStatus.object_id
        ball_position = tracked_object.kalmanFilter.x[:3]
        ball_velocity = tracked_object.kalmanFilter.x[3:5]
        ball_acceleration = tracked_object.kalmanFilter.x[6:8]

        self.balls[ball_id] = BallClassifiedObject(
            object_id=ball_id,
            dynamics=ObjectDynamic(ball_position, ball_velocity, ball_acceleration),
            classification=BallClass.UNKNOWN,
        )

        self._update_properties(self.balls[ball_id])
        self.balls[ball_id].properties.trustiness = tracked_object.kalmanFilterStatus.trustiness
        self.balls[ball_id].classification = self._classify_ball(self.balls[ball_id])
    
    def _update_properties(self, ball: BallClassifiedObject):
        """
        Update the properties of the ball classification.
        """
        ball.properties.cycle_count += 1
        ball.properties.distance_to_goal = np.linalg.norm(ball.dynamics.position[:2] - self.goal_center[:2])

        if ball.properties.distance_to_goal < self.closest_ball[1]:
            self.closest_ball = (ball.object_id, ball.properties.distance_to_goal)

        if ball.dynamics.velocity[0] > self.VX_THRESHOLD_GOING:
            ball.properties.cycles_going = min(10, ball.properties.cycles_going + 1)
            ball.properties.cycles_coming = 0
            ball.properties.time_to_goal = float("inf")
            ball.properties.crossing_point = np.array([float("inf"), float("inf"), float("inf")])
            return
        
        if ball.dynamics.velocity[0] < -self.VX_THRESHOLD_COMING:
            ball.properties.cycles_coming = min(10, ball.properties.cycles_coming + 1)
            ball.properties.cycles_going = 0
            
            # Compute crossing point
            relative_position = ball.dynamics.position - self.goal_center
            relative_velocity = ball.dynamics.velocity

            time_to_goal = abs(relative_position[0] / relative_velocity[0])
            if ball.properties.time_to_goal == float("inf"):
                ball.properties.time_to_goal = time_to_goal
            else:
                ball.properties.time_to_goal = (ball.properties.time_to_goal + time_to_goal) / 2.0
            
            if ball.properties.time_to_goal < self.closest_ball_by_time[1]:
                self.closest_ball_by_time = (ball.object_id, ball.properties.time_to_goal)

            crossing_point = ball.dynamics.position + ball.properties.time_to_goal * ball.dynamics.velocity
            if ball.properties.crossing_point[0] == float("inf"):
                ball.properties.crossing_point = crossing_point
            else:
                ball.properties.crossing_point = (ball.properties.crossing_point + crossing_point) / 2.0

            return

        ball.properties.cycles_going = max(0, ball.properties.cycles_going - 1)
        ball.properties.cycles_coming = max(0, ball.properties.cycles_coming - 1)
        ball.properties.time_to_goal = float("inf")
        ball.properties.crossing_point = np.array([float("inf"), float("inf"), float("inf")])
        ball.properties.distance_to_goal = float("inf")        

    def _classify_ball(self, ball: BallClassifiedObject) -> BallClass:
        """
        Classify the ball based on its properties.
        """
        
        if ball.properties.cycles_coming == 0 and ball.properties.cycles_going == 0:
            return BallClass.STOPPED

        if ball.properties.cycles_going > 0:
            return BallClass.GOING
        
        if ball.properties.cycles_coming > 0:
            if ball.properties.time_to_goal < self.TIME_TO_GOAL_THRESHOLD:
                return BallClass.KICK
            
            if ball.properties.distance_to_goal < self.DISTANCE_TO_GOAL_THRESHOLD:
                return BallClass.KICK

            if ball.dynamics.velocity[0] < -self.VX_THRESHOLD_KICK:
                return BallClass.KICK
            
            return BallClass.COMING
        
        return BallClass.UNKNOWN
