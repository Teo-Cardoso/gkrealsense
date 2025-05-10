from dataclasses import dataclass, field
import math

import numpy as np
from ball_classifier import BallClassifiedObject, BallClass, BallClassificationProperties
from enum import Enum

class ActionStatus(Enum):
    IDLE = 0,
    PRE_DEFENCE  = 1,
    DEFENCE = 2,

class Arm(Enum):
    NONE = 0
    LEFT = 1,
    RIGHT = 2,
    UP = 3,

@dataclass(slots=True, frozen=True)
class BahaviorControlAction:
    """
    Class representing the action to be taken by the robot.
    """
    action: ActionStatus = ActionStatus.IDLE
    target: tuple[float, float] = field(default_factory = lambda: (0.0, 0.0))
    arm: int = Arm.NONE

class BehaviorControl:
    """
    Behavior Control class to manage the Behavior of the robot.
    """

    def __init__(self):
        self.classification_mapping: dict[BallClass, ActionStatus] = {
            BallClass.UNKNOWN: ActionStatus.IDLE,
            BallClass.GOING: ActionStatus.IDLE,
            BallClass.STOPPED: ActionStatus.PRE_DEFENCE,
            BallClass.COMING: ActionStatus.PRE_DEFENCE,
            BallClass.KICK: ActionStatus.DEFENCE,
        }

        self.goal_center = np.array([0, 0, 0])
        # Parameters for pre defence position
        A = (0.9)**2
        self.B = 0.6
        self.K = (self.B ** 2) / A
    
    def _compute_target_predefence(self, ball: BallClassifiedObject) -> tuple[float, float]:
        """
        Compute the target position for pre-defence action.
        """
        delta_distances = ball.dynamics.position - self.goal_center
        if abs(delta_distances[1]) > 0.05:
            m = delta_distances[0] / delta_distances[1]
            defence_position_y = math.copysign(self.B / math.sqrt(m**2 + self.K), m)
            defence_position_x = math.sqrt(self.B**2 - (self.K * defence_position_y**2))
        else:
            defence_position_y = 0.0
            defence_position_x = self.B

        return defence_position_x + self.goal_center[0], defence_position_y + self.goal_center[1]
    
    def _compute_target_defence(self, ball: BallClassifiedObject) -> tuple[float, float]:
        """
        Compute the target position for defence action.
        """
        # Placeholder for actual computation
        return self._compute_target_predefence(ball)
    
    def control(self, closest_ball: BallClassifiedObject) -> BahaviorControlAction:
        """
        Control the robot based on the ball candidates.
        """

        if closest_ball is None:
            return BahaviorControlAction(ActionStatus.IDLE, (0.0, 0.0), Arm.NONE)
        
        action = self.classification_mapping[closest_ball.classification]
        match action:
            case ActionStatus.IDLE:
                target = (0.0, 0.0)
                arm = Arm.NONE
            case ActionStatus.PRE_DEFENCE:
                target = self._compute_target_predefence(closest_ball)
                arm = Arm.NONE
            case ActionStatus.DEFENCE:
                target = self._compute_target_defence(closest_ball)
                arm = Arm.NONE

        return BahaviorControlAction(action, target, arm)