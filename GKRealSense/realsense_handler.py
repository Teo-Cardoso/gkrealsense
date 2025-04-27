from enum import Enum
from dataclasses import dataclass
import pyrealsense2 as rs
import time


@dataclass
class RealSenseConfig:
    """Dataclass to store RealSense configuration"""

    width: int = 848
    height: int = 480
    depth_fps: int = 60
    rgb_fps: int = 60  # We have to use 60fps to ensure depth 90fps
    laser_power: int = 360


@dataclass
class FramesNumber:
    depth: int = 0
    infrared: int = 0
    color: int = 0


class FramesMix(Enum):
    DEPTH_INFRARED = 0
    DEPTH_COLOR = 1


class RealSenseHandler:
    """Class to handle RealSense camera"""

    def __init__(self, cam_config: RealSenseConfig):
        config = rs.config()

        config.enable_stream(
            rs.stream.depth,
            cam_config.width,
            cam_config.height,
            rs.format.z16,
            cam_config.depth_fps,
        )
        # config.enable_stream(
        #     rs.stream.infrared,
        #     cam_config.width,
        #     cam_config.height,
        #     rs.format.y8,
        #     cam_config.depth_fps,
        # )
        config.enable_stream(
            rs.stream.color,
            cam_config.width,
            cam_config.height,
            rs.format.bgr8,
            cam_config.rgb_fps,
        )

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_sensor.set_option(rs.option.laser_power, cam_config.laser_power)

        VISUAL_PRESET_HIGH_ACCURACY = 3
        self.depth_sensor.set_option(
            rs.option.visual_preset, VISUAL_PRESET_HIGH_ACCURACY
        )

        self.depth_intrinsics = (
            self.profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.color_intrinsics = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        # self.ir_intrinsics = (
        #     self.profile.get_stream(rs.stream.infrared)
        #     .as_video_stream_profile()
        #     .get_intrinsics()
        # )

        self.frames_number: FramesNumber = FramesNumber()
        self.last_time: int = time.time_ns()
        self.align = rs.align(rs.stream.color)

    def __del__(self):
        self.pipeline.stop()

    def get_frames(self) -> tuple[int, FramesMix, rs.depth_frame, rs.frame]:
        """Get frames from RealSense camera"""
        # TO DO: Add a check for repeated depth_frame, current depth frame must be different from the last frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        repeated_color_frame: bool = (
            self.frames_number.color == color_frame.get_frame_number()
        )

        result_type = (
            FramesMix.DEPTH_INFRARED if repeated_color_frame else FramesMix.DEPTH_COLOR
        )
        result_type = FramesMix.DEPTH_COLOR
        match result_type:
            case FramesMix.DEPTH_COLOR:
                self.frames_number.color = color_frame.get_frame_number()
                # Improvement point: Instead of align, we could only transform the color pixels into depth pixels
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                second_frame = aligned_frames.first(rs.stream.color)

            case FramesMix.DEPTH_INFRARED:
                second_frame = frames.get_infrared_frame()
                self.frames_number.infrared = second_frame.get_frame_number()
                depth_frame = frames.get_depth_frame()

        self.frames_number.depth = depth_frame.get_frame_number()
        self.last_time = time.time_ns()
        return self.last_time, result_type, depth_frame, second_frame
