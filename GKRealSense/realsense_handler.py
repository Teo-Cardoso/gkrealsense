from enum import Enum
from dataclasses import dataclass
import pyrealsense2 as rs
import time
from concurrent.futures import ThreadPoolExecutor, Future

@dataclass
class RealSenseConfig:
    """Dataclass to store RealSense configuration"""

    width: int = 848
    height: int = 480
    depth_fps: int = 60
    rgb_fps: int = 60  # We have to use 60fps to ensure depth 90fps
    laser_power: int = 360


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

        config.enable_stream(
            rs.stream.color,
            cam_config.width,
            cam_config.height,
            rs.format.bgr8,
            cam_config.rgb_fps,
        )
        self.pipeline = rs.pipeline()
        self.started = False
        self.profile = self.pipeline.start(config)
        self.started = True

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

        self.last_time: int = time.time_ns()
        self.align = rs.align(rs.stream.color)
        self.align_executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self):
        if self.started:
            self.pipeline.stop()

    def lazy_align(self, frames) -> rs.composite_frame:
        time.sleep(5 * 1e-3) # Sleep for 5ms to allow the program start the gpu prediction
        return self.align.process(frames)

    def get_frames(self, use_future=True) -> tuple[int, Future[rs.composite_frame], rs.frame]:
        """Get frames from RealSense camera"""

        frames = self.pipeline.wait_for_frames()
        self.last_time = time.time_ns()

        color_frame = frames.get_color_frame()
        if use_future:
            aligned_frames_future = self.align_executor.submit(self.lazy_align, frames)
            return self.last_time, aligned_frames_future, color_frame
        else:
            aligned_frames = self.align.process(frames)
            return self.last_time, aligned_frames, color_frame

