import dataclasses
import math
import time
import numpy as np
import pygame

from ball_classifier import ObjectDynamic, BallClass, BallClassifiedObject, BallClassificationProperties


@dataclasses.dataclass(slots=True)
class Point:
    x: float
    y: float

@dataclasses.dataclass(slots=True)
class Circle:
    center: Point
    radius: float

@dataclasses.dataclass(slots=True)
class Dynamic:
    position: Point
    velocity: Point
    acceleration: Point

class Engine:
    def __init__(self, width: int = 640):
        pygame.init()
        self.font = pygame.font.Font(None, 16)
        self.width = width
        self.height = int(self.width * 14 / 22)

        self.field_scale = self.width / 22.0

        padding_size = 1  # 1 metro de borda
        self.padding = math.ceil(padding_size * self.field_scale)  # 1 metro de borda
        self.half_padding = self.padding / 2

        self.screen = pygame.display.set_mode(
            (self.padding + width, self.padding + self.height)
        )
        self.clock = pygame.time.Clock()

        self.balls_sprint: pygame.sprite.Group = pygame.sprite.Group()
        self.goalkeeper: pygame.sprite.Group = pygame.sprite.Group()

        pygame.display.set_caption("MSL Small 2D Visualizer")
        self.field_settings = {
            "real_width": self.width + self.padding,
            "real_height": self.height + self.padding,
            "field_color": (34, 139, 34),  # Verde
            "line_color": (255, 255, 255),  # Branco
            "line_width": math.ceil(0.125 * self.field_scale),  # 0.125 metros
            "center_circle_radius": math.ceil(1.5 * self.field_scale),  # 1.5 metros
            "arc_radius": math.ceil(0.75 * self.field_scale),  # 0.75 metros
            "goal_width": math.ceil(2.4 * self.field_scale),  # 2.4 metros
            "goal_depth": math.ceil(0.5 * self.field_scale),  # 0.6 metros
            "center_x": (self.width + self.padding) // 2,
        }

        self.field_marks = {
            "boundary_lines": [
                self.field_settings["line_color"],
                pygame.Rect(
                    self.half_padding, self.half_padding, self.width, self.height
                ),
                self.field_settings["line_width"],
            ],
            "center_line": [
                self.field_settings["line_color"],
                (self.field_settings["center_x"], self.half_padding),
                (
                    self.field_settings["center_x"],
                    self.half_padding + self.height - self.field_settings["line_width"],
                ),
                self.field_settings["line_width"],
            ],
            "center_circle": [
                self.field_settings["line_color"],
                (
                    self.field_settings["center_x"],
                    self.field_settings["real_height"] / 2,
                ),
                self.field_settings["center_circle_radius"],
                self.field_settings["line_width"],
            ],
            "top_left_corner": [
                self.field_settings["line_color"],
                pygame.Rect(
                    -self.field_settings["arc_radius"]
                    + self.half_padding
                    + self.field_settings["line_width"],
                    -self.field_settings["arc_radius"]
                    + self.half_padding
                    + self.field_settings["line_width"],
                    2 * self.field_settings["arc_radius"],
                    2 * self.field_settings["arc_radius"],
                ),
                math.radians(270),
                0,
                self.field_settings["line_width"],
            ],
            "top_right_corner": [
                self.field_settings["line_color"],
                pygame.Rect(
                    self.width - 2 * self.field_settings["line_width"],
                    self.half_padding
                    + self.field_settings["line_width"]
                    - self.field_settings["arc_radius"],
                    2 * self.field_settings["arc_radius"],
                    2 * self.field_settings["arc_radius"],
                ),
                math.radians(180),
                math.radians(270),
                self.field_settings["line_width"],
            ],
            "bottom_left_corner": [
                self.field_settings["line_color"],
                pygame.Rect(
                    self.half_padding
                    + self.field_settings["line_width"]
                    - self.field_settings["arc_radius"],
                    self.height - 2 * self.field_settings["line_width"],
                    2 * self.field_settings["arc_radius"],
                    2 * self.field_settings["arc_radius"],
                ),
                0,
                math.radians(90),
                self.field_settings["line_width"],
            ],
            "bottom_right_corner": [
                self.field_settings["line_color"],
                pygame.Rect(
                    self.width - 2 * self.field_settings["line_width"],
                    self.height - 2 * self.field_settings["line_width"],
                    2 * self.field_settings["arc_radius"],
                    2 * self.field_settings["arc_radius"],
                ),
                math.radians(90),
                math.radians(180),
                self.field_settings["line_width"],
            ],
            "left_goal": [
                self.field_settings["line_color"],
                pygame.Rect(
                    self.half_padding
                    + self.field_settings["line_width"]
                    - self.field_settings["goal_depth"],
                    self.field_settings["real_height"] // 2
                    - self.field_settings["goal_width"] // 2
                    - self.field_settings["line_width"],
                    self.field_settings["goal_depth"],
                    self.field_settings["goal_width"]
                    + 2 * self.field_settings["line_width"],
                ),
                self.field_settings["line_width"],
            ],
            "right_goal": [
                self.field_settings["line_color"],
                pygame.Rect(
                    self.width
                    + self.field_settings["goal_depth"]
                    - self.field_settings["line_width"],
                    self.field_settings["real_height"] // 2
                    - self.field_settings["goal_width"] // 2
                    - self.field_settings["line_width"],
                    self.field_settings["goal_depth"],
                    self.field_settings["goal_width"]
                    + 2 * self.field_settings["line_width"],
                ),
                self.field_settings["line_width"],
            ],
        }

    def _draw_field(self):
        # Fill the background with the field color
        self.screen.fill(self.field_settings["field_color"])

        # Draw the boundary lines (outer rectangle)
        pygame.draw.rect(self.screen, *self.field_marks["boundary_lines"])

        # Draw the center line
        pygame.draw.line(self.screen, *self.field_marks["center_line"])

        # Draw the center circle
        pygame.draw.circle(self.screen, *self.field_marks["center_circle"])

        # Draw corner arcs
        # Top-left corner
        pygame.draw.arc(self.screen, *self.field_marks["top_left_corner"])

        # # Top-right corner
        pygame.draw.arc(self.screen, *self.field_marks["top_right_corner"])

        # # Bottom-left corner
        pygame.draw.arc(self.screen, *self.field_marks["bottom_left_corner"])

        # # Bottom-right corner
        pygame.draw.arc(self.screen, *self.field_marks["bottom_right_corner"])

        # Draw the goals
        # Left goal
        pygame.draw.rect(self.screen, *self.field_marks["left_goal"])

        # Right goal
        pygame.draw.rect(self.screen, *self.field_marks["right_goal"])

    def get_scale(self):
        return self.field_scale

    def add_ball(self, ball):
        self.balls_sprint.add(ball)

    def add_goalkeeper(self, goalkeeper):
        self.goalkeeper.add(goalkeeper)

    def run(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return False

        self._draw_field()

        self.balls_sprint.update()
        self.goalkeeper.update()

        self.balls_sprint.draw(self.screen)
        self.goalkeeper.draw(self.screen)

        pygame.display.flip()
        return True

    def clear(self):
        self.balls_sprint.empty()
        self.goalkeeper.empty()


class Ball(pygame.sprite.Sprite):
    def __init__(self, engine: Engine, ball: BallClassifiedObject):
        super().__init__()
        self.engine = engine

        # Dynamic Status
        self.status: BallClassifiedObject = ball
        self.color = {
            BallClass.UNKNOWN: (186, 186, 186), # Gray
            BallClass.COMING: (196, 185, 37), # Yellow
            BallClass.GOING: (37, 196, 42), # Green
            BallClass.KICK: (196, 37, 37), # Red
            BallClass.STOPPED: (33, 185, 219), # Blue
        }

        self.GOAL_CENTER = Point(0, 7)
        self._create_image(Point(ball.dynamics.position[0] + self.GOAL_CENTER.x, ball.dynamics.position[1] + self.GOAL_CENTER.y))

    def _create_image(self, position_field: Point):
        BALL_RADIUS = 0.11  # 0.11 meters (standard soccer ball radius)
        self.radius = math.ceil(BALL_RADIUS * self.engine.get_scale())

        if self.radius % 2 == 0:  # If even, make it odd
            self.radius += 1

        self.board_width = 1  # 1 pixel
        self.image = pygame.Surface(
            (2 * self.radius, 2 * self.radius),
            pygame.SRCALPHA,
        )

        pygame.draw.circle(
            self.image, (0, 0, 0), (self.radius, self.radius), self.radius
        )

        pygame.draw.circle(
            self.image,
            self.color[self.status.classification],
            (self.radius, self.radius),
            self.radius - self.board_width,
        )  # Draw a circle (the ball)

        self.rect = self.image.get_rect()
        self.rect.center = (
            round(position_field.x * self.engine.get_scale())
            + self.engine.half_padding
            - self.radius,
            round(position_field.y * self.engine.get_scale())
            + self.engine.half_padding
            - self.radius,
        )

    def _draw_simulation_line(self):
        if self.status.properties.crossing_point[0] == float("inf") or self.status.properties.crossing_point[1] == float("inf"):
            return

        impact_x = (self.GOAL_CENTER.x + self.status.properties.crossing_point[0]) * self.engine.get_scale() + self.engine.half_padding
        impact_y = (self.GOAL_CENTER.y + self.status.properties.crossing_point[1]) * self.engine.get_scale() + self.engine.half_padding

        ball_x = (self.GOAL_CENTER.x + self.status.dynamics.position[0]) * self.engine.get_scale() + self.engine.half_padding
        ball_y = (self.GOAL_CENTER.y + self.status.dynamics.position[1]) * self.engine.get_scale() + self.engine.half_padding

        pygame.draw.line(
            self.engine.screen,
            (0, 0, 255),
            (ball_x, ball_y),
            (impact_x, impact_y),
            2,
        )

    def update(self):
        self._draw_simulation_line()

        self.rect.x = (
            round((self.GOAL_CENTER.x + self.status.dynamics.position[0]) * self.engine.get_scale())
            + self.engine.half_padding
            - self.radius
        )
        self.rect.y = (
            round((self.GOAL_CENTER.y + self.status.dynamics.position[1]) * self.engine.get_scale())
            + self.engine.half_padding
            - self.radius
        )

        ball_status_text_up = self.engine.font.render(f"ID: {self.status.object_id} -  Vx: {self.status.dynamics.velocity[0]:.2f}  Vy: {self.status.dynamics.velocity[1]:.2f}", True, (0, 0, 0))
        self.engine.screen.blit(ball_status_text_up, self.rect.center)

        ball_status_text_down = self.engine.font.render(f"X: {self.status.dynamics.position[0]:.2f} -  Y: {self.status.dynamics.position[1]:.2f}", True, (0, 0, 0))
        self.engine.screen.blit(ball_status_text_down, (self.rect.center[0], self.rect.center[1] + 10))
    
class Goalkeeper(pygame.sprite.Sprite):
    def __init__(self, engine: Engine, position_field: Point):
        super().__init__()
        self.engine = engine
        self.GOAL_CENTER = Point(0, 7)
        position_field.x += self.GOAL_CENTER.x
        position_field.y += self.GOAL_CENTER.y

        self.status: Dynamic = Dynamic(
            position=position_field, velocity=Point(0, 0), acceleration=Point(0, 0)
        )

        self.create_image(position_field)

    def create_image(self, position_field: Point):
        self.width = 0.52 * self.engine.get_scale()  # 0.52 meters
        self.height = 0.52 * self.engine.get_scale()
        self.board_width = 1  # 1 pixel

        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(
            self.image,
            (0, 0, 0),
            (self.width // 2, self.height // 2),
            self.width // 2,
        )

        pygame.draw.circle(
            self.image,
            (0, 0, 255),
            (self.width // 2, self.height // 2),
            (self.width - self.board_width) // 2,
        )

        number = self.engine.font.render("1", True, (0, 0, 0))
        self.image.blit(number, (self.width // 3, self.width // 8))

        self.rect = self.image.get_rect()
        self.rect.center = (
            position_field.x * self.engine.get_scale() + self.engine.half_padding,
            position_field.y * self.engine.get_scale() + self.engine.half_padding,
        )

    def update(self):
        pass


if __name__ == "__main__":
    engine = Engine()
    ball_classification: BallClassifiedObject = BallClassifiedObject(
        object_id=1,
        dynamics=ObjectDynamic(
            position=np.array([2, 6]),
            velocity=np.array([0, 0]),
            acceleration=np.array([0, 0]),
        ),
        classification=BallClass.STOPPED,
        properties=BallClassificationProperties(
            cycle_count=2,
            distance_to_goal=float("inf"),
            cycles_coming=0,
            cycles_going=0,
            time_to_goal=float("inf"),
            crossing_point=np.array([0, 0.5, 0]),
        ),
    )

    ball = Ball(engine, ball_classification)
    goalkeeper = Goalkeeper(engine, Point(0, 0))

    engine.add_ball(ball)
    engine.add_goalkeeper(goalkeeper)
    while engine.run():
        time.sleep(0.015)
        engine.clear()
        engine.add_ball(ball)
        engine.add_goalkeeper(goalkeeper)
        


    pygame.quit()
