# Python imports
import random
from typing import List

# pymunk imports
import pymunk


class BouncyBalls(object):

    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 900.0)

        self.parameters = []
        self.history = []

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1


        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls: List[pymunk.Circle] = []
        self._create_ball()

        
    def run(self) -> None:
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self._balls[0].body.position.x < 600:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            self._draw_objects()


    def _add_static_scenery(self) -> None:
        """
        Create the static bodies.
        :return: None
        """
        static_body = self._space.static_body

        self._yh = random.randint(50,400)

        static_lines = [
            pymunk.Segment(static_body, (0.0, 600 - self._yh), (600.0, 600), 0.0),
            #pymunk.Segment(static_body, (407.0, 600 - 246), (407.0, 600 - 343), 0.0),
        ]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.9
        self._space.add(*static_lines)


        self.parameters.append(self._yh)


    def _create_ball(self) -> None:
        """
        Create a ball.
        :return:
        """
        mass = 5
        radius = random.randint(10, 30)
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(20, 60)
        y = random.randint(600 - self._yh - 100, 600 - self._yh - radius)
        body.position = x, y
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = random.uniform(0.8, 1)
        shape.friction = random.uniform(0.5, 1)
        self._space.add(body, shape)
        self._balls.append(shape)

        self.parameters += [radius, shape.elasticity, shape.friction]

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        
        #self._space.debug_draw(self._draw_options)
        self.history.append([self._balls[0].body.position.x, self._balls[0].body.position.y,
                            self._balls[0].body.velocity.x, self._balls[0].body.velocity.y] + self.parameters)


import torch
from torch.utils.data.dataset import IterableDataset


class BouncyBallsData(IterableDataset):
    def __init__(self, use_cuda=False):
        super().__init__()

    def __iter__(self):

        # Observe a bounce action
        
        for _ in range(100):
            game = BouncyBalls()
            game.run()
            xs = torch.Tensor(game.history)[:-1, :]
            ys = torch.Tensor(game.history)[:, :4].roll(-1, dims=0)[:-1, :]

            for x, y in zip(xs, ys):
                yield x, y


class BouncyBallsDataStatic(IterableDataset):
    def __init__(self, games=100, use_cuda=False):
        super().__init__()

        self.xs = []
        self.ys = []

        for _ in range(games):
            game = BouncyBalls()
            game.run()
            self.xs.append(torch.Tensor(game.history)[:-1, :])
            self.ys.append(torch.Tensor(game.history)[:, :4].roll(-1, dims=0)[:-1, :])

        self.xs = torch.cat(self.xs, dim=0)
        self.ys = torch.cat(self.ys, dim=0)

    def __iter__(self):
        
        for x, y in zip(self.xs, self.ys):
            yield x, y


class BouncyBallsDataBounce(IterableDataset):
    def __init__(self, use_cuda=False):
        super().__init__()
        self.theoretical_dvy = 15
        self.max_num_err = 0

    def __iter__(self):

        # Observe a bounce action
        
        for _ in range(1000):
            game = BouncyBalls()
            game.run()
            xs = torch.Tensor(game.history)[:-1, :]
            ys = torch.Tensor(game.history)[:, :4].roll(-1, dims=0)[:-1, :]

            dvy = (ys[:,3] - xs[:,3])
            self.max_num_err = max(self.max_num_err, dvy.detach().numpy().max() - self.theoretical_dvy)

            selection = dvy < self.theoretical_dvy - self.max_num_err
            xs = xs[selection]
            ys = ys[selection]

            for x, y in zip(xs, ys):
                yield x, y

class BouncyBallsDataBounceRatio(IterableDataset):
    def __init__(self, bounce_ratio, use_cuda=False):
        super().__init__()
        self.theoretical_dvy = 15
        self.max_num_err = 0
        self.bounce_ratio = bounce_ratio

    def __iter__(self):

        # Observe a bounce action
        
        for _ in range(1000):
            game = BouncyBalls()
            game.run()
            xs = torch.Tensor(game.history)[:-1, :]
            ys = torch.Tensor(game.history)[:, :4].roll(-1, dims=0)[:-1, :]

            dvy = (ys[:,3] - xs[:,3])
            self.max_num_err = max(self.max_num_err, dvy.detach().numpy().max() - self.theoretical_dvy)

            bounces = dvy < self.theoretical_dvy - self.max_num_err

            # select bounces and non-bounces according to the desired ratio
            selection = bounces.detach().clone() * 0 + 1
            desired_bounces = int(xs.shape[0] * self.bounce_ratio)
            realized_bounces = bounces.sum()

            if desired_bounces < realized_bounces:
                idx = torch.randperm(realized_bounces)[desired_bounces:]
                discard_bounces = bounces.nonzero(as_tuple=True)[0][idx]
                selection[discard_bounces] = 0
            elif desired_bounces > realized_bounces:
                desired_non_bounces = int(xs.shape[0] * (1 - self.bounce_ratio) * (realized_bounces / desired_bounces))
                realized_non_bounces = (~bounces).sum()
                idx = torch.randperm(realized_non_bounces)[desired_non_bounces:]
                discard_non_bounces = (~bounces).nonzero(as_tuple=True)[0][idx]
                selection[discard_non_bounces] = 0

            xs = xs[selection.nonzero(as_tuple=True)[0]]
            ys = ys[selection.nonzero(as_tuple=True)[0]]

            for x, y in zip(xs, ys):
                yield x, y

class BouncyBallsDataBounceRatioLabels(IterableDataset):
    def __init__(self, bounce_ratio, use_cuda=False):
        super().__init__()
        self.theoretical_dvy = 15
        self.max_num_err = 0
        self.bounce_ratio = bounce_ratio

    def __iter__(self):

        # Observe a bounce action
        
        for _ in range(1000):
            game = BouncyBalls()
            game.run()
            xs = torch.Tensor(game.history)[:-1, :]
            ys = torch.Tensor(game.history)[:, :4].roll(-1, dims=0)[:-1, :]

            dvy = (ys[:,3] - xs[:,3])
            self.max_num_err = max(self.max_num_err, dvy.detach().numpy().max() - self.theoretical_dvy)

            bounces = dvy < self.theoretical_dvy - self.max_num_err

            # select bounces and non-bounces according to the desired ratio
            selection = bounces.detach().clone() * 0 + 1
            desired_bounces = int(xs.shape[0] * self.bounce_ratio)
            realized_bounces = bounces.sum()

            if desired_bounces < realized_bounces:
                idx = torch.randperm(realized_bounces)[desired_bounces:]
                discard_bounces = bounces.nonzero(as_tuple=True)[0][idx]
                selection[discard_bounces] = 0
            elif desired_bounces > realized_bounces:
                desired_non_bounces = int(xs.shape[0] * (1 - self.bounce_ratio) * (realized_bounces / desired_bounces))
                realized_non_bounces = (~bounces).sum()
                idx = torch.randperm(realized_non_bounces)[desired_non_bounces:]
                discard_non_bounces = (~bounces).nonzero(as_tuple=True)[0][idx]
                selection[discard_non_bounces] = 0

            xs = xs[selection.nonzero(as_tuple=True)[0]]
            ys = ys[selection.nonzero(as_tuple=True)[0]]
            ls = bounces.int().unsqueeze(1)[selection.nonzero(as_tuple=True)[0]]
            for x, y, l in zip(xs, ys, ls):
                yield x, y, l