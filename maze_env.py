import math
import mazesim
from pdb import set_trace as bb
import pygame
import time


class MazeEnv():
    def __init__(self, maze_file="hard_maze.txt",fragile=False):
        #fragile flag indicates if touching wall terminates simulation
        #typically left off

        self.orig_e = mazesim.Environment(str(maze_file))
        e = self.orig_e

        self.action_space = 2
        self.observation_space = e.get_sensor_size()
        self.state = None
        self.timesteps = e.steps
        self.maze_bounds = None
        self.midx = 0
        self.midy = 0
        self.endx = 0
        self.endy = 0
        self.render_heading = True

    def _step(self, action):
        #action is list of floats between 0 and 1
        #action[0] is turning [0 is full turn left, 0.5 is go straight, 1 is full turn right]
        #action[1] is forward backward [0 is full backward, 0.5 is stop, 1 is full forward]

        e = self.e
        e.interpret_outputs(float(action[0]), float(action[1]))
        e.Update()
        self.elapsed += 1
        reward = 0
        done = False

        if self.elapsed == self.timesteps // 2:
            self.midx, self.midy = self.e.hero.location.x, self.e.hero.location.y
        if self.elapsed == self.timesteps:
            self.endx, self.endy = self.e.hero.location.x, self.e.hero.location.y
            #reward signal is negative distance to target
            reward = -e.distance_to_target()
            done = True

        #if fragile domain then kill off individual
        if self.e.hero.collide:
            reward = -1e6
            self.endx, self.endy = 400,400
            done = True

        state = self.e.generate_neural_inputs_wrapper(self.observation_space)

        return state, reward, done, {}

    #can render out using pygame
    def _draw_agent(self,screen):
        x = self.e.hero.location.x
        y = self.e.hero.location.y
        rad = 8
        rect = (int(x), int(y), 5, 5)
        pygame.draw.circle(screen, (80, 80, 255), rect[:2], rad)
        heading = self.e.hero.heading
        deltax = int(math.cos(heading / 180.0 * 3.14) * rad)
        deltay = int(math.sin(heading / 180.0 * 3.14) * rad)
        pygame.draw.line(screen, (255, 0, 0), (x, y), (x + deltax, y + deltay),
                         3)

    def _draw_walls(self,screen):
        lines = self.e.get_line_count()

        for idx in range(lines):
            line = self.e.get_line(idx)
            start_pos = int(line.a.x), int(line.a.y)
            end_pos = int(line.b.x), int(line.b.y)
            pygame.draw.line(screen, (0, 0, 0), start_pos, end_pos, 2)

    def _render(self, screen):
        screen.fill((255, 255, 255))

        self._draw_agent(screen)
        self._draw_walls(screen)

        pygame.display.flip()
        #time.sleep(0.05)

    def _trace(self):
        return self.e.hero.heading,self.e.hero.location.x, self.e.hero.location.y

    def _behavior(self):
        #return self.midx,self.midy,self.endx,self.endy
        return self.endx, self.endy

    def _reset(self):
        self.e = mazesim.Environment(self.orig_e)
        self.elapsed = 0
        state = self.e.generate_neural_inputs_wrapper(self.observation_space)
        return state


if __name__ == '__main__':
    maze = MazeEnv()

    import pygame
    from pygame.locals import *
    pygame.init()

    pygame.display.set_caption('Viz')
    SZX=SZY=400
    screen = pygame.display.set_mode((SZX, SZY))

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))

    max_episode_length = 400

    maze._reset()

    # Rollout
    for step in range(max_episode_length):

        action = [0.2,1.0]

        maze._render(screen)
        time.sleep(0.01)

        state, reward, done, _ = maze._step(action)

        print state

        if done:
            break

    print reward
