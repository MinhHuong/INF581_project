import numpy as np

class Converter:
    def __init__(self, env):
        self.width = env.width
        self.height = env.height
        self.pairs = [(i, j) for i in range(self.width + 2) for j in range(self.height + 2)]
    def state2tile(self, state):
        i = state[0] * (self.width + 2) + state[1] + state[2] * (self.width + 2) * (self.height + 2)
        if i < 2 * (self.width + 2) * (self.height + 2):
            return i
        return -1
    def tile2state(self, tile):
        if tile < 0:
            return None
        elif tile > (self.height + 2) * (self.width + 2):
            tile = tile - (self.height + 2) * (self.width + 2)        # i - the state where the position contains trash
            state = np.append(np.array(self.pairs[int(tile)]), 1)
        else:
            state = np.append(np.array(self.pairs[int(tile)]), 0)
        return state