from collections import deque
from DDQN.utils.general_utils import *
from DDQN.utils.general_utils import Memory


class ExperienceReplay:
    def __init__(self, type=''):
        if type.lower() == REPLAY_PRIORITIZED:
            self.prioritized_state = True
            self.replay_exp = Memory(BUFFER_SIZE)
        else:
            self.prioritized_state = False
            self.replay_exp = deque(maxlen=BUFFER_SIZE)
        self.size = 0

    def is_prioritized(self):
        return self.prioritized_state

    def memorize_exp(self, sample, prioritized_args):
        if self.is_prioritized():
            if prioritized_args['is_update'] is False:
                self.replay_exp.add(prioritized_args['e'], sample)
                self.size += 1
            else:
                mini_batch = sample
                for i in range(len(mini_batch)):
                    idx = mini_batch[i][0]
                    self.replay_exp.update(idx, prioritized_args['e'][i])
        else:
            self.replay_exp.append(sample)
            self.size += 1

    # def update(self, ):
    #     if self.is_prioritized():
