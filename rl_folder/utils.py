import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional values
        if self.n_calls % 100 == 0:
            self.logger.record('train/learning_rate', self.model.lr_schedule(self.model._current_progress_remaining))
            self.logger.record('train/reward', np.mean(self.locals['rewards']))
        return True