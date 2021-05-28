"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
from loguru import logger
import os
from utils.torch_utils import seed_everything


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        log_path = os.path.join(self.cfg.save.log_dir, f"{self.cfg.model.model_name}_{{time:YYYY-MM-DD-HH-mm}}.log")
        logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        logger.info(cfg)
        seed_everything(self.cfg.solver.seed)  # seed everything

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError
