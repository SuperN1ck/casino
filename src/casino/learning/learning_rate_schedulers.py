import dataclasses
import enum
import math
from typing import Optional


class LearningRateScheduleType(enum.Enum):
    STEP = enum.auto()
    WARMUP = enum.auto()
    CONSTANT = enum.auto()
    LEVEL_DECAY = enum.auto()
    COSINE = enum.auto()


# TODO [NH] 2025/05/23: Maybe should be a config class for each type of schedule?
@dataclasses.dataclass
class LearningRateScheduleConfig:
    type: LearningRateScheduleType = LearningRateScheduleType.CONSTANT
    initial_lr: float = 1e-4
    interval_step: Optional[float] = None
    decay_factor: Optional[float] = None
    final_lr: Optional[float] = None
    length: Optional[float] = None

    def get_scheduler(self):
        """Instantiate the learning rate schedule from the config"""
        # [NH] 2025/05/21: It is interesting that this patterns works, as the
        #                  config calls the class, which required the config.
        return LearningRateSchedule.get_from_config(self)


class LearningRateSchedule:
    def get_learning_rate(self, epoch: int):
        pass

    @staticmethod
    def get_from_config(cfg: LearningRateScheduleConfig):
        if cfg.type == LearningRateScheduleType.STEP:
            return StepLearningRateSchedule(
                initial_lr=cfg.initial_lr,
                interval_step=cfg.interval_step,
                decay_factor=cfg.decay_factor,
            )
        elif cfg.type == LearningRateScheduleType.WARMUP:
            return WarmupLearningRateSchedule(
                initial_lr=cfg.initial_lr,
                warmed_up_lr=cfg.final_lr,
                length=cfg.length,
            )

        elif cfg.type == LearningRateScheduleType.CONSTANT:
            return ConstantLearningRateSchedule(lr=cfg.initial_lr)
        elif cfg.type == LearningRateScheduleType.LEVEL_DECAY:
            return LevelDecayLearningRateSchedule(
                initial_lr=cfg.initial_lr, decay_factor=cfg.decay_factor
            )
        elif cfg.type == LearningRateScheduleType.COSINE:
            return CosineLearningRateSchedule(
                initial_lr=cfg.initial_lr,
                final_lr=cfg.final_lr,
                T_max=cfg.length,
            )
        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(cfg.type)
            )


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_learning_rate(self, epoch: int):
        return self.lr


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial_lr: float, interval_step: int, decay_factor: float):
        self.initial_lr = initial_lr
        self.interval_step = interval_step
        self.decay_factor = decay_factor

    def get_learning_rate(self, epoch):
        return self.initial_lr * (self.decay_factor ** (epoch // self.interval_step))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial_lr: float, warmed_up_lr: float, length: int):
        self.initial_lr = initial_lr
        self.warmed_up_lr = warmed_up_lr
        self.length = length

    def get_learning_rate(self, epoch: int):
        if epoch > self.length:
            return self.warmed_up_lr
        return (
            self.initial_lr
            + (self.warmed_up_lr - self.initial_lr) * epoch / self.length
        )


class LevelDecayLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial_lr: float, decay_factor: float):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.level = 0

    def inc_level(self, increment: int = 1):
        self.level += increment

    def get_learning_rate(self, epoch: int):
        """
        Epoch does not matter
        """
        return self.initial_lr * ((self.decay_factor) ** self.level)


class CosineLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial_lr: float, final_lr: float, T_max: int):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.T_max = T_max

    def get_learning_rate(self, epoch: int):
        if epoch >= self.T_max:
            return self.final_lr
        # Cosine decay formula
        # lr = final + (initial - final) * (1 + cos(pi * epoch / T_max)) / 2
        return (
            self.final_lr
            + (self.initial_lr - self.final_lr)
            * (1 + math.cos(math.pi * epoch / self.T_max))
            / 2
        )


# Torch Specific Implementations
def adjust_learning_rate(optimizer: "torch.optim.Optimizer", epoch: int):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["scheduler"].get_learning_rate(epoch)


### Example usage ###
def plot_examples():
    import matplotlib.pyplot as plt
    import numpy as np

    MAX_EPOCHS = 100

    epochs = np.arange(0, MAX_EPOCHS)

    # # Example usage
    # Set some dummy values for the config
    config = LearningRateScheduleConfig(
        initial_lr=0.1,
        interval_step=10,
        decay_factor=0.5,
        final_lr=0.01,
        length=50,
    )

    for schedule_type in LearningRateScheduleType:
        print(f"Testing schedule type: {schedule_type}")
        config.type = schedule_type
        # scheduler = LearningRateSchedule.get_from_config(config)
        scheduler = config.get_scheduler()

        if config.type is not LearningRateScheduleType.LEVEL_DECAY:
            lr_values = [scheduler.get_learning_rate(epoch) for epoch in epochs]
        else:
            lr_values = []
            decay_levels = [40, 70]
            for epoch in epochs:
                if epoch in decay_levels:
                    scheduler.inc_level()
                lr_values.append(scheduler.get_learning_rate(epoch))

        plt.plot(epochs, lr_values, label=scheduler.__class__.__name__)

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_examples()
