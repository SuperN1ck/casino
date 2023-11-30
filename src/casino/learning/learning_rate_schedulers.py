import enum
from typing import Optional
import dataclasses


class LearningRateScheduleType(enum.Enum):
    STEP = enum.auto()
    WARMUP = enum.auto()
    CONSTANT = enum.auto()
    LEVEL_DECAY = enum.auto()


@dataclasses.dataclass
class LearningRateScheduleConfig:
    type: LearningRateScheduleType = LearningRateScheduleType.CONSTANT
    initial: float = 1e-4
    interval: Optional[float] = None
    factor: Optional[float] = None
    final: Optional[float] = None
    length: Optional[float] = None


class LearningRateSchedule:
    def get_learning_rate(self, epoch: int):
        pass

    @staticmethod
    def get_from_config(cfg: LearningRateScheduleConfig):
        if cfg.type == LearningRateScheduleType.STEP:
            return StepLearningRateSchedule(
                cfg.initial,
                cfg.interval,
                cfg.factor,
            )
        elif cfg.type == LearningRateScheduleType.WARMUP:
            return WarmupLearningRateSchedule(
                cfg.initial,
                cfg.final,
                cfg.length,
            )

        elif cfg.type == LearningRateScheduleType.CONSTANT:
            return ConstantLearningRateSchedule(cfg.initial)
        elif cfg.type == LearningRateScheduleType.LEVEL_DECAY:
            return LevelDecayLearningRateSchedule(cfg.initial, cfg.factor)
        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(cfg.type)
            )


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value: float):
        self.value = value

    def get_learning_rate(self, epoch: int):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial: float, interval: int, factor: float):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial: float, warmed_up: float, length: int):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch: int):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


class LevelDecayLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial: float, decay: float):
        self.initial = initial
        self.decay = decay
        self.level = 0

    def inc_level(self, increment: int = 1):
        self.level += increment

    def get_learning_rate(self, epoch: int):
        """
        Epoch does not matter
        """
        return self.initial * ((self.decay) ** self.level)
    

# Torch Specific Implementations
def adjust_learning_rate(optimizer: "torch.optim.Optimizer", epoch: int):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["scheduler"].get_learning_rate(epoch)

