import enum


class CallableEnum(enum.Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

