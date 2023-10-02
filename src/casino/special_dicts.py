import operator
from typing import Dict
from collections import OrderedDict


class AccumulatorDict(dict):
    def __init__(self, *args, accumulator=operator.add, **kwargs):
        self.accumulator = accumulator
        self.update(*args, **kwargs)

    def increment(self, key, val):
        """
        This will increment the value for a given key
        """
        if dict.__contains__(self, key):
            val = self.accumulator(dict.__getitem__(self, key), val)
        dict.__setitem__(self, key, val)

    def increment_dict(self, other: Dict):
        for key, value in other.items():
            self.increment(key, value)


class IndexDict(dict):
    """
    This will incrementally count up for unseen keys
    """

    def __init__(self, counter_start: int = 0, *args, **kwargs):
        self.counter = counter_start
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            val = dict.__getitem__(self, key)
        else:
            val = self.counter
            dict.__setitem__(self, key, val)
            self.counter += 1
        return val

    # def __setitem__(self, key, val):
    #   print('SET', key, val)
    #   dict.__setitem__(self, key, val)

    # def __repr__(self):
    #   dictrepr = dict.__repr__(self)
    #   return '%s(%s)' % (type(self).__name__, dictrepr)

    # def update(self, *args, **kwargs):
    #   print('update', args, kwargs)
    #   for k, v in dict(*args, **kwargs).items():
    #     self[k] = v


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        assert "module." in k, "Trying to convert non data parallel checkpoint!"
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict
