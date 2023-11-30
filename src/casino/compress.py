import copy
import pickle
import io
import logging
from typing import List, Tuple
import functools

try:
    import zstd
except:
    logging.debug("zstd not availble. Some functionality in compress.py will break")


# From here: https://github.com/robot-learning-freiburg/CARTO/blob/07549580b8f2a42096cb7c8fb1486ceb74157e68/CARTO/simnet/lib/datapoint.py#L279C1-L295C13
def compress_datapoint(x):
    x = copy.deepcopy(x)
    buf = pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
    cbuf = zstd.compress(buf)
    return cbuf


def decompress_datapoint(cbuf, unpickler=pickle.Unpickler):
    buf = zstd.decompress(cbuf)
    x = unpickler(io.BytesIO(buf)).load()
    return x

# TODO Maybe move this to somewhere else?
def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class ReplaceUnpickler(pickle.Unpickler):
    def __init__(self, replacements: List[Tuple[str]] = [], *args, **kwargs):
        """
        Original Unpickler, but can replace modules, i.e. useful when refactoring the code

        replacements = [(old.module.path, new.module.path)]
        """
        super(ReplaceUnpickler, self).__init__(*args, **kwargs)
        self.replacements = replacements

    def find_class(self, module, name):
        renamed_module = module

        for to_replace, with_replacement in self.replacements:
            renamed_module = renamed_module.replace(to_replace, with_replacement)
        return super(ReplaceUnpickler, self).find_class(renamed_module, name)

    @staticmethod
    def with_replacements(replacements: List[Tuple[str]] = []):
        """
        Creates a callable pickle.Unpickler class replacing the modules
        """
        # return functools.partial(ReplaceUnpickler.__init__, replacements)
        return partialclass(ReplaceUnpickler, replacements)

    # TODO Untested
    # @staticmethod
    # def pickle_module_with_replacements(replacements: List[Tuple[str]] = []):
    #     pickle = imp.load_module("pickle2", *imp.find_module("pickle"))
    #     pickle.Unpickler = ReplaceUnpickler.with_replacements(replacements)
    #     return pickle
