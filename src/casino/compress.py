import copy
import pickle
import zstd
import io


# From here: https://github.com/robot-learning-freiburg/CARTO/blob/07549580b8f2a42096cb7c8fb1486ceb74157e68/CARTO/simnet/lib/datapoint.py#L279C1-L295C13
def compress_datapoint(x):
    x = copy.deepcopy(x)
    buf = pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
    cbuf = zstd.compress(buf)
    return cbuf


def decompress_datapoint(cbuf):
    buf = zstd.decompress(cbuf)
    x = pickle.Unpickler(io.BytesIO(buf)).load()
    return x
