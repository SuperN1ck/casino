import gc
import logging
from typing import Iterable, List

try:
    import torch
except:
    logging.debug("torch not availble. Most functionality in torch_utils will break.")


def torch_batched_eye(B: int, N: int, **kwargs):
    return torch.eye(N, **kwargs).reshape(1, N, N).repeat(B, 1, 1)


def clear_torch_memory():
    gc.collect()
    torch.cuda.empty_cache()


try:

    class SplittableTorchTensor(torch.Tensor):
        channel_dim: int
        split_channels: List[int]

        @staticmethod
        def __new__(
            cls, input_tensor_group: Iterable["torch.Tensor"], channel_dim: int = 0
        ):
            # First gather the splitting indices
            split_channels = []
            for x in input_tensor_group:
                split_channels.append(x.shape[channel_dim])
            single_tensor = torch.cat(input_tensor_group, dim=channel_dim)
            single_tensor = single_tensor.as_subclass(cls)
            single_tensor.channel_dim = channel_dim
            single_tensor.split_channels = split_channels
            return single_tensor

        def split_into_original_groups(self):
            """
            Splits the tensor into the originally provided groups and
            """
            return [
                split_tensor.as_subclass(torch.Tensor)
                for split_tensor in torch.split(
                    self, self.split_channels, dim=self.channel_dim
                )
            ]

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):

            # Do the actual function call
            ret = super().__torch_function__(func, types, args, kwargs)

            all_subtensors = [arg for arg in args if isinstance(arg, cls)]

            # Not sure how this could happen, but in case we do not have any splittable tensors here,
            # we just return the result
            # TODO --> Verify that it does not break anything in here
            if len(all_subtensors) == 0:
                return ret

            first_tensor = all_subtensors[0]
            assert all(
                [
                    subtensor.channel_dim == first_tensor.channel_dim
                    and subtensor.split_channels == first_tensor.split_channels
                    for subtensor in all_subtensors
                ]
            )

            # Restore metaparameter state
            # Assumes in-place
            def _assign_metaparameters(_ret):
                if isinstance(_ret, cls):
                    _ret.channel_dim = first_tensor.channel_dim
                    _ret.split_channels = first_tensor.split_channels
                elif isinstance(_ret, tuple):
                    for t_ret in _ret:
                        _assign_metaparameters(t_ret)
                else:
                    ...

            _assign_metaparameters(ret)
            return ret

except:
    logging.debug("torch not availble. SplittableTorchTensor not defined")
