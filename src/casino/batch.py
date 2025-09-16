import logging

try:
    import torch
except:
    logging.debug(
        "torch not available. Some functionality in pointcloud.py will break."
    )


def reduce_to_single_batch_dim_th(
    x: "torch.Tensor", data_start_dim: int = -1
) -> "torch.Tensor":
    """
    Given a tensor x, reduce all dimenions up unitl data_dim to a single dimension.
    This is useful for flattening the batch dimension of a tensor while keeping the
    data dimension intact. For example, if x has shape (batch_size, num_frames, 3, 64, 64),
    this function will return a tensor with shape (batch_size * num_frames, 3, 64, 64).
    It assumes that the data dimension is the last dimension of the tensor and batches come first.
        More abstract it does
        B_1 x B_2 x ... x B_n x D_1 x D_2 x ... x D_m
        to
        (B_1 * B_2 * ... * B_n) x D_1 x D_2 x ... x D_m

    It also returns a lambda function that takes a batch-flat tensor and returns the original tensor size.
        (batch_size * num_frames, 3, 64, 64) --> (batch_size * num_frames, 3, 64, 64)
    but also
        (batch_size * num_frames, 3) --> (batch_size * num_frames, 3)
    would work. The reshape function is not dependent on the new data dimension.

    Args:
        x (torch.Tensor): The input tensor to be flattened.
        data_start_dim (int): The dimension of the batch. Default is 0.
    Returns:
        torch.Tensor: The flattened tensor.
        lambda: A lambda function that takes a batch-flat tensor and returns the original tensor size.
    """
    batches_shape = x.shape[:data_start_dim]

    # Flatten the batch dimension
    x_f = x.flatten(end_dim=data_start_dim - 1)

    # Return the reshaped tensor
    def reshape_fn(x: "torch.Tensor"):
        """
        Reshape the batch-flat tensor back to its original shape.
        It is important to note that the data dimensions can be different
        Args:
            x (torch.Tensor): The batch-flat tensor to be reshaped.
        Returns:
            torch.Tensor: The reshaped tensor.
        """
        assert (
            x.shape[0] == batches_shape.numel()
        ), "The batch size of the reshaped tensor does not match the original batch sizes"
        data_shape = x.shape[1:]

        return x.reshape(
            (*batches_shape, *data_shape)
        )  # Reshape to original batch shape

    assert (
        x.shape == reshape_fn(x_f).shape
    ), "Reshape function does not return the same shape as input tensor"

    return x_f, reshape_fn
