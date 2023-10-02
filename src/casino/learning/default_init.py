def default_init_torch(module: "torch.nn.Module"):
    """Initialize parameters of the module.

    For convolution, weights are initialized by Kaiming method and
    biases are initialized to zero.
    For batch normalization, scales and biases are set to 1 and 0,
    respectively.
    """
    import torch

    # Make sure we don't overwrite previous weights
    # when weights are frozen (i.e. no gradient required)
    # TODO Does this cover all frozen cases?
    if hasattr(module, "weight") and not module.weight.requires_grad:
        return

    if (
        isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)
        or isinstance(module, torch.nn.Conv3d)
    ):
        torch.nn.init.kaiming_normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.BatchNorm2d) or isinstance(
        module, torch.nn.BatchNorm3d
    ):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
