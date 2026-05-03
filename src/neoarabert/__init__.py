import warnings

# xformers 0.0.27/0.0.28 still call `torch.library.impl_abstract`, which is
# deprecated in newer PyTorch in favor of `torch.library.register_fake`. The
# emitted FutureWarning is noise — we don't control xformers, and pinning a
# different xformers version is the only fix. Suppress it once at package
# import; this fires before any submodule imports xformers.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.library\.impl_abstract` was renamed.*",
    category=FutureWarning,
)
