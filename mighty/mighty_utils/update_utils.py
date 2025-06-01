def polyak_update(source_params, target_params, tau: float):
    """Polyak averaging for target network updates."""
    for source, target in zip(source_params, target_params):
        target.data.copy_(tau * source.data + (1 - tau) * target.data)
