slowness_space = torch.tensor(
    [(torch.cos(az) * s, torch.sin(az) * s) for az, s in product(azs, slows)]
)
