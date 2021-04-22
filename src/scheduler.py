from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, CosineAnnealingLR, LambdaLR
)


def ups_scheduler(optimizer, milestones, gamma, warmup=0):
    assert len(milestones) > 0
    assert warmup < milestones[0]
    from bisect import bisect_right  # do not pollute scheduler.py namespace

    def lr_policy(step):
        init_multiplier = 1.0
        if step < warmup:
            return (0.1 + 0.9 * step / warmup) * init_multiplier
        exponent = bisect_right(milestones, step)
        return init_multiplier * (gamma ** exponent)

    return LambdaLR(optimizer, lr_lambda=lr_policy)
