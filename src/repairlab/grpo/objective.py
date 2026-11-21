"""Classic clipped trajectory-normalized loss. Sampler probabilities are not inputs."""
def clipped_loss(new,old,advantage,trajectory_count,trajectory_tokens,epsilon):
    import torch
    if trajectory_count<=0 or trajectory_tokens<=0:raise ValueError('Nonempty trajectory batch required')
    ratio=(new-old.detach()).exp()
    loss=-torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage).sum()/(trajectory_count*trajectory_tokens)
    return loss,ratio
