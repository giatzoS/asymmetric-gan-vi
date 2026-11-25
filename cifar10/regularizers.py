# regularizers.py
import torch

def grad_norm_sq_mean(D, x):
    """
    Returns E ||∇_x D(x)||^2   (NO gamma, NO 1/2 factor)
    """
    x = x.detach().requires_grad_(True)
    d = D(x)
    grad = torch.autograd.grad(
        outputs=d.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return grad.pow(2).reshape(x.size(0), -1).sum(1).mean()

def asymmetric_gp_real_fake(D, real, fake_detached, gamma):
    """
    (gamma/2) * [ E||∇D(real)||^2 + E||∇D(fake)||^2 ]
    """
    if gamma <= 0:
        return torch.tensor(0.0, device=real.device)

    gp_real = grad_norm_sq_mean(D, real)
    gp_fake = grad_norm_sq_mean(D, fake_detached)

    return 0.5 * gamma * (gp_real + gp_fake)

def tikhonov_params(D, gamma):
    """
    (gamma/2) ||D||^2
    """
    if gamma <= 0:
        return torch.tensor(0.0, device=next(D.parameters()).device)

    tik = torch.tensor(0.0, device=next(D.parameters()).device)
    for p in D.parameters():
        tik = tik + p.pow(2).sum()
    return 0.5 * gamma * tik

