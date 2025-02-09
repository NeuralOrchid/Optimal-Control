import torch
import torch.nn as nn
import torch.nn.functional as F

from plotChaoticSystem import DIM

def loss_fn(
        t: torch.Tensor,
        X: tuple[nn.Module],
        U: tuple[nn.Module],
        *,
        dim: int = DIM,
) -> torch.Tensor:
    """Compact loss function appling the Hamiltonian for this specific chaos control"""
    # Coefficient
    dt = (1 - t)**2

    # Simple verion of error/loss
    x_t = [x(t) for x in X]
    u_t = [u(t) for u in U]

    du_dt = [
        torch.autograd.grad(
            (dt * u_t[i]).sum(), 
            t, 
            create_graph=True,
        )[0] 
        for i in range(dim)
    ]
    dx_dt = [
        torch.autograd.grad(
            (dt * x_t[i]).sum(), 
            t, 
            create_graph=True,
        )[0] 
        for i in range(dim)
    ]

    x, y, z = x_t
    u1, u2, u3 = u_t

    E1 = F.mse_loss(
        9*x + 3.2*u1 + z*u2*5 - 80*u3,
        2*du_dt[0]
    )
    E2 = F.mse_loss(
        10*y - 2*(z+1)*u1 - 15*u2 - 80*x*u3,
        5*du_dt[1]
    )
    E3 = F.mse_loss(
        10*z - 2*y*u1 + 5*x*u2 + 110*u3,
        10*du_dt[2]
    )
    E4 = F.mse_loss(
        y*(z+1) - 1.6*x + u1,
        dx_dt[0]
    )
    E5 = F.mse_loss(
        3*y - x*z + u2,
        dx_dt[1]
    )
    E6 = F.mse_loss(
        x*(8*y - 0.5*x) - 11*z,
        dx_dt[2]
    )

    loss = E1 + E2 + E3 + E4 + E5 + E6

    return loss

# # Implementation of Hamiltonian method
# x_t = [x(t) for x in X]
# u_t = [u(t) for u in U]
# λ_t = [λ(t) for λ in Λ]

# # min J = ∫ f(t, x, u) s.t dX/dt = g(t, x, u)
# f_t = 0.5 * sum([(x_t[i] - E[i])**2 + u_t[i]**2 for i in range(dim)])
# g_t = [dynamics[i](*x_t) + u_t[i] for i in range(dim)]

# # Hamiltonian
# h = dt * (f_t + sum([λ_t[i] * g_t[i] for i in range(dim)]))

# dh_dx = [torch.autograd.grad(h.sum(), x_t[i], create_graph=True)[0] for i in range(dim)]
# dh_dλ = [torch.autograd.grad(h.sum(), λ_t[i], create_graph=True)[0] for i in range(dim)]
# dh_du = [torch.autograd.grad(h.sum(), u_t[i], create_graph=True)[0] for i in range(dim)]

# dλ_dt = [torch.autograd.grad(λ_t[i].sum(), t, create_graph=True)[0] for i in range(dim)]
# dx_dt = [torch.autograd.grad(x_t[i].sum(), t, create_graph=True)[0] for i in range(dim)]

# # Errors
# E1 = sum([F.mse_loss((dh_dx[i] + dλ_dt[i]), torch.zeros_like(t).to(DEVICE)) for i in range(dim)])
# E2 = sum([F.mse_loss((dh_dλ[i] - dx_dt[i]), torch.zeros_like(t).to(DEVICE)) for i in range(dim)])
# E3 = sum([F.mse_loss((dh_du[i]), torch.zeros_like(t).to(DEVICE)) for i in range(dim)])

# loss = E1 + E2 + E3
