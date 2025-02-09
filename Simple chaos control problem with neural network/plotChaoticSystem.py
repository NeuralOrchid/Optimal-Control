import numpy as np
from matplotlib import pyplot as plt
import torch
from config import DEVICE

# System state variables & constant parameters
C1, C2, C3, C4, C5 = 1.6, 3, 8, 11, 0.5
X0, Y0, Z0 = -0.5, 1.0, 0.5
systemStateVariables = (X0, Y0, Z0)

# The equations describing the dynamics of 3D autonomous chaotic system
X_dot = lambda x, y, z: y - C1*x + y*z
Y_dot = lambda x, y, z: C2*y - x*z
Z_dot = lambda x, y, z: C3*x*y - C4*z - C5*x*x
DYNAMICS = (X_dot, Y_dot, Z_dot)
DIM = len(DYNAMICS)

# an equilibrium point that used in loss function (Hamiltonian method)
EQUILIBRIUM_POINT = (0, 0, 0)

# Plot the 3d chaotic system
if __name__ == "__main__":

    dt = 1e-3
    num_steps = 100_000
    xyzs = np.empty((num_steps + 1, 3)) 
    xyzs[0] = systemStateVariables

    for time_step in range(num_steps):
        step = np.array([dynamic(*xyzs[time_step]) for dynamic in DYNAMICS])
        xyzs[time_step + 1] = xyzs[time_step] + step * dt
    
    # Plot
    x, y, z = xyzs.T
    fig = plt.figure()
    fig.set_label("Chaotic attractor of 3D system with a = 1.6, b = 3, c = 8, d = 11, h = 0.5")

    ## X-Y
    ax = fig.add_subplot(221)
    ax.plot(x, y, "b", lw=0.2)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.grid()

    ## X-Z
    ax = fig.add_subplot(222)
    ax.plot(x, z, "b", lw=0.2)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Z Axis")
    ax.grid()

    ## Y-Z
    ax = fig.add_subplot(223)
    ax.plot(y, z, "b", lw=0.2)
    ax.set_xlabel("Y Axis")
    ax.set_ylabel("Z Axis")
    ax.grid()

    ## 3D
    ax = fig.add_subplot(224, projection = '3d')
    ax.plot(x, y, z, "b", lw=0.2)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.grid()

    plt.show()


# Plot control
def plot_control_chaotic_system(
		x_mdl: list[torch.nn.Module],
		u_mdl: list[torch.nn.Module],
		start: float = 0,
		end: float = 4,
		steps: int = 1000
) -> None:
    fig, ax = plt.subplots(1, 2)
    fig.set_label("Optimal control of the chaotic and hyper-chaotic system")
    t = torch.linspace(start, end, steps, requires_grad=False).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        for i in range(DIM):
            x_mdl[i].eval()
            u_mdl[i].eval()

        x_plot, = ax[0].plot((t).cpu().squeeze().numpy(), x_mdl[0](t/(t+1)).cpu().squeeze().numpy(), "r-")
        y_plot, = ax[0].plot((t).cpu().squeeze().numpy(), x_mdl[1](t/(t+1)).cpu().squeeze().numpy(), "b--")
        z_plot, = ax[0].plot((t).cpu().squeeze().numpy(), x_mdl[2](t/(t+1)).cpu().squeeze().numpy(), "g-.")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("State")
        ax[0].grid()
        ax[0].legend((x_plot, y_plot, z_plot), ("x(τ)", "y(τ)", "z(τ)"))

        ux_plot, = ax[1].plot((t).cpu().squeeze().numpy(), u_mdl[0](t/(t+1)).cpu().squeeze().numpy(), "r-")
        uy_plot, = ax[1].plot((t).cpu().squeeze().numpy(), u_mdl[1](t/(t+1)).cpu().squeeze().numpy(), "b--")
        uz_plot, = ax[1].plot((t).cpu().squeeze().numpy(), u_mdl[2](t/(t+1)).cpu().squeeze().numpy(), "g-.")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("Control")
        ax[1].grid()
        ax[1].legend((ux_plot, uy_plot, uz_plot), ("u1(τ)", "u2(τ)", "u3(τ)"))

        for i in range(DIM):
            x_mdl[i].train()
            u_mdl[i].train()

        plt.show()