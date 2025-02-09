# Chaos Control Using Neural Networks
![https://opensource.org/licenses/MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project is a simple exploration of controlling a 3D chaotic system using neural networks. We utilize the Hamiltonian method and Pontryagin's Minimum Principle (PMP) to create a loss function that guides the training of a neural network aimed at stabilizing the chaotic behavior. While controlling chaos has important implications in fields like engineering and physics, this project serves as an introductory demonstration of how neural networks can be applied to chaos control.

## Key Features

- Chaotic System Control: Stabilizes a chaotic system sensitive to initial conditions.
- Hamiltonian Method: Models the dynamics of the chaotic system, allowing for manipulation to achieve control.
- Pontryagin's Minimum Principle (PMP): Determines optimal control inputs to minimize cost and stabilize the system.
- Neural Network (PyTorch): Implements a neural network to approximate a control function, trained using the derived loss function.
- Problem Transformation: Transforms the control problem from an unbounded (0 to ∞) to a bounded domain (0 to 1) for efficient optimization.

## Explanation
Before applying any control techniques, it's essential to understand the inherent chaotic behavior of the system. The following plot illustrates the complex and unpredictable trajectory of the uncontrolled system in 3D space:

<img src="plots/3d chaotic system.png" class="plot" alt="">

The equations describing the dynamics of 3D autonomous chaotic system can be written as follows:

$$ \begin{cases}
\dot x = y - az + yz \\
\dot y = by - xz \\
\dot z = cxy - dz - hx^2 \\
\end{cases} $$

Where $a=1.6,b=3,c=8,d=11$ and $h=0.5$ and $(x_0,y_0,z_0)=(-0.5,1,0.5)$

The controlled chaotic system is defined by

### Optimal control of the chaotic system

In this section, we study the optimal control problem of the autonomous chaotic systems about its equilibrium states. For the purpose of optimal control, we will apply the PMP

Consider a dynamic system modeled by the state equations

$$
\dot x_i(t) = f_i(x_1(t), x_2(t), ..., x_n(t)) \\
i = 0, 1, .., n
$$

The controlled chaotic system is defined by

$$
\dot x_i(t) = f_i(x_1(t), x_2(t), ..., x_n(t)) + u_i(t)
$$

Where $u_i(t), i=0,1,...,n$ are control inputs which will be satisfied from the conditions of the optimal dynamical system about its equilibrium points with respect to the cost function $J$. The proposed control strategy is designed to achieve in a given time $t_f$ to the equilibrium point with an optimal control inputs. The initial and final conditions are

$$
x_i(0)=x_{i,0}, x_i(t_f)= \bar x_i
$$

where $\bar x_i$, denote the coordinates of the equilibrium points. The objective functional to be minimized is defined as

$$
J = \frac{1}{2} \int^{t_f}\_{0}{ \sum^{n}_{i=1}{(\alpha_i(x_i-\bar x_i)^2 + \beta_i {u_i}^2)}dt }
$$

where $\alpha_i, \beta_i, (i=0,1,...,n)$ are positive constants.
$\alpha_1=10,\alpha_2=10,\alpha_3=10,\beta_1=2,\beta_2=5,\beta_3=10$.

The corresponding Hamiltonian function will be

$$
H = \frac{1}{2} \sum^{n}_{i=1}{(\alpha_i(x_i-\bar x_i)^2 + \beta_i u_i^2)}dt + \lambda_i(f_i + u_i)
$$

where $\lambda_i, i=0,1,...,n$ are co-state variables. Using this notation, the optimality conditions can be written as follows:

$$ \begin{cases}
\dot x = \frac{\partial H}{\partial \lambda_i} \\
\dot \lambda_i = - \frac{\partial H}{\partial x_i} \\
\frac{\partial H}{\partial u_i} = 0 \\
\end{cases} $$

The optimality conditions are mathematically formulated into a loss function. This allows the neural network to learn both the stabilization of the chaotic system and its optimal control, guided by the principles of optimality.

To facilitate efficient learning and computation, we employ a variable transformation that maps the original, unbounded time domain (ranging from 0 to infinity) onto a bounded interval [0, 1). This transformation allows the neural network to be trained over a finite horizon, significantly reducing the computational complexity and enabling practical implementation of the control strategy. By working within a bounded domain, we can effectively approximate the solution to the infinite-horizon optimal control problem.

After applying the neural network-based control strategy, the system's behavior changes dramatically. The following plot shows the evolution of the state variables $(x, y, z)$ and the control inputs $(u_1,u_2,u_3)$ over time, demonstrating the successful stabilization of the chaotic system:
<img src="plots/optimal control of the chaotic system.png" class="plot" alt="">

## Getting Started
### Modify the Configuration
To customize the training process, you can modify the `config.py` file.

### Change the Loss Function
To align the training with your specific system dynamics, you can change the loss function defined in `loss.py`. This file contains the implementation of the Hamiltonian method.

### Adjust Chaotic System Parameters
For a better fit to your research goals, you can change the parameters of the chaotic system in `plotChaoticSystem.py`. This file contains the configuration for the chaotic system, including initial conditions and etc

### Usage
- Training the Neural Network:

```bash
python main.py
```

### Prerequisites

- Python 3
- PyTorch
- NumPy
- Matplotlib

## Citations

This project builds upon the following research:

- Effati, S., Saberi-Nadjafi, J., & Saberi Nik, H. (2014). Optimal and adaptive control for a kind of 3D chaotic and 4D hyper-chaotic systems. Applied Mathematical Modelling, 38, 759-774.

- Ghasemi, S., Nazemi, A., & Hosseinpour, S. (2017). Nonlinear fractional optimal control problems with neural network and dynamic optimization schemes. Nonlinear Dynamics, 89, 2669-2682.


## License

This project is licensed under the MIT License

<style>
	.plot {
		border-radius: 10px;
	}
</style>
