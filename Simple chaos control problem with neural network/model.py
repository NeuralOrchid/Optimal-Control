import torch
import torch.nn as nn
from typing import Callable, Optional


class NeuralNetwork(nn.Module):
	def __init__(
			self,
			p1: Optional[tuple[float]] = None,
			p2: Optional[tuple[float]] = None,
	):
		"""η 
        ---
        Neural network with initial points
        
        ### Parameters
        p1 : (float, float)
            initial point. [t_0, x_0]
        p2 : (float, float)
            final point. [t_f, x_f]
		"""
		super(NeuralNetwork, self).__init__()
		self.in_proj = nn.Linear(1, 8)
		self.activation = nn.Tanh()
		self.out_proj = nn.Linear(8, 1, bias=False)

		if p1 == None:
			self.final = None
		else:
			self.final = self._initial_condition(p1, p2)

	def _initial_condition(
			self,
			p1: tuple[float],
			p2: Optional[tuple[float]] = None,
	) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
		if p2 == None:
			t_0, x_0 = p1
			return lambda t, x: x_0 + ((t - t_0) * x)
		else:
			t_0, x_0 = p1
			t_f, x_f = p2
			m = (x_f - x_0)/(t_f - t_0)
			c = x_0 - (m * t_0)
			return lambda t, x: m * t + c + ((t - t_0) * (t - t_f) * x)

	def forward(self, t:torch.Tensor) -> torch.Tensor:
		x = self.in_proj(t/(1-t))
		x = self.activation(x)
		x = self.out_proj(x)

		if self.final != None:
			x = self.final(t, x)

		return x
	
