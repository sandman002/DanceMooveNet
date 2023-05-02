import math
import random


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np
import math
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
def t2v(tau, f, out_features, w, b, w0, b0):
	v1 = f(torch.matmul(tau, w) + b)
	v2 = torch.matmul(tau, w0) + b0

	return torch.cat([v1, v2], 1)

class EqualLinear(nn.Module):
	def __init__(
		self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
	):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

		else:
			self.bias = None

		self.activation = activation
		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul

	def forward(self, input):
		if self.activation:

			# print(input.shape, self.weight.shape, )
			out = F.linear(input, self.weight * self.scale)
			out = fused_leaky_relu(out, self.bias * self.lr_mul)

		else:
			# print("asdf---", input.shape, self.weight.shape)
			out = F.linear(
				input, self.weight * self.scale, bias=self.bias * self.lr_mul
			)

		return out

	def __repr__(self):
		return (
			f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
		)

class SineActivation(nn.Module):
	def __init__(self, inFeat, outFeat):
		super(SineActivation, self).__init__()
		self.out_features = outFeat
		self.in_features = inFeat
		self.w0 = nn.parameter.Parameter(torch.randn(self.in_features, 1))
		self.b0 = nn.parameter.Parameter(torch.randn(1))

		self.w = nn.parameter.Parameter(torch.randn(self.in_features, self.out_features-1))
		self.b = nn.parameter.Parameter(torch.randn(self.out_features-1))

		self.f = torch.sin

	def forward(self, tau):
		return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class SineWave_Mixer(nn.Module):
	def __init__(self, inFeat, outFeat):
		super(SineWave_Mixer, self).__init__()
		self.out_features = outFeat
		self.in_features = inFeat
		self.w0 = nn.parameter.Parameter(torch.randn(self.in_features, 1))
		self.b0 = nn.parameter.Parameter(torch.randn(1))

		self.w = nn.parameter.Parameter(torch.randn(self.in_features, self.out_features-1))
		self.b = nn.parameter.Parameter(torch.randn(self.out_features-1))

		self.f = torch.sin

	def forward(self, x, tau):
		y = t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

		return x*y

class TemporalCodeGen(nn.Module): #for the dynamic style generation
	def __init__(self, channel_dim: int):
		super().__init__()
		self.channel_dim=channel_dim
		self.sinewavemixer = SineWave_Mixer(1, self.channel_dim)

	def forward(self, motion_style, tau: torch.Tensor) -> torch.Tensor:
		motion_code = self.sinewavemixer(motion_style, tau)
		return motion_code


class MoveNet(nn.Module):
	def __init__(self, style_dim, embed_dim, num_keys, num_class, n_mlp=4, lr_mlp=0.001):
		super(MoveNet, self).__init__()
		self.style_dim = style_dim
		self.embed_dim = embed_dim
		self.num_keys = num_keys
		self.num_class = num_class
		
		layers = []
		Fs = []
		for i in range(n_mlp):
			Fs.append(
				EqualLinear(
					self.style_dim, self.style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
				)
			)

		self.temporalStyle = TemporalCodeGen(self.style_dim)

		layers.append(EqualLinear(self.style_dim, 128))
		layers.append(EqualLinear(128, 64, activation="fused_lrelu"))
		layers.append(EqualLinear(64, 32))
		layers.append(EqualLinear(32, num_keys*3))
		self.layers = nn.Sequential(*layers)
		self.Fs = nn.Sequential(*Fs)
		self.class_embed = nn.Sequential(nn.Embedding(self.num_class, self.embed_dim),
			EqualLinear(self.embed_dim, self.style_dim))

	def forward(self, noise, class_label, tau,transition_factor):
		action_class = self.class_embed(class_label)
		temporalCode = self.Fs(noise) + transition_factor*action_class
		temporalStyle = self.temporalStyle(temporalCode, tau)
		# style = torch.hstack([action_class, temporalStyle])
	
	
		out = self.layers(temporalStyle)
		return out

class MoveDiscriminate(nn.Module):
	def __init__(self, style_dim, embed_dim, num_keys, num_class):
		super(MoveDiscriminate, self).__init__()
		self.style_dim = style_dim
		self.embed_dim = embed_dim
		self.num_keys = num_keys
		self.num_class = num_class
		
		layers = []
		self.linear_primary = nn.Sequential(EqualLinear(num_keys*3, 128),EqualLinear(128, 64, activation="fused_lrelu"))
		self.linear_secondary_time = nn.Sequential(	
			EqualLinear(64*3, 64, activation="fused_lrelu"),
			EqualLinear(64, self.style_dim*3),
		)

		self.linear_secondary_action = nn.Sequential(	
			EqualLinear(64*3, 64, activation="fused_lrelu"),
			EqualLinear(64, self.embed_dim),
		)
	
		self.tauEmbedding = SineActivation(1, self.style_dim)
		self.class_embed = nn.Sequential(nn.Embedding(self.num_class, self.embed_dim),
							EqualLinear(self.embed_dim, self.embed_dim))


	def forward(self, pose, action_class, timepoints, offset, transition_factor):

		batch, xyz_k = pose.shape

		action_rep = self.class_embed(action_class)

		pose_rep = self.linear_primary(pose)

	
		batch_offset = batch//3 #because we are using three frames
		
		tc1 = self.tauEmbedding(timepoints)
		tc2 = self.tauEmbedding(timepoints+offset)
		tc3 = self.tauEmbedding(timepoints+2*offset)

		tc1 = tc1.view(batch_offset, -1)
		tc2 = tc2.view(batch_offset, -1)
		tc3 = tc3.view(batch_offset, -1)

		tc_joined_segments = torch.cat([tc1, tc2, tc3], 1)

		pose1 = pose_rep[:batch_offset, :]
		pose2 = pose_rep[batch_offset:2*batch_offset, :]
		pose3 = pose_rep[2*batch_offset:3*batch_offset, :]

		pose1 = pose1.view(batch_offset, -1)
		pose2 = pose2.view(batch_offset, -1)
		pose3 = pose3.view(batch_offset, -1)

		pose_segments = torch.cat([pose1, pose2, pose3], 1)
		pose_time = self.linear_secondary_time(pose_segments)
		pose_action = self.linear_secondary_action(pose_segments)

		
		x = (pose_time * tc_joined_segments).sum(dim=1, keepdim=True) * (1 / np.sqrt(3*self.style_dim))
		# print(action_rep.shape, pose_action.shape)
		y = (action_rep * pose_action).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.embed_dim))

		return x + transition_factor*y



