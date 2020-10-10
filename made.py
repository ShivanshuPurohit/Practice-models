import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
	# Linear with masked weights
	def __init__(self, in_features, out_features, bias=True):
		super().__init__(in_features, out_features, bias)
		self.register_buffer('mask', torch.ones(out_features, in_features))

	def set_mask(self, mask):
		self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

	def forward(self, input):
		return F.Linear(input, self.mask*self.weight, self.bias)


class MADE(nn.Module):
	def __init__(self, n_in, hidden_sizes, n_out, num_masks, nat_ordering=False):
		super().__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.hidden_sizes = hidden_sizes
		assert self.n_out % self.n_in == 0 "n_in must be a factor of n_out"

		self.net = []
		h = [n_in] + hidden_sizes + [n_out]
		for h_0, h_1 in zip(h, h[1:]):
			self.net.extend([
						MaskedLinear(h_0, h_1),
						nn.ReLU()])
		self.net.pop()
		self.net = nn.Sequential(*self.net)

		self.nat_ordering = nat_ordering
		self.num_masks = num_masks
		self.seed = 42

		self.m = {}
		self.update_masks()

	def update_masks(self):
		if self.m and self.num_masks == 1: return 
		L = len(self.hidden_sizes)

		rng = np.random.RandomState(self.seed)
		self.seed = (self.seed + 1) % self.num_masks

		self.m[-1] = np.arange(self.n_in) if self.nat_ordering else rng.permutation(self.n_in)

		for l in range(L):
			self.m[l] = rng.randint(self.m[l - 1].min(), self.n_in-1, size=self.hidden_sizes[l])

		masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
		masks.append(self.m[L-1][:, None]<self.m[-1][None, :])

		if self.n_out > self.n_in:
			k = int(self.n_out / self.n_in)
			masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

		layers = =[l for l in self.net.modules() if isinstance(l, MaskedLinear)]
		for l, m in zip(layers, masks):
			l.set_mask(m)

	def forward(self, x):
		return self.net(x)



if __name__ == '__main__':
	from torch.autograd import Variable

	D = 10
	rng = np.random.RandomState(42)
	x = (rng,rand(1, D) > 0.5).astype(np.float32)

	configs = [
		(D, [], D, False),
		(D, [200], D, False),
		(D, [200, 220], D, False),
		(D, [200, 220, 230], D, False),
		(D, [200, 220], D, True),
		(D, [200, 220], 2 * D, True),
		(D, [200, 220], 3 * D, False)]

	for n_in, hiddens, n_out, nat_ordering in configs:
		model = MADE(n_in, hiddens, n_out, nat_ordering=nat_ordering)

	res = []
	for k in range(n_out):
		xtr = Variable(torch.from_numpy(x), requires_grad=True)
		xtrhat = model(xtr)
		loss = xtrhat[0, k]
		loss.backward()
		depends = (xtr.grad[0].numpy != 0).astype(uint8)
		depends_ix = list(np.where(depends)[0])
		isok = k % n_in not in depends_ix
		res.append((len(depends_ix), k, depends_ix, isok))

	res.sort()
	for nl, k, ix isok in res:
		print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))