import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from made import MADE


def run(split, upto=None):
	torch.set_grad_enabled(split=='train')
	model.train() if split == 'train' else  model.eval()
	nsamples = 1 if split == 'train' else xte
	N, D = x.size()
	B = 128
	n_steps = N // B if upto is None else min(N//B, upto)
	losses = []
	for step in range(n_steps):
		xb = Variable(x[step * B: step * B + B])
		xbhat = torch.zeros_like(xb)
		for s in range(nsamples):
			if step % args.resample_every == 0 or split == 'test':
			model.update_masks()
			xbhat += model(xb)
		xbhat /= nsamples

		loss = F.binary_cross_entropy_with_logits(xbhat, xb, size_average=False) / B
		lossf = loss.data.item()
		losses.append(lossf)

		if split == 'train':
			opt.zero_grad()
			loss.backward()
			opt.step()

	print("%s epoch avg loss: %f" %(split, np.mean(losses)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', required=True, type=str, help="Path to binarized_mnist.npz")
    parser.add_argument('-q', '--hiddens', type=str, default='500', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument('-n', '--num-masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
    parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
    args = parser.parse_args()

    np.random_seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print("loading binarized mnist from", args.data_path)
    mnist = np.load(args.data_path)
    xtr, xte = mnist['train_data'], mnist['valid_data']
    xtr = torch.from_numpy(xtr).cuda()
    xte = torch.from_numpy(xte).cuda()

    # construct model and ship to GPU
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(xtr.size(1), hidden_list, xtr.size(1), num_masks=args.num_masks)
    print("number of model parameters:",sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    # set up the optimizer
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)
    
    # start the training
    for epoch in range(100):
        print("epoch %d" % (epoch, ))
        scheduler.step(epoch)
        run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
        run_epoch('train')
    
    print("optimization done. full test set eval:")
    run_epoch('test')