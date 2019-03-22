import torch
import gp_regression_hilbert as gprh
import numpy as np
import gpnets
from matplotlib import cm

# add path to optimisation code
import sys
sys.path.append('./opti_functions/')
from Adam_ls import Adam_ls
from LBFGS import FullBatchLBFGS

# use integral
integral = True

filepath = 'mymodel_phantom_1300'

justaplot = True

if justaplot:
	vmin = -0.5
	vmax = 	1.1
	gprh.makeplot2D_new(filepath,vmin=vmin,vmax=vmax,cmap=cm.Greys_r)
else:
	(model, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
				ntx, nty, test_x, dom_points, m, diml, mt,
				test_f, cov_f, noise_std, nLL, buildPhi, lr, it_number) = \
	torch.load(filepath)

	# function that computes the solution and save stuff (declared here to enable usage within the optimisation loop)
	def compute_and_save(it_number):
		# update phi
		phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points)

		# now make predictions
		test_f, cov_f = model(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

		ground_truth = torch.from_numpy(Z).float().view(np.size(Z))

		error = torch.mean( (ground_truth - test_f.squeeze()).pow(2) ).sqrt()
		print('RMS error: %.10f' %(error.item()))
		print('RMS error fbp: %.10f' %(err_fbp))

		# save variables
		torch.save((model, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
				ntx, nty, test_x, dom_points, m, diml, mt,
				test_f, cov_f, noise_std, nLL, buildPhi, optimiser.param_groups[0]['lr'], it_number),
			   'mymodel_'+dataname+'_'+str(it_number))

	# optimiser
	optimiser = Adam_ls(model.parameters(), lr=lr, weight_decay=0.0)  # Adam with line search

	# compute initial loss
	def closure():
		global L
		phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points)
		loss = nLL(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)
		return loss
	loss = closure()

	saveFreq = 50 # how often do you wanna save the model?

	training_iterations = 1000
	for i in range(it_number+1, it_number+training_iterations):
		# build phi

		options = {'line_search': True, 'closure': closure, 'max_ls': 10, 'ls_debug': False, 'inplace': False, 'interpolate': False,
			   'eta': 3, 'c1': 1e-6, 'decrease_lr_on_max_ls': 0.5, 'increase_lr_on_min_ls': 2}

		optimiser.zero_grad()  # zero gradients
		loss.backward()  # Backprop derivatives
		loss, lr, ls_step = optimiser.step(options=options) # compute new loss

		# print
		gprh.optiprint(i,training_iterations+it_number,loss.item(),lr,ls_step,model,L)

		if (i+1-it_number)%saveFreq==0:
			compute_and_save(i+1)


	compute_and_save(it_number+training_iterations)

	try: # since plotting might produce an error on remote machines
		vmin = -2
		vmax = 2
		gprh.makeplot2D_new('mymodel_'+dataname+'_'+str(it_number+training_iterations),vmin=vmin,vmax=vmax)
	except:
		pass

