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

filepath = 'mymodel_point_phantom_2000'
justaplot = True  # if you only want a plot

# use integral
point = 'point' in filepath
if point:
	meastype = 'point'
	integral = False
elif 'int' in filepath:
	meastype = 'int'
	integral = True
else:
	raise ValueError('Incorrect file path!')

if justaplot:
	vmin = 0
	vmax = 	1
	gprh.makeplot2D_new(filepath,vmin=vmin,vmax=vmax)#,cmap=cm.Greys_r)
else:
	if integral:
		(model, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
					ntx, nty, test_x, dom_points, m, diml, mt,
					test_f, cov_f, noise_std, lossfu, buildPhi, opti_state, it_number) = \
		torch.load(filepath)

	if point:
		(model, dataname, train_y, n, train_x, X, Y, Z,
						ntx, nty, test_x, dom_points, m, diml, mt,
						test_f, cov_f, noise_std, lossfu, buildPhi, opti_state, it_number) = \
				torch.load(filepath)

	# function that computes the solution and save stuff (declared here to enable usage within the optimisation loop)
	def compute_and_save(it_number):
		# update phi
		if integral:
			phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points)
		else:
			phi,sq_lambda,L = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)

		# now make predictions
		test_f, cov_f = model(y_train=train_y, phi=phi, sq_lambda=sq_lambda, L=L, x_test=test_x)

		# RMS error
		ground_truth = torch.from_numpy(Z).float().view(np.size(Z))
		error = torch.mean( (ground_truth - test_f.squeeze()).pow(2) ).sqrt()
		print('RMS error: %.10f' %(error.item()))
		if integral:
			print('RMS error fbp: %.10f' %(err_fbp))

		# save variables
		if integral:
			torch.save((model, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
						ntx, nty, test_x, dom_points, m, diml, mt,
						test_f, cov_f, noise_std, lossfu, buildPhi, optimiser.__getstate__(), it_number),
					   'mymodel_'+meastype+'_'+dataname+'_'+str(it_number))
		if point:
			torch.save((model, dataname, train_y, n, train_x, X, Y, Z,
						ntx, nty, test_x, dom_points, m, diml, mt,
						test_f, cov_f, noise_std, lossfu, buildPhi, optimiser.__getstate__(), it_number),
					   'mymodel_'+meastype+'_'+dataname+'_'+str(it_number))

	# optimiser
	optimiser = FullBatchLBFGS(model.parameters()) # make sure it's the same
	optimiser.__setstate__(opti_state)

	# compute initial loss
	def closure():
		optimiser.zero_grad()
		global L
		if integral:
			phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,dom_points)
		else:
			phi, sq_lambda, L = buildPhi.getphi(model,m,n,mt,dom_points,train_x=train_x)
		return lossfu(model.gp.log_sigma_f, model.gp.log_lengthscale, model.gp.log_sigma_n, phi, train_y, sq_lambda)
	loss = closure()

	saveFreq = 500 # how often do you wanna save the model?

	training_iterations = 3000
	for i in range(it_number, it_number+training_iterations):
		# build phi
		options = {'closure': closure, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
				   'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

		optimiser.zero_grad()  # zero gradients
		loss.backward()  # Backprop derivatives
		loss, lr, ls_step = optimiser.step(options=options) # compute new loss

		# print
		gprh.optiprint(i,training_iterations+it_number,loss.item(),lr,ls_step,model,L)

		if (i+1-it_number)%saveFreq==0:
			compute_and_save(i+1)


	compute_and_save(it_number+training_iterations)

	try:  # since plotting might produce an error on remote machines
		vmin = 0
		vmax = 1
		gprh.makeplot2D_new('mymodel_'+meastype+'_'+dataname+'_'+str(it_number+training_iterations),vmin=vmin,vmax=vmax)
	except:
		pass

