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

filepath = 'mymodel_point_cheese_20'
justaplot = True # if you only want a plot

# use integral
point = 'point' in filepath
if point:
    meastype = 'point'
    integral = False
elif 'int' in filepath:
    integral = True
else:
    raise ValueError('Incorrect file path!')

if justaplot:
    vmin = -0.2
    vmax = 2
    gprh.makeplot2D_new(filepath,vmin=vmin,vmax=vmax,data=True)#,cmap=cm.Greys_r)
else:
    saveFreq = 200 # how often do you wanna save the model?
    training_iterations = 5000

    if integral:
        (model, dataname, train_y, n, x0, unitvecs, Rlim, X, Y, Z, rec_fbp, err_fbp,
                    ntx, nty, test_x, dom_points, m, dim, mt,
                    test_f, cov_f, noise_std, lossfu, buildPhi, opti_state, it_number) = \
        torch.load(filepath)

    if point:
        (model, dataname, train_y, n, train_x, X, Y, Z,
                        ntx, nty, test_x, dom_points, m, dim, mt,
                        test_f, cov_f, noise_std, lossfu, buildPhi, opti_state, it_number) = \
                torch.load(filepath)

    # optimiser
    optimiser = FullBatchLBFGS(model.parameters()) # make sure it's the same
    optimiser.__setstate__(opti_state)

    if integral:
        closure = gprh.gp_closure(model, meastype, buildPhi, lossfu, n, dom_points, train_y, regweight=regweight)
    else:
        closure = gprh.gp_closure(model, meastype, buildPhi, lossfu, n, dom_points, train_y, train_x=train_x, regweight=regweight)

    loss = closure()

    for i in range(training_iterations):

        options = {'line_search': True, 'closure': closure, 'max_ls': 3, 'ls_debug': False, 'inplace': False, 'interpolate': False,
                   'eta': 3, 'c1': 1e-4, 'decrease_lr_on_max_ls': 0.1, 'increase_lr_on_min_ls': 5}

        optimiser.zero_grad() # zero gradients
        loss.backward()  # Backprop derivatives
        loss, lr, ls_step = optimiser.step(options=options) # compute new loss

        # print
        gprh.optiprint(i, training_iterations, loss.item(), lr, ls_step, model, buildPhi.L)

        if (i+1)%saveFreq==0:
            if integral:
                gprh.compute_and_save(model, meastype, dataname, train_y, n, X, Y, Z,
                        ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu, buildPhi, optimiser, i+1,
                         joint=True, x0=x0, unitvecs=unitvecs, Rlim=Rlim, rec_fbp=rec_fbp, err_fbp=err_fbp)
            if point:
                gprh.compute_and_save(model, meastype, dataname, train_y, n, X, Y, Z,
                    ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu, buildPhi, optimiser, i+1,
                     joint=True, train_x=train_x)

    if integral:
        gprh.compute_and_save(model, meastype, dataname, train_y, n, X, Y, Z,
                ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu, buildPhi, optimiser, training_iterations,
                 joint=True, x0=x0, unitvecs=unitvecs, Rlim=Rlim, rec_fbp=rec_fbp, err_fbp=err_fbp)
    if point:
        gprh.compute_and_save(model, meastype, dataname, train_y, n, X, Y, Z,
            ntx, nty, test_x, dom_points, m, dim, mt, noise_std, lossfu, buildPhi, optimiser, training_iterations,
             joint=True, train_x=train_x)

    try:  # since plotting might produce an error on remote machines
        vmin = 0
        vmax = 1
        gprh.makeplot2D_new('mymodel_'+meastype+'_'+dataname+'_'+str(it_number+training_iterations),vmin=vmin,vmax=vmax,data=True)
    except:
        pass
