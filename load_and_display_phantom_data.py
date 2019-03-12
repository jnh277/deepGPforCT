import scipy.io as sio
from matplotlib import pyplot as plt
import sys
import numpy as np
import math

def getdata(dataname='circle_square',makeplot=False):

    if sys.platform=='linux':
        # data = sio.loadmat('/home/carl/Dropbox/deepGPforCT/matlab/phantom_data.mat')
        # data = sio.loadmat('/home/carl//Dropbox/deepGPforCT/matlab/circle_data.mat')
        # data=sio.loadmat('/home/carl/Dropbox/deepGPforCT/matlab/circle_square_data.mat')
        data=sio.loadmat('/home/carl/Dropbox/deepGPforCT/matlab/'+dataname+'_data.mat')
    else:
        # data = sio.loadmat('/Users/johannes/Dropbox/deepGPforCT/matlab/phantom_data.mat')
        # data = sio.loadmat('/Users/johannes/Dropbox/deepGPforCT/matlab/circle_data.mat')
        # data = sio.loadmat('/Users/johannes/Dropbox/deepGPforCT/matlab/circle_square_data.mat')
        data=sio.loadmat('/Users/johannes/Dropbox/deepGPforCT/matlab/'+dataname+'_data.mat')

    im = data['im']                     # original image
    X = data['X']                       # X values of the original image
    Y = data['Y']                       # Y values of the originhal image

    theta = data['theta']               # projection angles
    R = data['R']                       # measurement information, each column corresponds to a projection
    Xp = data['Xp']                     # the radial position that each pixel of the projection passes through

    fbp_result = data['fbp_result']     # result from applying filtered back projections (noise free)

    Rlim = 1.0*data['Rlim'][0][0] # radius

    n = np.size(R) # number of measurements

    # build input data
    r_gv = np.linspace(-Rlim,Rlim,np.size(R,0))
    theta_gv = theta[0]*math.pi/180
    [thetas, r] = np.meshgrid(theta_gv,r_gv)  # angles
    thetas = thetas.reshape(n,1)
    r = r.reshape(n,1)

    x0 = np.concatenate((r*np.cos(thetas), r*np.sin(thetas)),axis=1) # center points
    unitvecs = np.concatenate((-np.sin(thetas), np.cos(thetas)),axis=1) # normal unit vector

    startPoints = x0 - Rlim*unitvecs
    endPoints   = x0 + Rlim*unitvecs

    # measurements
    train_y = R.flatten()

    if makeplot:
        # plot the original image
        fplot, ax = plt.subplots(2, 2, figsize=(27,15))
        h1 = ax[0,0].pcolor(X,Y,im)
        h1.set_clim(0,1)
        # plt.title('Original image')

        # plot the measurement data
        h2 = ax[0,1].pcolor(theta, Xp, R)
        h2.set_clim(0,1)

        # plot the reconstruction resulst from using iradon in matlab
        h3 = ax[1,0].pcolor(X,Y,fbp_result)
        h3.set_clim(0,1)

        # plot the measurement lines
        lines_x = np.concatenate((startPoints[:,0].reshape(1,n),endPoints[:,0].reshape(1,n)),axis=0)
        lines_y = np.concatenate((startPoints[:,1].reshape(1,n),endPoints[:,1].reshape(1,n)),axis=0)
        h4 = ax[1,1].plot(lines_x, lines_y)

        plt.show()

    return train_y, n, x0, unitvecs, Rlim, X, Y, im
