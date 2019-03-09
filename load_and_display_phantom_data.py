import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib import cm
# data = sio.loadmat('../matlab/phantom_data.mat')
data = sio.loadmat('/Users/johannes/Dropbox/deepGPforCT/matlab/phantom_data.mat')

im = data['im']                     # original image
X = data['X']                       # X values of the original image
Y = data['Y']                       # Y values of the originhal image

theta = data['theta']               # projection angles
R = data['R']                       # measurement information, each column corresponds to a projection
Xp = data['Xp']                     # the radial position that each pixel of the projection passes through


fbp_result = data['fbp_result']     # result from applying filtered back projections

# plot the original image
fplot, ax = plt.subplots(2, 2, figsize=(27,15))
h1 = ax[0,0].pcolor(X,Y,im)
# plt.title('Original image')

# plot the measurement data
h2 = ax[0,1].pcolor(theta, Xp, R)

h3 = ax[1,0].pcolor(X,Y,fbp_result)


plt.show()




