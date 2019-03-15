import numpy as np
import matplotlib.pyplot as plt
import math

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, resize

D = 1 # we normalise the data to the domain 0 < x,y < D
R = 0.5*D # radius

nr = 200

expand = int( np.ceil( nr* 1/math.sqrt(2)*(1-1/math.sqrt(2))  ) )
image = np.zeros((nr+2*expand,nr+2*expand))
image[expand:nr+expand,expand:nr+expand] = np.ones((nr,nr))


theta = np.linspace(0., 90., 3)
sinogram = radon(image, theta=theta, circle=True) *D/nr # normalise wrt desired domain

# check
print(sinogram[:,0].max()) # should expect 1
print(sinogram[:,1].max()) # should expect D*sqrt(2)
print(D*math.sqrt(2))


n = np.size(sinogram) # number of measurements

# build input data
r_gv = np.linspace(-R,R,np.size(sinogram,0))
theta_gv = theta*math.pi/180
[thetas, r] = np.meshgrid(theta_gv,r_gv)  # angles
thetas = thetas.transpose().reshape(n,1)
r = r.transpose().reshape(n,1)

x0 = np.concatenate((r*np.cos(thetas), r*np.sin(thetas)),axis=1) + R # center points
unitvecs = np.concatenate((-np.sin(thetas), np.cos(thetas)),axis=1) # normal unit vectors
