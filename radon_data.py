import numpy as np
import math

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, iradon, resize

def getdata(dataname='circle_square',nproj=180,nmeas_proj=100,image_res=3000, D=1, nt=None, reconstruct_fbp=False, sigma_n=0):
    """
    dataname: what data do you want?
    nproj: how many projections? chosen in 0<=theta<180
    nmeas_proj: number of measurements per projection
    image_res: resolution of the image used to compute the radon transform (the higher, the more accurate, but slower)
    D:  we normalise the data to the domain 0 < x,y < D
    nt: size of test grid, default=nmeas_proj
    reconstruct: return fbp reconstruction?
    """

    R = 0.5*D # radius

    nextract = int( np.ceil( image_res / (nmeas_proj-1) ) ) # used to extract desired data
    nr = (nmeas_proj-1)* nextract + 1 # size of image that will be used for radon

    if nt is not None:
        xt = np.linspace(0, D, nt)
        Xt,Yt = np.meshgrid(xt, xt)
    else:
        xt = np.linspace(0, D, nmeas_proj)
        Xt,Yt = np.meshgrid(xt, xt)

    if dataname=='phantom':
        image_r = imread(data_dir + "/phantom.png", as_gray=True)
        image = resize(image_r, (nr,nr), mode='reflect')
        if nt is not None:
            image_t = resize(image_r, (nt,nt), mode='reflect')
        else:
            image_t = image[0::nextract,0::nextract]
    elif dataname=='circle_square':
        x = np.linspace(0, D, nr)
        X,Y = np.meshgrid(x, x)
        Rad = np.sqrt(np.power(X-R,2)+np.power(Y-R,2))
        image = np.zeros((nr,nr))
        image[Rad<0.7*R]=1
        image[(X<0.6*D)*(X>0.4*D)*(Y<0.6*D)*(Y>0.4*D)]=0
        if nt is not None:
            Rad = np.sqrt(np.power(Xt-R,2)+np.power(Yt-R,2))
            image_t = np.zeros((nt,nt))
            image_t[Rad<0.7*R]=1
            image_t[(Xt<0.6*D)*(Xt>0.4*D)*(Yt<0.6*D)*(Yt>0.4*D)]=0
        else:
            image_t = image[0::nextract,0::nextract]

    # get measurements
    theta = np.linspace(0., 180., nproj, endpoint=False) # angles
    sinogram = radon(image, theta=theta, circle=True) / (nr/D) # normalise wrt desired domain

    # extract data
    train_y = sinogram[0::nextract,:] + sigma_n * np.random.randn(nmeas_proj,nproj)

    # number of measurements
    n = np.size(train_y)

    # build input data
    r_grid = np.linspace(-R,R,nmeas_proj)
    theta_grid = theta*math.pi/180
    [thetas, r] = np.meshgrid(theta_grid,r_grid)  # angles
    thetas = thetas.transpose().reshape(n,1)
    r = r.transpose().reshape(n,1)

    x0 = np.concatenate((r*np.cos(thetas), r*np.sin(thetas)),axis=1) + R # center points
    unitvecs = np.concatenate((-np.sin(thetas), np.cos(thetas)),axis=1) # normal unit vectors

    if reconstruct_fbp:
        reconstruction_fbp = iradon(train_y, theta=theta, circle=True) * nmeas_proj # scale with number of meas/proj
        # compute error
        error_fbp = reconstruction_fbp - image[0::nextract,0::nextract]
        error_fbp = np.sqrt(np.mean(error_fbp**2))
        if nt is not None:
            reconstruction_fbp = resize(reconstruction_fbp, (nt,nt), mode='reflect')
        return train_y.flatten('F'), n, x0, unitvecs, R, Xt, Yt, image_t, reconstruction_fbp, error_fbp
    else:
        return train_y.flatten('F'), n, x0, unitvecs, R, Xt, Yt, image_t

