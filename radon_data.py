import numpy as np
import math

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, iradon, resize

def getdata(dataname='circle_square',nproj=180,nmeas_proj=100,image_res=3000, R=1, nt=None, reconstruct_fbp=False, sigma_n=0, points=False, npmeas=1000):
    """
    dataname: what data do you want?
    nproj: how many projections? chosen in 0<=theta<180
    nmeas_proj: number of measurements per projection
    image_res: resolution of the image used to compute the radon transform (the higher, the more accurate, but slower)
    D:  we normalise the data to the domain 0 < x,y < D
    nt: size of test grid, default=nmeas_proj
    reconstruct: return fbp reconstruction?
    """
    np.random.seed(seed=0)

    # R = 0.5*D # radius

    nextract = int( np.ceil( image_res / (nmeas_proj-1) ) ) # used to extract desired data
    nr = (nmeas_proj-1)* nextract + 1 # size of image that will be used for radon

    x = np.linspace(-R, R, nr)
    X,Y = np.meshgrid(x, x)

    if nt is not None:
        xt = np.linspace(-R, R, nt)
        Xt,Yt = np.meshgrid(xt, xt)
    else:
        xt = np.linspace(-R, R, nmeas_proj)
        Xt,Yt = np.meshgrid(xt, xt)

    if dataname=='phantom':
        image_r = imread(data_dir + "/phantom.png", as_gray=True)
        image = resize(image_r, (nr,nr), mode='reflect', anti_aliasing=True)
        if nt is not None:
            image_t = resize(image_r, (nt,nt), mode='reflect', anti_aliasing=True)
        else:
            image_t = image[0::nextract,0::nextract]
    elif dataname=='circle_square':
        Rad = np.sqrt(np.power(X,2)+np.power(Y,2))
        image = np.zeros((nr,nr))
        image[Rad<0.7*R]=1
        image[(X<0.2*R)*(X>-0.2*R)*(Y<0.2*R)*(Y>-0.2*R)]=0
        if nt is not None:
            Rad = np.sqrt(np.power(Xt,2)+np.power(Yt,2))
            image_t = np.zeros((nt,nt))
            image_t[Rad<0.7*R]=1
            image_t[(Xt<0.2*R)*(Xt>-0.2*R)*(Yt<0.2*R)*(Yt>-0.2*R)]=0
        else:
            image_t = image[0::nextract,0::nextract]

        image = np.flipud(image)
        image_t = np.flipud(image_t)

    if not(points):
        # get measurements
        theta = np.linspace(0., 180., nproj, endpoint=False) # angles
        sinogram = radon(image, theta=theta, circle=True) / (nr/R/2.0) # normalise wrt desired domain

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

        x0 = np.concatenate((r*np.cos(thetas), r*np.sin(thetas)),axis=1) # center points
        unitvecs = np.concatenate((-np.sin(thetas), np.cos(thetas)),axis=1) # normal unit vectors

        if reconstruct_fbp:
            reconstruction_fbp = iradon(train_y, theta=theta, circle=True) * nmeas_proj # scale with number of meas/proj
            # compute error
            error_fbp = reconstruction_fbp - image[0::nextract,0::nextract]
            error_fbp = np.sqrt(np.mean(error_fbp**2))
            if nt is not None:
                reconstruction_fbp = resize(reconstruction_fbp, (nt,nt), mode='reflect', anti_aliasing=True)
            return train_y.flatten('F'), n, x0, unitvecs, R, Xt, Yt, np.flipud(image_t).copy(), reconstruction_fbp, error_fbp
        else:
            return train_y.flatten('F'), n, x0, unitvecs, R, Xt, Yt, np.flipud(image_t).copy()
    else:
        indices = (np.prod(X.shape)*np.random.rand(npmeas)).astype(int)
        train_y = np.flipud(image).flatten()[indices] + sigma_n * np.random.randn(npmeas)
        train_x = np.concatenate( (X.flatten()[indices].reshape(train_y.size,1), Y.flatten()[indices].reshape(train_y.size,1) ),axis=1)

        return train_y, train_x, Xt, Yt, np.flipud(image_t).copy()
