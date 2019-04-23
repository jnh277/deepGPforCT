import numpy as np
import matplotlib.pyplot as plt
import radon_data as rd


nmeas_proj = 140
nproj = 15
nt = 150

# noise level
noise_std = 0.0000

# import data
dataname='cheese'
print('Getting data...')
train_y, n, x0, unitvecs, Rlim, X, Y, image, rec_fbp, err_fbp = rd.getdata(dataname=dataname,nmeas_proj=nmeas_proj,
                                                                           nproj=nproj,nt=nt,sigma_n=noise_std,reconstruct_fbp=True)
print('Data ready!')

train_y = train_y.reshape(nproj,nmeas_proj).transpose()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20))

ax1.set_title("Original")
pc = ax1.imshow(np.flipud(image), cmap=plt.cm.Greys_r, extent=(X.min(), X.max(), Y.min(), Y.max()))
pc.set_clim(image.min(),image.max())

ax2.set_title("Reconstruction\nFiltered back projection")
pc = ax2.imshow(rec_fbp, cmap=plt.cm.Greys_r, extent=(X.min(), X.max(), Y.min(), Y.max()))
pc.set_clim(image.min(),image.max())

fig.tight_layout()
plt.show()


print('\nFBP rms reconstruction error: %.10f' % err_fbp)

