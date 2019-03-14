"""
===============
Radon transform
===============

In computed tomography, the tomography reconstruction problem is to obtain
a tomographic slice image from a set of projections [1]_. A projection is
formed by drawing a set of parallel rays through the 2D object of interest,
assigning the integral of the object's contrast along each ray to a single
pixel in the projection. A single projection of a 2D object is one dimensional.
To enable computed tomography reconstruction of the object, several projections
must be acquired, each of them corresponding to a different angle between the
rays with respect to the object. A collection of projections at several angles
is called a sinogram, which is a linear transform of the original image.

The inverse Radon transform is used in computed tomography to reconstruct
a 2D image from the measured projections (the sinogram). A practical, exact
implementation of the inverse Radon transform does not exist, but there are
several good approximate algorithms available.

As the inverse Radon transform reconstructs the object from a set of
projections, the (forward) Radon transform can be used to simulate a
tomography experiment.

This script performs the Radon transform to simulate a tomography experiment
and reconstructs the input image based on the resulting sinogram formed by
the simulation. Two methods for performing the inverse Radon transform
and reconstructing the original image are compared: The Filtered Back
Projection (FBP) and the Simultaneous Algebraic Reconstruction
Technique (SART).

For further information on tomographic reconstruction, see

.. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic Imaging",
       IEEE Press 1988. http://www.slaney.org/pct/pct-toc.html

.. [2] Wikipedia, Radon transform,
       https://en.wikipedia.org/wiki/Radon_transform#Relationship_with_the_Fourier_transform

.. [3] S Kaczmarz, "Angenaeherte Aufloesung von Systemen linearer
       Gleichungen", Bulletin International de l'Academie Polonaise
       des Sciences et des Lettres, 35 pp 355--357 (1937)

.. [4] AH Andersen, AC Kak, "Simultaneous algebraic reconstruction
       technique (SART): a superior implementation of the ART algorithm",
       Ultrasonic Imaging 6 pp 81--94 (1984)

The forward transform
=====================

As our original image, we will use the Shepp-Logan phantom. When calculating
the Radon transform, we need to decide how many projection angles we wish
to use. As a rule of thumb, the number of projections should be about the
same as the number of pixels there are across the object (to see why this
is so, consider how many unknown pixel values must be determined in the
reconstruction process and compare this to the number of measurements
provided by the projections), and we follow that rule here. Below is the
original image and its Radon transform, often known as its *sinogram*:
"""


import numpy as np
import matplotlib.pyplot as plt
import math

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, resize

nr = 500

expand = int( np.ceil( nr* 1/math.sqrt(2)*(1-1/math.sqrt(2))  ) )
image = np.zeros((nr+2*expand,nr+2*expand))
image[expand:nr+expand,expand:nr+expand] = np.ones((nr,nr))

theta = np.linspace(0., 45., 2)
sinogram = radon(image, theta=theta, circle=True)

# we should normalise by nr
sinogram = sinogram / nr

# check
print(sinogram[:,0].max()) # should expect 1
print(sinogram[:,1].max()) # should expect sqrt(2)
