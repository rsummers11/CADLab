import os
import numpy as np
from skimage import io, exposure
def make_lungs():
    path = '/path/to/JSRT/All247images/'
    for i, filename in enumerate(os.listdir(path)):
        img = 1.0 - np.fromfile(path + filename, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        img = exposure.equalize_hist(img)
        io.imsave('/path/to/JSRT/new/' + filename[:-4] + '.png', img)
        print 'Lung', i, filename

make_lungs()
