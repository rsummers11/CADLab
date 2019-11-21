Please download and put [our data](https://nihcc.box.com/s/r8kf5xcthjvvvf6r7l1an99e1nj4080m), [JSRT](http://db.jsrt.or.jp/eng.php) and [Montgomery](http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip) in this directory. For JSRT and Montgomery, you need to merge the left and right lung masks as one image. For JSRT, you need to add '_0' at the end of the filename if it is normal, otherwise, add '_1'. The lung masks for JSRT can be downloaded from this link: https://www.isi.uu.nl/Research/Databases/SCR/download.php

Since the images are 12 bit in JSRT dataset, we use the following code to convert them into 8 bit:

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
