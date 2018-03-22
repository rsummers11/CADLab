"""
Ke Yan
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
March 2018

THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import cfg


def windowing(im, win):
    im1 = im.copy()
    im1 -= win[0] + 32768
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1


def load_img(fn, spacing=None):
    im = cv2.imread(fn, -1)
    assert im is not None, 'Cannot find %s' % fn
    im = im.astype(np.float32, copy=False)

    # plt.figure()
    # plt.imshow(im, vmin=0, vmax=255, cmap='gray')
    # plt.show()

    im_scale = float(cfg.SCALE) / float(np.min(im.shape[0:2]))

    if spacing is not None and cfg.NORM_SPACING > 0:  # spacing adjust, will overwrite simple scaling
        im_scale = float(spacing)/cfg.NORM_SPACING
        
    if im_scale != 1:
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    if np.round(np.max(im.shape[0:2]) > cfg.MAX_SIZE):
        im_scale = float(cfg.MAX_SIZE) / float(np.max(im.shape[0:2]))
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    if cfg.IMG_IS_16bit:
        im = windowing(im, [-175, 275])

    return im


def im_list_to_blob(ims, use_max_size=True):
    """Convert a list of images into a network input.
    """
    # max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # min_shape = np.array([im.shape for im in ims]).min(axis=0)
    # print max_shape, min_shape
    if use_max_size:
        max_shape = np.array([cfg.MAX_SIZE, cfg.MAX_SIZE])
    else:
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        
    num_images = len(ims)
    blob = np.zeros((num_images, 3, max_shape[0], max_shape[1]),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        m = (max_shape-im.shape)/2
        for chn in range(3):
            blob[i, chn, m[0]:m[0]+im.shape[0], m[1]:m[1]+im.shape[1]] = im

    return blob
