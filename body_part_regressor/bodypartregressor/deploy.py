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

import sys
import os
import _init_paths
import caffe
import cv2
import numpy as np
from scipy.io import savemat
from load_img import load_img
from config import cfg, cfg_from_file


default_model = 'snapshots/_iter_10000.caffemodel'  # your trained model
default_prototxt = "test.prototxt"
image_set_file = 'test_image_list_example.txt'  # your list of image folders or filenames
data_dir = 'test_data'  # if images are in one folder
default_cfg = "config.yml"

GPU_ID = 0
IMG_SUFFIX = '.png'

#rtdir = os.path.join(os.path.dirname(__file__), os.pardir)
#os.chdir(rtdir)  # go to root dir of this project


def get_image_index(image_set_file):
    with open(image_set_file) as f:
        volume_index = [x.strip() for x in f.readlines()]
    img_list = []
    for v in volume_index:
        fd_path = os.path.join(data_dir, v)
        if os.path.isfile(fd_path) and fd_path.endswith(IMG_SUFFIX):
            img_list += [v]
        else:
            img_list1 = [f for f in os.listdir(fd_path) if f.endswith(IMG_SUFFIX)]
            img_list1.sort(key=lambda x: int(x[:-4]))
            img_list1 = [os.path.join(v,f) for f in img_list1]
            img_list += img_list1
    return img_list


if __name__ == '__main__':
    cfg_from_file(default_cfg)
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    net = caffe.Net(default_prototxt, default_model, caffe.TEST)

    print('Processing img ...')
    image_index = get_image_index(image_set_file)
    im_num = len(image_index)
    vals = np.empty((im_num,))
    for j in range(im_num):
        fn = os.path.join(data_dir, image_index[j])
        print(fn, '\t', end=' ')
        im = load_img(fn)
        im -= cfg.PIXEL_MEANS

        data = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for chn in range(3):
            data[0, chn, :, :] = im

        blobs_out = net.forward(data=data)
        vals[j] = blobs_out['reg_value']
        print(vals[j])

    fn = 'slice_scores.txt'
    with open(fn,'w') as f:
        f.writelines([a + '\t' + str(b) + '\r\n' for a,b in zip(image_index, vals)])
    print('Written to', fn)
