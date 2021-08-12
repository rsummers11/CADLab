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

import numpy as np
import yaml
import cv2
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

import caffe

from config import cfg
from load_img import load_img, im_list_to_blob


DEBUG = False

class DataLayer(caffe.Layer):

    def _load_imdb(self, image_set_file):
        """"""
        self._data_path = cfg.DATA_DIR
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            volume_index = [x.strip() for x in f.readlines()]

        self.data_list = []
        self.fd_list = []
        for v in volume_index:
            fd_path = os.path.join(self._data_path, v)

            img_list = [ f for f in os.listdir(fd_path) if f.endswith(cfg.IMG_SUFFIX) and
                         os.path.getsize(os.path.join(fd_path,f)) > cfg.TRAIN.MIN_IM_SIZE_KB]
            if len(img_list) < self.slice_num:
                print('only',len(img_list), 'images in', v)
                continue
            img_list.sort(key=lambda x: int(x[:-4]))  # make sure the former image is above the latter one in the volume, assuming the indices are ordered
            self.data_list.append(img_list)
            self.fd_list.append(fd_path)

        self.volume_num = len(self.fd_list)
        assert self.volume_num >= cfg.TRAIN.GROUPS_PER_BATCH, 'too few training volumes'

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(self.volume_num))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.GROUPS_PER_BATCH >= self.volume_num:
            self._shuffle_roidb_inds()

        v_inds = self._perm[self._cur:self._cur + cfg.TRAIN.GROUPS_PER_BATCH]
        self._cur += cfg.TRAIN.GROUPS_PER_BATCH
        return v_inds

    def _get_next_minibatch(self):
        """Return the image indices for the next minibatch."""
        if cfg.TRAIN.USE_PREFETCH:
            assert self.error is None, self.error.message

            ims = self._data_queue.get()
            # print 'got from queue, now size', self._data_queue.qsize()
            return ims
        else:
            v_inds = self._get_next_minibatch_inds()
            return self._get_minibatch(v_inds)

    def _get_minibatch(self, volume_ids):
        ims = []
        for i in range(self.group_size):
            while True:
                try_again = False
                volume_id = volume_ids[i]
                nIm = len(self.data_list[volume_id])

                k = np.random.randint(np.floor(nIm / self.slice_num))+1
                s = 1/(1+np.exp((np.random.rand(1)-.5)*2*5))  # give more chance to top and bottom slices because
                    # the middle slices naturally have larger probability of being selected
                start = int(s*(nIm - k*(self.slice_num-1)))
                # start = np.random.choice(nIm - d*(self.slice_num-1)) # original
                img_ids = [start+j*k for j in range(self.slice_num)]

                # count the frequency of selection of each part of the volume
                r = np.floor(np.array(img_ids,dtype=np.float)/nIm*50)
                self._sel_hist[r.astype(np.int)] += 1

                ims1 = []
                for img_id in img_ids:
                    # print volume_id, img_id, len(self.fd_list), len(self.data_list[volume_id])
                    fn = os.path.join(self.fd_list[volume_id], self.data_list[volume_id][img_id])
                    if os.path.getsize(fn) < cfg.TRAIN.MIN_IM_SIZE_KB:  # very small size indicates the image contains very little contents
                        try_again = True
                        break

                    im = load_img(fn)
                    # print im.shape

                    if im is None:
                        try_again = True
                        break

                    if cfg.TRAIN.CROP_RANDOM_PATCH:
                        # randomly crop a 2D patch
                        H,W = im.shape
                        area = im.sum()
                        while True:
                            r = np.random.rand()*.5+.5
                            h = np.random.randint(H/2,H+1)
                            w = min(W,int(np.ceil(float(H*W)*r/h)))
                            y = np.random.randint(H-h+1)
                            x = np.random.randint(W-w+1)
                            # print mask[y:y+h, x:x+w].sum(), area
                            if im[y:y+h, x:x+w].sum() < area/3:
                                continue  # prevent selecting large black area
                            im = im[y:y+h, x:x+w]
                            break

                    im -= self.avg_img
                    ims1.append(im)

                if try_again:
                    print('~', fn)
                    continue
                else:
                    ims += ims1
                    break
            if DEBUG: print(img_ids, nIm)

        # print self._sel_hist, self._sel_hist/self._sel_hist.sum()
        return ims

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""
        self.layer_params = yaml.load(self.param_str)
        self.group_size = cfg.TRAIN.GROUPS_PER_BATCH
        self.slice_num = cfg.TRAIN.SLICE_NUM
        self.batch_size = self.group_size * self.slice_num
        # assert self.batch_size % 2 == 0, "self.batch_size must be even!"

        self._load_imdb(cfg.train_imdb)
        print('{:d} volume entries'.format(self.volume_num))
        self._shuffle_roidb_inds()
        self._training = self.phase == 0

        self.avg_img = cfg.PIXEL_MEANS
        self._iter = 0
        self.error = None

        self._sel_hist = np.zeros((50,))

        top[0].reshape(self.batch_size, 3, cfg.MAX_SIZE, cfg.MAX_SIZE)
        # top[1].reshape(self.batch_size, 1, cfg.MAX_SIZE, cfg.MAX_SIZE)

        if cfg.TRAIN.USE_PREFETCH:
            from multiprocessing import Process, Queue
            self._data_queue = Queue(10)

            def prefetch():
                while True:
                    try:
                        # print 'begin prefetch'
                        v_inds = self._get_next_minibatch_inds()
                        ims = self._get_minibatch(v_inds)
                        self._data_queue.put(ims)
                        # print 'prefetched', v_inds, self._data_queue.qsize()
                    except Exception as e:
                        self.error = e
                        print((e.message))
                        exit(1)

            self._prefetch_process = Process(target=prefetch)
            self._prefetch_process.start()

            # Terminate the child process when the parent exists
            def cleanup():
                print('Terminating BlobFetcher')
                self._prefetch_process.terminate()
                self._prefetch_process.join()

            import atexit
            atexit.register(cleanup)


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        self._iter += 1

        ims = self._get_next_minibatch()
        im_blob = im_list_to_blob(ims, use_max_size=False)
        # print im_blob.shape
        top[0].reshape(*(im_blob.shape))
        top[0].data[...] = im_blob

        if DEBUG: print(im_blob.shape)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
