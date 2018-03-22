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

import caffe

from config import cfg


DEBUG = False

class CompareDiffLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""
        self.layer_params = yaml.load(self.param_str)
        self.slice_num =  cfg.TRAIN.SLICE_NUM
        self.group_size = cfg.TRAIN.GROUPS_PER_BATCH
        self.out_num = self.group_size * (self.slice_num - 2)
        self.in_num = self.group_size * self.slice_num
        assert self.in_num == bottom[0].data.shape[0]

        top[0].reshape(self.out_num,1)
        top[1].reshape(self.out_num,1)

        self.Af = np.zeros((self.out_num, self.in_num))
        idx1 = 0
        idx2 = 0
        for i in xrange(self.group_size):
            for j in xrange(self.slice_num-2):
                self.Af[idx1, idx2:idx2+3] = [1,-2,1]
                idx1 += 1
                idx2 += 1
            idx2 += 2
        self.Ab = self.Af.transpose()

    def reshape(self, bottom, top):
        top[0].reshape(self.out_num,1)
        top[1].reshape(self.out_num,1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        vals = bottom[0].data.reshape((self.in_num, 1))
        out = self.Af.dot(vals)
        top[0].data[...] = out
        top[1].data[...] = np.zeros((self.out_num, 1))

        if DEBUG:
            print 'reg_values:', bottom[0].data.ravel().tolist()
            print 'diff values:', out

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        top_diff = top[0].diff
        out = self.Ab.dot(top_diff)
        #print out
	bottom[0].diff[...] = out.reshape(*bottom[0].diff.shape)
