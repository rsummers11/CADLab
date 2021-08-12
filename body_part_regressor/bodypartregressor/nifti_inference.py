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
import numpy as np
from config import cfg, cfg_from_file
from PIL import Image
import nibabel as nib
import os
import sys
from sklearn.linear_model import LinearRegression


default_model = os.path.join(os.path.dirname(__file__), 'snapshots/_iter_10000.caffemodel') # trained model
default_prototxt = os.path.join(os.path.dirname(__file__), "test.prototxt")
default_cfg = os.path.join(os.path.dirname(__file__), "config.yml")
GPU_ID = 0

#-------------------------------------------------------------------------------
def nifti_inference(ct, progressbar=False):
    ''' inference routine that takes in a ct as a numpy array
        converts each 2D transverse slice of the 3D numpy array to desired
        8 bit image format (soft tissue window (-175 - 275 HU) with size 128x128.
        arguments:
            ct : numpy array, 3D - the CT volumes
            progressbar : Boolean - display graphical progressbar (default : False)
        returns:
            vals : list - the slice scores
    '''
    if (progressbar):
        from tqdm import tqdm

    cfg_from_file(default_cfg)
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    net = caffe.Net(default_prototxt, default_model, caffe.TEST)

    sx, sy, sz = ct.shape
    cliplow = -175
    cliphigh = 275

    ct = (np.clip(ct, cliplow, cliphigh) - cliplow) / ((cliphigh - cliplow))*255

    vals = np.empty((sz,))

    if (progressbar):
        todo = tqdm(range(sz))
    else:
        todo = range(sz)

    for iz in todo:
        slice = ct[:,:,iz]
        slice = np.flipud(np.rot90(slice))
        img = Image.fromarray(slice)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((128,128))

        ##img.save(str(iz)+'.png')

        img -= cfg.PIXEL_MEANS

        data = np.zeros((1, 3, img.shape[0], img.shape[1]), dtype=np.float32)
        for chn in range(3):
            data[0, chn, :, :] = img[:,:,chn]

        blobs_out = net.forward(data=data)
        vals[iz] = blobs_out['reg_value']
        #print(vals[iz])

    #fn = os.path.join(sys.path[0], 'slice_scores.txt')
    #with open(fn,'w') as f:
    #    f.writelines([a + '\t' + str(b) + '\r\n' for a,b in zip(image_index, vals)])
    #print('Written to', fn)
    return list(vals)

if __name__ == '__main__':

    OFFSET = 0 # can be set to -1024

    file = sys.argv[1]

    print("running", file)
    ct = nib.load(file).get_data().astype(np.float32) + OFFSET

    vals = nifti_inference(ct)

    sx, sy, sz = ct.shape

    #-------------------- linear regression fitting
    xs = np.array([x for x in range(sz)]).reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg = reg.fit(xs, vals)
    m = float(reg.coef_)
    b = float(reg.intercept_)
    lin_reg_line = reg.predict(xs)

    filename = os.path.split(file)[-1]

    #------------ visualization of linear regression fit
    #import matplotlib.pyplot as plt
    #plt.figure(figsize=(15,5))
    #plt.plot(vals, ".")
    #plt.plot(lin_reg_line, "-")
    #plt.savefig(filename+"_scores_viz.png")

    fn = 'slice_scores_'+str(filename)+'.txt'

    #with open(fn, 'w') as f: #ke's original output file
        #f.writelines([str(a) + '\t' + str(b) + '\r\n' for a,b in zip(list(range(sz)), vals)])

    with open(fn, 'w') as f: #new output with two columns
        lines = []
        for il in range(sz):
            lines += [str(il) + ',' + str(vals[il]) + ',' + str(lin_reg_line[il]) + '\n']
        f.writelines(lines) #new file with

    print('Results written to', fn)
