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

from time import clock, strftime
import os
import argparse
import pprint
import numpy as np


from train import train_net
# import _init_paths
import caffe
from config import cfg, cfg_from_file


default_net = 'VGG16.v2'

# these paths are for step-by-step debugging in PyCharm
exp_name = 'SSBR'
default_pretrained_model = "~/ct/data/imagenet_models/{}.caffemodel".format(default_net)
train_imdb = 'train_volume_list_example.txt'

default_solver = "solver.prototxt"
default_cfg = "config.yml"
test_prototxt = "test.prototxt"

#os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project


def parse_args_train():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--name', dest='exp_name',
                        help='experiment name',
                        default=exp_name, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=default_solver, type=str)
    parser.add_argument('--test_prototxt', dest='test_prototxt',
                        help='test prototxt',
                        default=test_prototxt, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=default_cfg, type=str)
    parser.add_argument('--train_imdb', dest='train_imdb',
                        help='dataset to train on',
                        default=train_imdb, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    print('Doing training...')
    args = parse_args_train()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id
    cfg.train_imdb = args.train_imdb
    cfg.val_prototxt = args.test_prototxt
    cfg.test_prototxt = args.test_prototxt
    cfg.TRAIN.VALIDATION_ITERATION = eval(cfg.TRAIN.VALIDATION_ITERATION)

    print('Using config:')
    pprint.pprint(cfg)

    # if not args.randomize:
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    from easydict import EasyDict as edict
    imdb = edict()
    imdb.name = cfg.train_imdb

    # train
    start = clock()
    model_paths, validate_acc = train_net(args.solver,
                                          pretrained_model=args.pretrained_model)
    elapse_time_train = (clock() - start) / 60
    print('Training time', elapse_time_train, 'min')
