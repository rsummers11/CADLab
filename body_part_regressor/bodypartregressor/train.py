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
import os

import _init_paths
import caffe
import google.protobuf as pb2
from caffe.proto import caffe_pb2

from config import cfg
from test import test_net


def snapshot(solver, solver_param):
    """Take a snapshot of the network after unnormalizing the learned
    bounding-box regression weights. This enables easy use at test-time.
    """
    net = solver.net

    filename = (solver_param.snapshot_prefix +
                '_iter_{:d}'.format(solver.iter) + '.caffemodel')

    net.save(str(filename))
    print('Wrote snapshot to: {:s}'.format(filename))

    return filename


def validate_net(model_path, prev_accs):
    print("Validating ...")

    # net.name = experiment_name
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    net = caffe.Net(cfg.val_prototxt, model_path, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(model_path))[0]

    acc = test_net(net, model_path.split('/')[-1], vis=False, prev_accs=prev_accs)
    return acc


def train(solver, solver_param):
    """Network training loop."""
    last_snapshot_iter = -1
    # timer = Timer()
    model_paths = {}
    validate_accuracies = []
    while solver.iter < solver_param.max_iter:
        # Make one SGD update
        # timer.tic()
        solver.step(1)
        # timer.toc()
        # if self.solver.iter % (10 * self.solver_param.display) == 0:
        #     print 'speed: {:.3f}s / iter'.format(timer.average_time)

        need_val = (cfg.TRAIN.DO_VALIDATION and
            solver.iter in cfg.TRAIN.VALIDATION_ITERATION)

        if solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0 \
                or need_val:
            last_snapshot_iter = solver.iter
            snapshot_path = str(snapshot(solver, solver_param))  # snapshot_path seems to be unicode
            model_paths[solver.iter] = snapshot_path

        if need_val:
            acc = validate_net(snapshot_path, validate_accuracies)
            validate_accuracies.append(acc)
            print('validate_accuracies:')
            for i in range(len(validate_accuracies)):
                print(cfg.TRAIN.VALIDATION_ITERATION[i], ':', '%.5f' % validate_accuracies[i])

    if last_snapshot_iter != solver.iter:
        model_paths.append.snapshot()
    return model_paths, validate_accuracies


def train_net(solver_prototxt,
              pretrained_model=None):

    solver = caffe.SGDSolver(solver_prototxt)
    if pretrained_model is not None:
        print(('Loading pretrained model '
               'weights from {:s}').format(pretrained_model))
        solver.net.copy_from(pretrained_model)

    solver_param = caffe_pb2.SolverParameter()
    with open(solver_prototxt, 'rt') as f:
        pb2.text_format.Merge(f.read(), solver_param)

    print('Solving...')
    model_paths, validate_acc = train(solver, solver_param)
    print('done solving')
    return model_paths, validate_acc
