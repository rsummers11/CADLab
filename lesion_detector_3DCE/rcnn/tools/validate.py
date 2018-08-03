import mxnet as mx
from rcnn.logger import logger
from rcnn.config import config, default
from rcnn.tools.test import test_rcnn
import numpy as np
import sys
import os


def validate(prefix, iter_no):
    logger.info('Validating ...')
    default.testing = True
    ctx = mx.gpu(int(default.val_gpu))
    # ctx = mx.gpu(int(default.gpus.split(',')[0]))
    epoch = iter_no + 1
    acc = test_rcnn(default.network, default.dataset, default.val_image_set,
                      default.dataset_path,
                      ctx, prefix, epoch,
                      default.val_vis, default.val_shuffle,
                      default.val_has_rpn, default.proposal,
                      default.val_max_box, default.val_thresh)

    fn = '%s-%04d.params' % (prefix, epoch)
    fn_to_del = None
    if len(default.accs.keys()) == 0:
        default.best_model = fn
        default.best_acc = acc
        default.best_epoch = epoch
    else:
        if acc > default.best_acc:
            fn_to_del = default.best_model
            default.best_model = fn
            default.best_acc = acc
            default.best_epoch = epoch
        else:
            fn_to_del = fn

    default.accs[str(epoch)] = acc
    epochs = np.sort([int(a) for a in default.accs.keys()]).tolist()
    for e in epochs:
        print 'Iter %s: %.4f' % (e, default.accs[str(e)])
    sys.stdout.flush()

    if default.keep_best_model and fn_to_del:
        os.remove(fn_to_del)
        print fn_to_del, 'deleted to keep only the best model'
        sys.stdout.flush()

    default.testing = False
