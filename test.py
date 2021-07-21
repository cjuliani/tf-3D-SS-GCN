# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from solver import Solver
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_memory", default=0.1, help='Memory fraction of GPU used for training.')
parser.add_argument("--gpu_allow_growth", default=False)
parser.add_argument("--soft_placement", default=True)
parser.add_argument("--model", type=str, default='train_1')
ARGS, unknown = parser.parse_known_args()

tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    solver = Solver()
    # soft_placement:  parts of your network (which didn't fit in the GPU's memory) might be placed at the CPU
    tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=bool(ARGS.soft_placement))
    tf_config.gpu_options.allow_growth = bool(ARGS.gpu_allow_growth)
    tf_config.gpu_options.per_process_gpu_memory_fraction = float(ARGS.gpu_memory)
    solver.test(model=ARGS.model, sess_config=tf_config)
