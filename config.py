import tensorflow as tf
import random

# Dataset
DATA_ROOT = './dataset/'
OBJECT_RATE = 0.5
ROTATED_BOXES = False

VOXEL_SIZE = lambda : random.choice([2])
POINTSET_RADIUS = lambda : random.choice([1])

# Sequential nets
MAX_ANGLE = 90.     # for angle class convertion in <dataset>
ANGLE_BINS = 9
SEG_UNITS = [64,32,16,8,2]
VOTING_UNITS = [32, 32, 32+3]     # 8, 8, 16, 16+3
PROPOSAL_UNITS = [32, 32, 32, (ANGLE_BINS * 2) + 3]  # (num_ang_bins*2) + size

# Training
MAX_ITER = 300000
SUMMARY_ITER = 1
VALIDATION_ITER = 15
SAVE_ITER = 3
LR = 1e-3
LR_CHANGE_EPOCH = 2
LR_CHANGE_VAL = 10
LR_MINIMUM = 1e-3
LOSS_WEIGHTS = [50., 1., 0.1, 0.1, 0.01, 0.001]    # sem, center, angle_cls, angle_res, sizes, corners
SEG_WEIGHT_1s, SEG_WEIGHT_0s = 0.7, 0.3  # segmentation weights
NUMB_OBJ_TO_LEARN = 1    # number of  objects to learn per step
IOU_NUMBER = 80     # number of IOU calculations considered
OBJ_RATIO = 0.75     # the ratio of object points from batch over the original number of points constituting the object
BATCH_WINDOW_RADIUS = lambda : random.choice([50])
DETECT_WINDOW_RADIUS = 100

BIAS = True
AUGMENT= False
IS_TRAINING = False
BATCH_NORM = False
WEIGHTS_REG = tf.keras.regularizers.L1L2(0.1,0.0)

SUMMARY_DIR = './summary/'
MODEL_DIR = 'train_5/'
CHECKPOINT_DIR = './checkpoint/'+MODEL_DIR
MODEL_NAME = "model.ckpt"
FOLDERS_TO_USE = []
