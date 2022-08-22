"""
shared constants
"""
import os

IMAGE_SIZE = 512 // 2
BATCH_SIZE = 4
NUM_CLASSES = 2
DATA_DIR = "./unreal-car-dataset"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 500

MODEL_DIR = "./deeplabv3plus_weights.h5"
COLORMAP_DIR = "./instance-level_human_parsing/human_colormap.mat"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
