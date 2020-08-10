from os import path

basePath = ""

# Images Path
TRAIN_IMAGES = basePath + "/train"
VAL_IMAGES=basePath + "/val/images"

VAL_MAPPING=basePath + "/val/val_annotations.txt"

# Labels Path
WORDNET_IDS=basePath + "/wnids.txt"
WORD_LABELS=basePath + "/words.txt"

# Batch Size
BATCH_SIZE=

# No of classes for train and val
NUM_CLASSES=
NUM_TEST_IMAGES=

# HDF5 File Path
TRAIN_HDF5=basePath + "/hdf5/train.hdf5"
VAL_HDF5=basePath + "/hdf5/val.hdf5"
TEST_HDF5=basePath + "/hdf5/test.hdf5"

OUTPUT_PATH=""

DATASET_MEAN=OUTPUT_PATH + "/tiny-image-net-200-mean.json"

MODEL_PATH=path.sep.join([OUTPUT_PATH, "epoch_60.hdf5"])
FIG_PATH=path.sep.join([OUTPUT_PATH, "deepergooglenet_tinyimagenet.png"])
JSON_PATH=path.sep.join([OUTPUT_PATH, "deepergooglenet_tinyimagenet.json"])

# Netpune API
API_TOKEN=""