from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from neptunecontrib.monitoring.keras import NeptuneMonitor
from config import config
import argparse
import json
import datetime

from alexnet import AlexNet
from googlenet import GoogLeNet
from resnet import ResNet

from prepocessing import SimplePreprocessor, MeanPreprocessor, ImageToArrayPreprocessor
from io.hdf5datasetgenerator import HDF5DatasetGenerator
from utils.neptune_experiment import create_experiment
from utils.poly_decay import poly_decay
from utils import EpochCheckpoint, NeptuneLogger

parser = argparse.ArgumentParser(description="Data Preprocessing")
parser.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
parser.add_argument("-m", "--model", required=True, type=str, help='path to *specifi* model checkpoints to load')
parser.add_argument('-s', "--start", type=int, default=0, help="Epoch to start the training at")
parser.add_argument("-e", "--epochs", type=int, default=1e-2, help='number of epochs for training')
parser.add_argument("-l", "--learning_rate", type=int, default=1e-2, help='learning rate for training')
parser.add_argument("-r", "--reg", type=int, default=0.0002, help='regularization for training')


args = vars(parser.parse_args())

experiment = create_experiment(exp_no=1, model=args["model"].__name__)

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15, 
                          width_shift_range=0.2, height_shift_range=0.2, 
                          shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

if args["model"] is None:
    print(f"[INFO] compiling model...{args["model"]}")
    model = GoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=args["reg"])
    model.summary(print_fn=lambda x: experiment.log_text('model_summary', x))

    opt = SGD(learning_rate=args["learning_rate"], momentum=0.9)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print(f"[INFO] loading {args['model']}...")
    model = load_model(args["model"])

    print(f"[INFO] old learning rate: {K.get_value(model.optimizer.lr)}")
    K.set_value(model.optimizer.lr, args["learning_rate"])
    
    print(f"[INFO] new learning rate: {K.get_value(model.optimizer.lr)}")

experiment.log_metric('learning_rate', args["learning_rate"])
callbacks = [
      EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start"]),
      LearningRateScheduler(poly_decay),
      NeptuneLogger(experiment),
      NeptuneMonitor()
]

model.fit(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=args["epochs"],
    max_queue_size=config.BATCH_SIZE * 2,
    callbacks=callbacks,
    verbose=1


trainGen.close()
valGen.close()