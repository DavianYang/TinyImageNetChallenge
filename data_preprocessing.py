from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from io.hdf5datasetgenerator import HDF5DatasetWriter
from config import config

from imutils import paths
import numpy as np
import progressbar
import argparse
import json
import cv2
import os


trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

M = open(config.VAL_MAPPING).read().strip().split('\n')
M = [r.split('\t')[:2] for r in M]
valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.fit_transform([m[1] for m in M])

datasets = [
      ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
      ("val", valPaths, valLabels, config.VAL_HDF5),
      ("test", testPaths, testLabels, config.TEST_HDF5),
]


(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
  print(f"[INFO] building {outputPath}...")
  writer = HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)

  widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
  pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

  for(i, (path, label)) in enumerate(zip(paths, labels)):
    image = cv2.imread(path)

    if dType == "train":
      (b, g, r) = cv2.mean(image)[:3]
      R.append(r)
      G.append(g)
      B.append(b)

    writer.add([image], [label])
    pbar.update()

  pbar.finish()
  writer.close()