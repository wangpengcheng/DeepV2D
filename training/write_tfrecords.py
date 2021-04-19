import sys
sys.path.append('deepv2d')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import csv
import cv2
import os
import time
import argparse
import random
from core import config
from data_stream.nyu import NYU
from data_stream.kitti import KittiRaw


def to_tfrecord(data_blob, xi ):
    """Write (image, depth) pair to tfrecords example"""

    id = np.array(xi, dtype=np.int32).tobytes()
    dim = np.array([5, 240, 320, 3], dtype=np.int32).tobytes()

    images = np.array(data_blob[0], dtype=np.uint8).tobytes()
    poses = np.array(data_blob[1], dtype=np.float32).tobytes()
    depth = np.array(data_blob[2], dtype=np.float32).tobytes()
    filled = np.array(data_blob[3], dtype=np.float32).tobytes()
    intrinsics = np.array(data_blob[5], dtype=np.float32).tobytes()


    example = tf.train.Example(features=tf.train.Features(feature={
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id])),
        'dim': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dim])),
        'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images])),
        'poses': tf.train.Feature(bytes_list=tf.train.BytesList(value=[poses])),
        'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[depth])),
        'filled': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filled])),
        'intrinsics': tf.train.Feature(bytes_list=tf.train.BytesList(value=[intrinsics])),
    }))

    return example


def main_nyu(args):

    np.random.seed(1234)
    cfg = config.cfg_from_file(args.cfg)
    db = NYU(cfg.INPUT.RESIZE, args.dataset_dir)
    ix = np.arange(len(db))
    np.random.shuffle(ix)

    tfwriter = tf.python_io.TFRecordWriter(args.records_file)
    for i in range(len(ix)):
        if i%100 == 0:
            print("Writing example %d of %d"%(i, len(ix)))
        data_blob = db[ix[i]]
        record = to_tfrecord(data_blob, ix[i])
        tfwriter.write(record.SerializeToString())

    tfwriter.close()


def main_kitti(args):

    np.random.seed(1234)

    db = KittiRaw(args.dataset_dir)
    # 构建数组
    ix = np.arange(len(db))
    np.random.shuffle(ix)

    tfwriter = tf.python_io.TFRecordWriter(args.records_file)
    for i in range(len(ix)):
        if i%100 == 0:
            print("Writing example %d of %d"%(i, len(ix)))
        data_blob = db[ix[i]]
        record = to_tfrecord(data_blob, ix[i])
        tfwriter.write(record.SerializeToString())

    tfwriter.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='cfg file')
    parser.add_argument('--dataset', help='kitti or nyu')
    parser.add_argument('--dataset_dir', help='path to dataset directory')
    parser.add_argument('--records_file', help='path to dataset directory')
    args = parser.parse_args()
  
    if args.dataset == 'nyu':
        main_nyu(args)

    elif args.dataset == 'kitti':
        main_kitti(args)
