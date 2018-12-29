import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
import json
import argparse
import logging
from helperfunc import *
# from utils.sitk_utils import resample_to_spacing
from utils.utils import start_logger
from keras_med_io.utils.io_func import get_multi_class_labels, resample_img
from glob import glob

class TFRdataset(object):
    '''
    Goal: To read all the preprocessed .npy patches and make a tfrecords dataset
    '''
    def __init__(self, in_dirs):
        self.in_dir = in_dir
        self.dset_path = os.path.join(self.fpath, 'dataset.tfrecord')
        dformat = '*.npy'
        self.inputs_paths = glob(os.path.join(self.in_dir, 'imageTr'), recursive = True)
        self.labels_paths = glob(os.path.join(self.in_dir, 'labelsTr'), recursive = True)

    def make_train_dataset(self, save_interpolation=None, compression=None,
                           output_path=None):
        # initialising the tfrecord writer
        if compression:
            options = tf.python_io.TFRecordOptions(
                            tf.python_io.TFRecordCompressionType.GZIP)
        else:
            options = tf.python_io.TFRecordOptions(
                            tf.python_io.TFRecordCompressionType.NONE)
        self.writer = tf.python_io.TFRecordWriter(
                            os.path.join(output_path, 'dataset.tfrecord'),
                            options=options)
        holder = {}
        logging.info('Constructing training dataset')
        # loading and writing
        for img_pair in zip(self.inputs_paths, self.labels_paths):
            holder['idx'] = img_pair[0].split('./')[-1]
            holder['image'] = np.load(img_pair[0])
            holder['label'] = np.load(img_pair[1])
            self.write_to_tfrecord(holder)
            logging.info(holder['idx'] + ' processed')
        self.writer.close()

    def write_to_tfrecord(self, holder):
        # type checking
        try:
            assert(holder[0].dtype == np.float32)
        except AssertionError:
            logging.info(holder['idx'] + ': np.float32 expected, got {}, \
            converting to np.float32'.format(holder['image'].dtype))
            holder['image'] = holder['image'].astype(np.float32)

        try:
            assert(holder['label'].dtype == np.uint8)
        except AssertionError:
            logging.info(holder['idx'] + ': np.uint8 expected, got {}, \
            converting to uint8'.format(holder['label'].dtype))
            holder['label'] = holder['label'].astype(np.uint8)

        # shape checking
        if len(holder['image'].shape) == 4:
            try:
                assert(holder['image'].shape[:-1]
                       == holder['label'].shape[:-1])
            except AssertionError:
                logging.warning(holder['idx'] + ':expected shapes to be \
                                equal, but image was {} and label was {}'.format(
                                    holder['image'].shape[:-1],
                                    holder['label'].shape[:-1]))
        else:
            try:
                assert(holder['image'].shape == holder['label'].shape)
            except AssertionError:
                logging.warning(holder['idx'] + ':expected shapes to be \
                                equal, but image was {} and label was {}'.format(
                                    holder['image'].shape,
                                    holder['label'].shape))

        raw_image = holder['image'].tobytes()
        raw_label = holder['label'].tobytes()
        shape_i = np.array(holder['image'].shape, np.int32).tobytes()
        shape_l = np.array(holder['label'].shape, np.int32).tobytes()

        # 32bit constrained from google protobof
        Kint32Max = 2147483647.0

        if Kint32Max - len(raw_image) - len(raw_label) - len(shape_i) - len(shape_l) < 0:
            logging.warning(
                holder['idx'] + ': raw bytecounts > Kint32Max, \
                cropping image and label')

        entry = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(raw_image),
                        'shape_i': _bytes_feature(shape_i),
                        'img_dtype': _bytes_feature(
                            tf.compat.as_bytes(str(holder['image'].dtype))),
                        'label': _bytes_feature(raw_label),
                        'shape_l': _bytes_feature(shape_l),
                        'label_dtype': _bytes_feature(
                            tf.compat.as_bytes(str(holder['label'].dtype))),
                        'idx': _bytes_feature(
                            tf.compat.as_bytes(holder['idx'])),
                    }))
        self.writer.write(entry.SerializeToString())

    def provide_tf_dataset(self, number_epochs, shape=None, batch_size=1,
                           num_parallel_calls=8, mode='train', fraction=0.0,
                           shuffle_size=100, variance=0.1, compression=None,
                           multi_GPU=False):

        self.mode = mode
        self.shape = shape
        self.batch_size = batch_size
        self.fraction = fraction
        self.variance = variance

        if compression:
            compression_type_ = 'GZIP'
        else:
            compression_type_ = ''
        dataset = tf.data.TFRecordDataset(
                    self.dset_path, compression_type=compression_type_)
        dataset = dataset.apply(
                    tf.contrib.data.shuffle_and_repeat(shuffle_size,
                                                       count=number_epochs))
        dataset = dataset.apply(
                    tf.contrib.data.map_and_batch(
                        map_func=self.parse_image,
                        num_parallel_batches=num_parallel_calls,
                        batch_size=batch_size))

        dataset = dataset.prefetch(buffer_size=None)

        if multi_GPU:
            return dataset

        else:
            iterator = dataset.make_one_shot_iterator()
            inputs, outputs = iterator.get_next()
            return inputs, outputs

    def parse_image(self, entry):
        parsed = tf.parse_single_example(entry, features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'shape_i': tf.FixedLenFeature([], tf.string),
                    'img_dtype': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string),
                    'shape_l': tf.FixedLenFeature([], tf.string),
                    'label_dtype': tf.FixedLenFeature([], tf.string),
                    'idx': tf.FixedLenFeature([], tf.string),
                    })

        # TODO: make dtype flexible
        shape_i = tf.decode_raw(parsed['shape_i'], tf.int32)
        image = tf.decode_raw(parsed['image'], tf.float32)
        image = tf.reshape(image, shape_i)
        image = tf.cond(
                    tf.greater_equal(
                        tf.rank(image), 4),
                    lambda: image,
                    lambda: tf.expand_dims(image, 0))

        shape_l = tf.decode_raw(parsed['shape_l'], tf.int32)
        mask = tf.decode_raw(parsed['label'], tf.uint8)
        mask = tf.reshape(mask, shape_l)

        image, mask, rval = tf.py_func(
                                        transforms,
                                        [image, mask, self.shape, self.fraction, self.spacing, self.variance],
                                        Tout=[tf.float32, tf.float32, tf.float64])
        cropped_s = tf.cast(cropped_s, tf.uint8)

        # static output shape required by tf.dataset.batch() method!
        cropped_i.set_shape([self.num_modalities] + self.shape)
        cropped_s.set_shape([self.num_classes] + self.shape)
        print("input shape" + str(cropped_i.shape))
        print("segmentation shape" + str(cropped_s.shape))

        if self.mode == 'train':
            return({'img': cropped_i, 'object_in_crop': objects_in_crop},
                   cropped_s)
        elif self.mode == 'inference':
            tf.Print(['img shape'], [tf.shape(stacked[:self.num_modalities])])
            return({'img': cropped_i, 'rval': rval},
                   cropped_s)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='generate dataset for'
                                     'MSD_challenge')
    parser.add_argument('-i', '--input', help='input (root) path',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='path to store generated TFRecord and json',
                        required=True)
    parser.add_argument('-s', '--save_interpolation',
                        help='save interpolated images',
                        required=False,
                        default=None)
    parser.add_argument('-c', '--compression',
                        help='compress generated TFRecord',
                        required=False,
                        default=None)

    args = vars(parser.parse_args())

    fpath = args['input']
    output_path = args['output']
    compression = args['compression']
    save_images = args['save_interpolation']

    start_logger(os.path.join(fpath + 'dataset_generation'))
    logger = logging.getLogger(__name__)
    dset = TFRdataset(fpath)
    if dset.spacing is None or dset.shapes is None:
        dset.scan_dataset_spacing_shapes()
    dset.make_train_dataset(output_path=output_path,
                            compression=compression,
                            save_interpolation=save_images)
