import tensorflow as tf
import numpy as np
import os
import argparse
import logging
from keras_med_io.local.helperfunc import *
from glob import glob

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class TFRdataset(object):
    '''
    Goal: To read all the preprocessed .npy patches and make a tfrecords dataset
     -> Assumes file structure:
        * in_dir
            -> saves tfrecord here
            * imagesTr
            * labelsTr
     -> Assumes that files/labels have same corresponding names
    '''
    def __init__(self, in_dir, n_dim, shape):
        self.in_dir = in_dir
        self.n_dim = n_dim

        self.dset_path = os.path.join(self.in_dir, 'dataset.tfrecord')
        dformat = '*.npy'
        self.inputs_path = os.path.join(self.in_dir, 'imagesTr')
        self.labels_path = os.path.join(self.in_dir, 'labelsTr')
        self.ids = os.listdir(self.labels_path) #assumes that inputs and labels have same names
        np.random.shuffle(self.ids)
        # should include channels (...., n_channels)
        self.shape = shape
        # assumes all .npy arrays have the same shape

    def make_train_dataset(self, compression=None, output_path=None):
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
        for id in self.ids:
            holder['idx'] = id
            # holder['image'] = np.expand_dims(np.load(os.path.join(self.inputs_path, id)), 0) # adding batch_size dim
            # holder['label'] = np.expand_dims(np.load(os.path.join(self.labels_path, id)), 0)
            holder['image'] = np.load(os.path.join(self.inputs_path, id)) # (..., n_channels)
            holder['label'] = np.load(os.path.join(self.labels_path, id))
            self.write_to_tfrecord(holder)
            logging.info(holder['idx'] + ' processed' + \
                        '\nShape: ' + str(holder['image'].shape))
        self.writer.close()

    def write_to_tfrecord(self, holder):
        # type checking
        try:
            assert(holder['image'].dtype == np.float32)
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

        # # shape checking
        # try:
        #     assert(holder['image'].shape[:-1]
        #            == holder['label'].shape[:-1])
        # except AssertionError:
        #     logging.warning(holder['idx'] + ':expected shapes to be \
        #                     equal, but image was {} and label was {}'.format(
        #                         holder['image'].shape[:-1],
        #                         holder['label'].shape[:-1]))

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

    def provide_tf_dataset(self, number_epochs, batch_size=1,
                           num_parallel_calls=8, mode='train', fraction=0.0,
                           shuffle_size=100, variance=0.1, compression=None,
                           multi_GPU=False):

        self.mode = mode
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
        '''
        Decodes the TFRecordDataset and does data aug
        '''
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
        # image = tf.cond(
        #             tf.greater_equal(
        #                 tf.rank(image), 4),
        #             lambda: image,
        #             lambda: tf.expand_dims(image, 0))

        shape_l = tf.decode_raw(parsed['shape_l'], tf.int32)
        mask = tf.decode_raw(parsed['label'], tf.uint8)
        mask = tf.reshape(mask, shape_l)

        # data aug
        image, mask = tf.py_func(
                                 transforms,
                                 [image, mask, self.n_dim , self.fraction, self.variance],
                                 Tout=[tf.float32, tf.float32])
        mask = tf.cast(mask, tf.uint8)
        # static output shape required by tf.dataset.batch() method!
        # assumes shape (..., n_channels)
        image.set_shape(self.shape)
        mask.set_shape(self.shape)

        print("input shape: " + str(image.shape))
        print("segmentation shape: " + str(image.shape))

        # what's the difference??
        if self.mode == 'train':
            return(image, mask)
        elif self.mode == 'inference':
            # tf.Print(['img shape'], [tf.shape(stacked[:self.num_modalities])])
            return(image, mask)

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='generate dataset for'
#                                      'MSD_challenge')
#     parser.add_argument('-i', '--input', help='input (root) path',
#                         required=True)
#     parser.add_argument('-o', '--output',
#                         help='path to store generated TFRecord and json',
#                         required=True)
#     parser.add_argument('-s', '--save_interpolation',
#                         help='save interpolated images',
#                         required=False,
#                         default=None)
#     parser.add_argument('-c', '--compression',
#                         help='compress generated TFRecord',
#                         required=False,
#                         default=None)
#
#     args = vars(parser.parse_args())
#
#     fpath = args['input']
#     output_path = args['output']
#     compression = args['compression']
#     # save_images = args['save_interpolation']
#
#     start_logger(os.path.join(fpath + 'dataset_generation'))
#     logger = logging.getLogger(__name__)
#     dset = TFRdataset(fpath)
#     if dset.spacing is None or dset.shapes is None:
#         dset.scan_dataset_spacing_shapes()
#     dset.make_train_dataset(output_path=output_path,
#                             compression=compression,
#                             )
#                             # save_interpolation=save_images)
