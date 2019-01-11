from keras_med_io.utils.gen_utils import BaseGenerator
from keras_med_io.local.helperfunc import transforms
import tensorflow as tf

def docs():
    '''
    Is it better to have the data aug during parsing or during data_gen?
        probs during data_gen
        Problem is that you can only apply it once and data aug isn't real time
            is also a limitation of the TF data API as well
    '''
class DatasetGenerator(BaseGenerator):
    '''
    Simple generator that reads .npy arrays (channels_last) without the batch_size dim
    '''
    def __init__(self, list_IDs, data_dirs, batch_size, ndim, fraction = 0.0, variance = 0.1, shuffle = True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.ndim = ndim
        self.fraction = fraction
        self.variance = variance
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.list_IDs))

    def data_gen(self, list_IDs_temp):
        '''
        Preprocesses the data
        Args:
            list_IDs_temp: a batch of list_IDs
        Returns
            x, y
        '''
        patches_x = []
        patches_y = []
        for id in list_IDs_temp:
            arr_x, arr_y = np.load(os.path.join(self.data_dirs[0] + id)), np.load(os.path.join(self.data_dirs[1] + id))
            assert self.ndim == len(arr_x.shape[:-1]), "Please make sure that your arr does not have the batch_size dimension \
                                                        and that the data is channels_last mode"
            # data augmentation
            augmented_x, augmented_y = transforms(arr_x, arr_y, self.ndim, self.fraction, self.variance)

        self.shape = patches_x[0].shape
        return np.stack(augmented_x), np.stack(augmented_y)

    def dataset_from_gen(self, number_epochs, batch_size=1,
                               num_parallel_calls=8, mode='train', fraction=0.0,
                               shuffle_size=20, buffer_size = 10, variance=0.1, compression=None,
                               multi_GPU=False):
            '''
            Args:
                number_epochs:
                batch_size:
                num_parallel_calls:
                mode:
                fraction:
                shuffle_size:
                variance:
                compression:
                multi_GPU:
            '''
            self.mode = mode
            self.batch_size = batch_size
            self.fraction = fraction
            self.variance = variance

            if compression:
                compression_type_ = 'GZIP'
            else:
                compression_type_ = ''

            generator = DatasetGenerator(self.list_IDs,
                                        self.data_dirs,
                                        self.batch_size,
                                        self.ndim,
                                        self.fraction,
                                        self.variance,
                                        self.shuffle)

            dataset = tf.data.Dataset.from_generator(generator,
                                                     output_types = [tf.float32, tf.uint8],
                                                     output_shapes = (self.shape, self.shape)) # for segmentation
            dataset = dataset.apply(
                        tf.contrib.data.shuffle_and_repeat(shuffle_size,
                                                           count=number_epochs))
            # dataset = dataset.apply(
            #             tf.contrib.data.map_and_batch(
            #                 map_func=self.parse_image,
            #                 num_parallel_batches=num_parallel_calls,
            #                 batch_size=batch_size))

            dataset = dataset.prefetch(buffer_size=buffer_size)

            if multi_GPU:
                return dataset
            else:
                iterator = dataset.make_one_shot_iterator()
                inputs, outputs = iterator.get_next()
                return inputs, outputs
