"""Dataset loading utilities.

All images are scaled to [0, 255] instead of [0, 1]
"""

import functools

import numpy as np

import torch
from torchvision import transforms
import torch.utils.data as data
import torchvision.datasets as datasets

class SimpleDataset(data.Dataset):
    DATASET_NAMES = ('cifar10', 'celebahq256')

    def __init__(self, name, tfds_data_dir):
        self._name = name
        self._data_dir = tfds_data_dir
        self._img_size = {'cifar10': 32, 'celebahq256': 256}[name]
        self._img_shape = [self._img_size, self._img_size, 3]
        self._tfds_name = {
            'cifar10': 'cifar10:3.0.0',
            'celebahq256': 'celeb_a_hq/256:2.0.0',
        }[name]
        self.num_train_examples, self.num_eval_examples = {
            'cifar10': (50000, 10000),
            'celebahq256': (30000, 0),
        }[name]
        self.num_classes = 1  # unconditional
        self.eval_split_name = {
            'cifar10': 'test',
            'celebahq256': None,
        }[name]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image data
        ])

        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self._name == 'cifar10':
            return datasets.CIFAR10(self._data_dir, train=True, download=True, transform=self.transform)
        elif self._name == 'celebahq256':
            # Custom code to load celebahq256 dataset (not available in torchvision)
            # Replace this with your own code to load the CelebA-HQ 256x256 dataset
            raise NotImplementedError("Code to load celebahq256 dataset is not implemented.")
        else:
            raise ValueError(f"Unsupported dataset name: {self._name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, 0  # Return a tuple (image, label)

    @property
    def image_shape(self):
        """Returns a tuple with the image shape."""
        return tuple(self._img_shape)



def pack(image, label):
  label = tf.cast(label, tf.int32)
  return {'image': image, 'label': label}


class LsunDataset:
  def __init__(self,
    tfr_file,            # Path to tfrecord file.
    resolution=256,      # Dataset resolution.
    max_images=None,     # Maximum number of images to use, None = use all images.
    shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
    buffer_mb=256,       # Read buffer size (megabytes).
  ):
    """Adapted from https://github.com/NVlabs/stylegan2/blob/master/training/dataset.py.
    Use StyleGAN2 dataset_tool.py to generate tf record files.
    """
    self.tfr_file           = tfr_file
    self.dtype              = 'int32'
    self.max_images         = max_images
    self.buffer_mb          = buffer_mb
    self.num_classes        = 1         # unconditional

    # Determine shape and resolution.
    self.resolution = resolution
    self.resolution_log2 = int(np.log2(self.resolution))
    self.image_shape = [self.resolution, self.resolution, 3]

  def _train_input_fn(self, params, one_pass: bool):
    # Build TF expressions.
    dset = tf.data.TFRecordDataset(self.tfr_file, compression_type='', buffer_size=self.buffer_mb<<20)
    if self.max_images is not None:
      dset = dset.take(self.max_images)
    if not one_pass:
      dset = dset.repeat()
    dset = dset.map(self._parse_tfrecord_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle and prefetch
    dset = dset.shuffle(50000)
    dset = dset.batch(params['batch_size'], drop_remainder=True)
    dset = dset.prefetch(tf.data.experimental.AUTOTUNE)
    return dset

  def train_input_fn(self, params):
    return self._train_input_fn(params, one_pass=False)

  def train_one_pass_input_fn(self, params):
    return self._train_input_fn(params, one_pass=True)

  def eval_input_fn(self, params):
    return None

  # Parse individual image from a tfrecords file into TensorFlow expression.
  def _parse_tfrecord_tf(self, record):
    features = tf.parse_single_example(record, features={
      'shape': tf.FixedLenFeature([3], tf.int64),
      'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    data = tf.cast(data, tf.int32)
    data = tf.reshape(data, features['shape'])
    data = tf.transpose(data, [1, 2, 0])  # CHW -> HWC
    data.set_shape(self.image_shape)
    return pack(image=data, label=tf.constant(0, dtype=tf.int32))


DATASETS = {
  "cifar10": functools.partial(SimpleDataset, name="cifar10"),
  "celebahq256": functools.partial(SimpleDataset, name="celebahq256"),
  "lsun": LsunDataset,
}


def get_dataset(name, *, tfds_data_dir=None, tfr_file=None, seed=547):
  """Instantiates a data set and sets the random seed."""
  if name not in DATASETS:
    raise ValueError("Dataset %s is not available." % name)
  kwargs = {}

  if name == 'lsun':
    # LsunDataset takes the path to the tf record, not a directory
    assert tfr_file is not None
    kwargs['tfr_file'] = tfr_file
  else:
    kwargs['tfds_data_dir'] = tfds_data_dir

  if name not in ['lsun', *SimpleDataset.DATASET_NAMES]:
    kwargs['seed'] = seed

  return DATASETS[name](**kwargs)
