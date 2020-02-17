import tensorflow as tf
import tensorflow.keras.layers as layers
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 416
IMG_WIDTH = 416

def build(img_w, img_h, grid_w, grid_h, nb_boxes, nb_classes):
    inputs = tf.keras.Input(shape=(img_w, img_h, 3))
    x = layers.Conv2D(16, (1, 1))(inputs)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='sigmoid')(x)
    x = layers.Dense(
        grid_w * grid_h * (nb_boxes * 5 + nb_classes), activation='sigmoid')(x)
    outputs = layers.Reshape(
        (grid_w * grid_h, (nb_boxes * 5 + nb_classes)))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='YoloV3')

data_dir = pathlib.Path('./images')

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
#for f in list_ds.take(5):
    #print(f.numpy())

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label





# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

#for image, label in labeled_ds.take(1):
  #print("Image shape: ", image.numpy().shape)
  #print("Label: ", label.numpy())



def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')

raw_dataset = tf.data.TFRecordDataset(['train.record'])
for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)


train_ds = prepare_for_training(labeled_ds) #I'm pretty sure this is the dataset now

image_batch, label_batch = next(iter(train_ds))

#print(label_batch.numpy())

show_batch(image_batch.numpy(), label_batch.numpy())

def calc_IOU(truth, pred):
    overlap_w = (truth.w + pred.w) / 2 - abs(truth.x - pred.x)
    overlap_h = (truth.h + pred.h) / 2 - abs(truth.y - pred.y)
    overlap_a = overlap_w * overlap_h
    truth_a = truth.w * truth.h
    pred_a = pred.w * pred.h
    union_a = truth_a + pred_a - overlap_a
    return overlap_a / union_a
