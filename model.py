import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras
from keras import backend as k
import IPython.display as display
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pathlib
import io

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 416
IMG_WIDTH = 416
GRID_CELLS = 12
N_BOXES = 2
N_CLASSES = 1


def build(img_w, img_h, grid_w, grid_h, n_boxes, n_classes):
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
        grid_w * grid_h * (n_boxes * 5 + n_classes), activation='sigmoid')(x)
    outputs = layers.Reshape(
        (grid_w * grid_h, (n_boxes * 5 + n_classes)))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='YoloV3')


raw_image_dataset = tf.data.TFRecordDataset('./train.record')

# Create a dictionary describing the features.
image_feature_description = {"image/filename": tf.io.FixedLenFeature((), tf.string, ""),
                             'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                             'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
                             'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                             'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                             'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                             'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset = parsed_image_dataset.shuffle(500)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def encode_img(tensor):
    # resize the image to the desired size.
    # Use `convert_image_dtype` to convert to uint8.
    img = tf.image.convert_image_dtype(tensor, tf.uint8)
    # convert the 3D uint8 tensor to a compressed string
    return tf.image.encode_jpeg(img)


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def format_dataset(dataset):
    images = []
    labels = []
    for element in dataset:
        # Map elements through function to turn them into tuples with format (image, label)
        filename = element['image/filename'].numpy().decode('utf-8')
        image_tensor = process_path('./images/power_cell/' + filename)
        # Map labels to format [gridPos, [x, y, w, h, conf]]
        label_tensor = k.zeros([GRID_CELLS ** 2, 2 * N_BOXES + N_CLASSES])
        x_mins = element['image/object/bbox/xmin'].values.numpy()
        x_maxes = element['image/object/bbox/xmax'].values.numpy()
        y_mins = element['image/object/bbox/ymin'].values.numpy()
        y_maxes = element['image/object/bbox/ymax'].values.numpy()
        widths = x_maxes - x_mins
        heights = y_maxes - y_mins
        x_centers = x_mins + widths / 2
        y_centers = y_mins + heights / 2
        grid_x = tf.math.floor(x_centers * 12)
        grid_y = tf.math.floor(y_centers * 12)
        for i in range(grid_x.shape[0]):
            label_tensor[int(grid_y[i] * GRID_CELLS + grid_x[i])].assign(
                [x_centers[i], y_centers[i], widths[i], heights[i], 1])
        images.append(image_tensor)
        labels.append(label_tensor)
    return images, labels


images, labels = format_dataset(parsed_image_dataset)

fh = open("imageToSave.jpeg", "wb")
fh.write(encode_img(images[0]).numpy())
fh.close()
pic = Image.open('imageToSave.jpeg')
draw = ImageDraw.Draw(pic)
example_label = labels[0]
label = example_label[..., 4] > 0.3
for i in range(label.shape[0]):
    if label[i]:
        x_center = example_label[i][0]
        y_center = example_label[i][1]
        width = example_label[i][2]
        height = example_label[i][3]
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_min + width
        y_max = y_min + height
        draw.rectangle([(x_min * IMG_WIDTH, y_min * IMG_HEIGHT), (x_max * IMG_WIDTH, y_max * IMG_HEIGHT)],
                       outline=0xff0000,
                       width=3, fill=None)
pic.show()


# I'm pretty sure this is the dataset now
# train_ds=prepare_for_training(labeled_ds)

# image_batch, label_batch=next(iter(train_ds))

# print(label_batch.numpy())

# show_batch(image_batch.numpy(), label_batch.numpy())

def calc_IOU(truth, pred):
    overlap_w = (truth.w + pred.w) / 2 - abs(truth.x - pred.x)
    overlap_h = (truth.h + pred.h) / 2 - abs(truth.y - pred.y)
    overlap_a = overlap_w * overlap_h
    truth_a = truth.w * truth.h
    pred_a = pred.w * pred.h
    union_a = truth_a + pred_a - overlap_a
    return overlap_a / union_a
