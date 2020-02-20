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
import argparse

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 416
IMG_WIDTH = 416
GRID_CELLS = 12
N_BOXES = 1
N_CLASSES = 0


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
    img = tf.image.resize(tensor, [240, 320])
    # Use `convert_image_dtype` to convert to uint8.
    img = tf.image.convert_image_dtype(img, tf.uint8)
    # convert the 3D uint8 tensor to a compressed string
    return tf.image.encode_jpeg(img)


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def format_dataset(dataset):
    #print("format called")
    images = []
    labels = []
    for element in dataset:
        # Map elements through function to turn them into tuples with format (image, label)
        filename = element['image/filename'].numpy().decode('utf-8')
        image_tensor = process_path('./images/power_cell/' + filename)
        # Map labels to format [gridPos, [x, y, w, h, conf]]
        # Default label should be [0, 0, 0, 0, 0]
        label_tensor = k.zeros([GRID_CELLS ** 2, N_BOXES * 5 + N_CLASSES])
        # Get data from element
        x_mins = element['image/object/bbox/xmin'].values.numpy()
        x_maxes = element['image/object/bbox/xmax'].values.numpy()
        y_mins = element['image/object/bbox/ymin'].values.numpy()
        y_maxes = element['image/object/bbox/ymax'].values.numpy()
        # width = max - min
        widths = x_maxes - x_mins
        heights = y_maxes - y_mins
        # center = avg(max, min)
        img_x_centers = (x_mins + widths / 2)
        img_y_centers = (y_mins + heights / 2)
        # grid_xy = floor(center * grid_cells)
        grid_x = tf.math.floor(img_x_centers * GRID_CELLS)
        grid_y = tf.math.floor(img_y_centers * GRID_CELLS)
        # Map centers relative to grid cell by removing reference to img position and scaling to grid size
        x_centers = (img_x_centers % (1 / GRID_CELLS)) * GRID_CELLS
        y_centers = (img_y_centers % (1 / GRID_CELLS)) * GRID_CELLS
        # Update label tensor to have new data from bboxes
        for i in range(grid_x.shape[0]):
            label_tensor[int(grid_y[i] * GRID_CELLS + grid_x[i])].assign([x_centers[i], y_centers[i], widths[i], heights[i], 1])
        # Add tensor to image/label list
        images.append(image_tensor)
        labels.append(label_tensor)
    #print("Format completed")
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    #print("Images saved")
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    #print("vars saved")
    return (images, labels)


def build(img_w, img_h, grid_w, grid_h, n_boxes, n_classes):
    #print("build called")
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
    #print("build completed")
    return model


def calc_loss(true, pred):
    #print("calc loss called")
    true_xy = true[..., :2]
    #print(true_xy.shape)
    pred_xy = pred[..., :2]
    #print(pred_xy.shape)
    true_wh = true[..., 2:4]
    pred_wh = pred[..., 2:4]
    true_conf = true[..., 4]
    pred_conf = pred[..., 4]
    xy_loss = calc_xy_loss(true_xy, pred_xy, true_conf)
    wh_loss = calc_wh_loss(true_wh, pred_wh, true_conf)
    conf_loss = calc_conf_loss(true_conf, pred_conf, calc_IOU(true_xy, pred_xy, true_wh, pred_wh))
    #print("calc loss completed")
    return 5 * xy_loss + 5 * wh_loss + conf_loss


def calc_xy_loss(true_xy, pred_xy, true_conf):
    #return k.sum(k.square(true_xy - pred_xy) * true_conf, axis=-1)
    return k.sum(k.sum(k.square(true_xy - pred_xy),axis=-1)*true_conf, axis=-1)


def calc_wh_loss(true_wh, pred_wh, true_conf):
    #return k.sum(k.square(true_wh - pred_wh) * true_conf, axis=-1)
    return k.sum(k.sum(k.square(true_wh - pred_wh),axis=-1)*true_conf, axis=-1)


def calc_IOU(true_xy, pred_xy, true_wh, pred_wh):
    intersect_wh = k.maximum(k.zeros_like(pred_wh), (pred_wh + true_wh) / 2 - k.square(pred_xy - true_xy))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    union_area = pred_area + true_area - intersect_area
    return intersect_area / union_area

#@tf.function
def calc_conf_loss(true_conf, pred_conf, iou):
    c = tf.map_fn(lambda x: tf.cond(x, lambda: 1.0, lambda: 0.01), tf.math.equal(true_conf, tf.ones_like(true_conf)), dtype=tf.float32)
    print("C made")
    conf_loss = k.sum(tf.multiply(k.square(true_conf*iou - pred_conf), c), axis=-1)
    print("conf_loss made")
    return conf_loss


# Format dataset
(images, labels) = format_dataset(parsed_image_dataset)

#print("calling build")
model = build(IMG_WIDTH, IMG_HEIGHT, GRID_CELLS, GRID_CELLS, N_BOXES, N_CLASSES)
#print("Build should be completed")
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
#print("compiling")
model.compile(loss=calc_loss, optimizer=adam)
#print("Done compiling")


def test():
    # TESTS
    fh = open("imageToSave.jpeg", "wb")
    # Get example image from images
    fh.write(encode_img(images[0]).numpy())
    fh.close()
    pic = Image.open('imageToSave.jpeg')
    draw = ImageDraw.Draw(pic)
    # Get example label from labels
    example_label = model.predict(images[:1])[0]
    label = example_label[..., 4] > 0.3
    for i in range(label.shape[0]):
        if label[i]:
            print(example_label[i])
            # Convert tensor type features back into image type features
            grid_x = i % GRID_CELLS
            grid_y = (i - grid_x) / GRID_CELLS
            x_center = grid_x / GRID_CELLS + example_label[i][0] / GRID_CELLS
            y_center = grid_y / GRID_CELLS + example_label[i][1] / GRID_CELLS
            width = example_label[i][2]
            height = example_label[i][3]
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_min + width
            y_max = y_min + height
            # Draw bboxes with image type features
            draw.rectangle([(x_min * 320, y_min * 240), (x_max * 320, y_max * 240)],
                           outline=0xff0000,
                           width=3, fill=None)
    pic.show()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', help='train', action='store_true')
parser.add_argument('--epoch', help='epoch', const='int', nargs='?', default=1)
args = parser.parse_args()

if args.train:
    #print("Before fit")
    model.fit(images, labels, batch_size=32, epochs=int(args.epoch))
    model.save_weights('weights_006.h5')
    test()
else:
    model.load_weights('weights_006.h5')
    test()
