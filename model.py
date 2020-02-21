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
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape
from keras.layers import add, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import tensorflow.keras

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMG_HEIGHT = 416
IMG_WIDTH = 416
GRID_CELLS = 12
N_BOXES = 1
N_CLASSES = 0
#tf.config.experimental_run_functions_eagerly(True)


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
        x_centers = ((img_x_centers % (1 / GRID_CELLS)) * GRID_CELLS) + grid_x
        y_centers = ((img_y_centers % (1 / GRID_CELLS)) * GRID_CELLS) + grid_y
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


# def build(img_w, img_h, grid_w, grid_h, n_boxes, n_classes):
#     #print("build called")
#     inputs = tf.keras.Input(shape=(img_w, img_h, 3))
#     x = layers.Conv2D(16, (1, 1))(inputs)
#     x = layers.Conv2D(32, (3, 3))(x)
#     x = layers.LeakyReLU(alpha=0.3)(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(16, (3, 3))(x)
#     x = layers.Conv2D(32, (3, 3))(x)
#     x = layers.LeakyReLU(alpha=0.3)(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(16, (3, 3))(x)
#     x = layers.Conv2D(32, (3, 3))(x)
#     x = layers.LeakyReLU(alpha=0.3)(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Conv2D(16, (3, 3))(x)
#     x = layers.Conv2D(32, (3, 3))(x)
#     x = layers.LeakyReLU(alpha=0.3)(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(256, activation='sigmoid')(x)
#     x = layers.Dense(
#         grid_w * grid_h * (n_boxes * 5 + n_classes), activation='sigmoid')(x)
#     outputs = layers.Reshape(
#         (grid_w * grid_h, (n_boxes * 5 + n_classes)))(x)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name='YoloV3')
#     #print("build completed")
#     return model
    
def conv2d_unit(x, filters, kernels, strides=1):
    """Convolution Unit
    This function defines a 2D convolution operation with BN and LeakyReLU.
    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and
            height. Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
            Output tensor.
    """
    x = Conv2D(filters, kernels,
               padding='same',
               strides=strides,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def residual_block(inputs, filters):
    """Residual Block
    This function defines a 2D convolution operation with BN and LeakyReLU.
    # Arguments
        x: Tensor, input tensor of residual block.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
    # Returns
        Output tensor.
    """
    x = conv2d_unit(inputs, filters, (1, 1))
    x = conv2d_unit(x, 2 * filters, (3, 3))
    x = add([inputs, x])
    x = Activation('linear')(x)

    return x


def stack_residual_block(inputs, filters, n):
    """Stacked residual Block
    """
    x = residual_block(inputs, filters)

    for i in range(n - 1):
        x = residual_block(x, filters)

    return x


def darknet_base(inputs):
    """Darknet-53 base model.
    """

    x = conv2d_unit(inputs, 32, (3, 3))

    x = conv2d_unit(x, 64, (3, 3), strides=2)
    x = stack_residual_block(x, 32, n=1)

    x = conv2d_unit(x, 128, (3, 3), strides=2)
    x = stack_residual_block(x, 64, n=2)

    x = conv2d_unit(x, 256, (3, 3), strides=2)
    x = stack_residual_block(x, 128, n=8)

    x = conv2d_unit(x, 512, (3, 3), strides=2)
    x = stack_residual_block(x, 256, n=8)

    x = conv2d_unit(x, 1024, (3, 3), strides=2)
    x = stack_residual_block(x, 512, n=4)

    return x


def build(img_w, img_h, grid_w, grid_h, n_boxes, n_classes):
    """Darknet-53 classifier.
    """
    inputs = tf.keras.Input(shape=(img_w, img_h, 3))
    x = darknet_base(inputs)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='sigmoid')(x)
    x = tensorflow.keras.layers.Dense(grid_w * grid_h * (n_boxes * 5 + n_classes), activation='sigmoid')(x)
    outputs = tensorflow.keras.layers.Reshape( (grid_w * grid_h, (n_boxes * 5 + n_classes)))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='YoloV3')

    return model


grid_loss = tf.Variable([[float(x), float(y)] for y in range(GRID_CELLS) for x in range(GRID_CELLS)])


def calc_loss(true, pred):
    #print(true.shape)
    #print(pred.shape)
    #print(grid_loss.shape)
    true_xy = true[..., :2]
    #print(true_xy.shape)
    pred_xy = pred[..., :2] + grid_loss
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
    return k.sum(k.square(true_xy - pred_xy),axis=-1)*true_conf


def calc_wh_loss(true_wh, pred_wh, true_conf):
    return k.sum(k.square(k.sqrt(true_wh) - k.sqrt(pred_wh)),axis=-1)*true_conf


def calc_IOU(true_xy, pred_xy, true_wh, pred_wh):
    intersect_wh = k.maximum(k.zeros_like(pred_wh), (pred_wh + true_wh) / 2 - k.square(pred_xy - true_xy))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    union_area = pred_area + true_area - intersect_area
    return intersect_area / union_area


def calc_conf_loss(true_conf, pred_conf, iou):
    obj_conf_loss = (true_conf*iou - pred_conf) + true_conf * 3.0
    noobj_conf_loss = (true_conf * iou - pred_conf) + (1 - true_conf) * 0.05
    conf_loss = k.square(obj_conf_loss + noobj_conf_loss)
    #print(conf_loss.shape)
    return conf_loss


# Format dataset
(images, labels) = format_dataset(parsed_image_dataset)


print("calling build")
model = build(IMG_WIDTH, IMG_HEIGHT, GRID_CELLS, GRID_CELLS, N_BOXES, N_CLASSES)
#model = build()
print("Build should be completed")
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
print("compiling")
model.compile(loss=calc_loss, optimizer=adam)
print("Done compiling")


def test(model, image_n):
    # TESTS
    fh = open("imageToSave.jpeg", "wb")
    # Get example image from images
    fh.write(encode_img(images[image_n]).numpy())
    fh.close()
    pic = Image.open('imageToSave.jpeg')
    draw = ImageDraw.Draw(pic)
    # Get example label from labels
    layer_outputs = model.layers[-1].output
    print(layer_outputs)
    # Extracts the outputs of the top 12 layers
    #activation_model = keras.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    example_label = model.predict(tf.convert_to_tensor([images[image_n]], dtype=tf.float32))[0]
    #print (example_label)
    #print(example_label.shape)
    for i in range(144):
        if example_label[i][4] > 0.15:
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
                            width=int(example_label[i][4] * 3), fill=None)
    pic.show()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', help='train', action='store_true')
parser.add_argument('--epoch', help='epoch', const='int', nargs='?', default=1)
parser.add_argument('--batch', help='batch size', const='int', nargs='?', default=4)
args = parser.parse_args()

if args.train:
    # print("Before fit")
    # images = []
    # for i in range(100):
    #     t = tf.random.uniform(shape = [416, 416, 3], dtype = tf.float32)
    #     images.append(t)
    # images = tf.convert_to_tensor(images, dtype=tf.float32)
    # print('images made')
       
    # labels = []
    # for i in range(100):
    #     t = tf.random.uniform(shape = [144, 5], dtype = tf.float32)
    #     labels.append(t)
    # labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    # print('labels made')

    # print(images.shape)
    print("fitting")
    model.fit(images, labels, steps_per_epoch=105, epochs=int(args.epoch))
    model.save_weights('weights_006.h5')
    test(model, 0)
    test(model, 1)
    test(model, 2)
    test(model, 3)
else:
    model.load_weights('weights_006.h5')
    test(model, 0)
    test(model, 1)
    test(model, 2)
    test(model, 107)
