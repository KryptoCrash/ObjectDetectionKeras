import tensorflow as tf
import tensorflow.keras.layers as layers


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



def calc_IOU(truth, pred):
    overlap_w = (truth.w + pred.w) / 2 - abs(truth.x - pred.x)
    overlap_h = (truth.h + pred.h) / 2 - abs(truth.y - pred.y)
    overlap_a = overlap_w * overlap_h
    truth_a = truth.w * truth.h
    pred_a = pred.w * pred.h
    union_a = truth_a + pred_a - overlap_a
    return overlap_a / union_a
