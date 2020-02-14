import tensorflow as tf

def build(img_w, img_h, grid_w, grid_h, nb_boxes, nb_classes):
    yolov3 = tf.keras.Sequential()
    yolov3.add(tf.keras.layers.Input(shape=(img_w,img_h,3)))
    yolov3.add(tf.keras.layers.Conv2D(16, (1, 1)))
    yolov3.add(tf.keras.layers.Conv2D(32, (3, 3)))
    yolov3.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    yolov3.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    yolov3.add(tf.keras.layers.Conv2D(16, (3, 3)))
    yolov3.add(tf.keras.layers.Conv2D(32, (3, 3)))
    yolov3.add(tf.keras.layers.LeakyReLU(alpha=0.3))
    yolov3.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    yolov3.add(tf.keras.layers.Flatten())
    yolov3.add(tf.keras.layers.Dense(256, activation='sigmoid'))
    yolov3.add(tf.keras.layers.Dense(grid_w*grid_h*(nb_boxes*5 + nb_classes), activation='sigmoid'))
    yolov3.add(tf.keras.layers.Reshape((grid_w*grid_h,(nb_boxes*5 + nb_classes))))
def calc_IOU(truth, pred):
    overlap_w = (truth.w + pred.w) / 2 - abs(truth.x - pred.x)
    overlap_h = (truth.h + pred.h) / 2 - abs(truth.y - pred.y)
    overlap_a = overlap_w * overlap_h
    truth_a = truth.w * truth.h
    pred_a = pred.w * pred.h
    union_a = truth_a + pred_a - overlap_a
    return overlap_a / union_a
