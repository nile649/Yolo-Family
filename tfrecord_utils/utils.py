import tensorflow as tf

def image_feature(value,ftype="png"):
    """Returns a bytes_list from a string / byte."""
    if ftype=="jpeg":
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def float_feature_list_list(value):
    # value must be a numpy array.
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def create_example(image, path, bbox):
    """
        Example to create dataset from COCO
    
    """
    feature = {
        "image": image_feature(image[0]),
        "path": bytes_feature(path),
        "bbox": float_feature_list_list(bbox[0]),
        "category_id": float_feature_list(bbox[1]),
        "height": int64_feature(image[1][0]),
        "width": int64_feature(image[1][1]),
        "channel": int64_feature(image[1][2]),
    }
    temp = tf.train.Features(feature=feature)
    lol = tf.train.Example(features=temp)
    return lol


def parse_tfrecord_fn(example,image_size=(224,224)):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.VarLenFeature(tf.float32),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channel": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    img = tf.io.decode_png(example["image"], channels=4)
    image = tf.image.resize(img, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    example["image"] = image
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    example["category_id"] = tf.sparse.to_dense(example["category_id"])
    
    return example


def xywh2xyxy(x):
    if(len(x)==5):
        x,y,w,h,c = x[0],x[1],x[2],x[3],x[4]
    else:
        x,y,w,h,c = x[0],x[1],x[2],x[3],-1        
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return (x1,y1,x2,y2,c)