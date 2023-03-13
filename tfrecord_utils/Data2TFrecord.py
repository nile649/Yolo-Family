import os, sys, gc, argparse, numpy as np
import random
from tfrecord_utils import create_example
import tensorflow as tf


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default = "",required=True,help="path to file which has gt and lbl")
    parser.add_argument("--data_name", default = "dataname",required=True)
    parser.add_argument("--num_samples", type=int, default = 4096)
    opt = parser.parse_args()
    return opt

    
class Data2TFrecord():
    def __init__(self,path,data_name,tfrecords_dir='./tfrecord_dir/',augmentation=None,num_samples=4096) -> None:
        self.num_samples = num_samples # number of samples per tfrecord
        if not os.path.exists(tfrecords_dir):
            os.makedirs(tfrecords_dir)  # creating TFRecords output folder
        self.path = path 
        self.dir = tfrecords_dir
        self.augmentation = augmentation
        self.data_name = data_name

    def chunk(self,img_files,label_files,image_type="jpeg"):
        a_tuple = (len(img_files)!=len(label_files),"number doesn't match")
        assert(a_tuple)
        sz = len(img_files)
        num_tfrecords = sz//4096
        # print(num_tfrecords)
        for tfrec_num in range(num_tfrecords):
            image_bytes = [self.readIMG2Bytes(f) for f in img_files[tfrec_num*4096:(tfrec_num+1)*4096]]
            bbox_list = [self.readBBox2List(f) for f in label_files[tfrec_num*4096:(tfrec_num+1)*4096]]
            print(len(image_bytes),len(bbox_list))
            with tf.io.TFRecordWriter(
                self.dir + "/%s_file_%.2i-%i.tfrec" % (self.data_name,tfrec_num, len(image_bytes))
            ) as writer:
                for img_byte,bbox_lis,image_path in zip(image_bytes,bbox_list,img_files):
                    example = create_example(img_byte, image_path, bbox_lis)
                    writer.write(example.SerializeToString())
        image_bytes = [self.readIMG2Bytes(f) for f in img_files[num_tfrecords*4096:]]
        bbox_list = [self.readBBox2List(f) for f in label_files[num_tfrecords*4096:]]
        print(len(image_bytes),len(bbox_list))
        with tf.io.TFRecordWriter(
            self.dir + "/%s_file_%.2i-%i.tfrec" % (self.data_name,num_tfrecords+1, len(image_bytes))
        ) as writer:
            for img_byte,bbox_lis,image_path in zip(image_bytes,bbox_list,img_files):
                example = create_example(img_byte, image_path, bbox_lis)
                writer.write(example.SerializeToString())

    def readIMG2Bytes(self,img_file,ftype="png"):
        if ftype=="jpeg":
            return self.readJPEG2Bytes(img_file)
        else:
            return self.readPNG2Bytes(img_file)

    def readJPEG2Bytes(self,img_file):
        image = tf.io.decode_jpeg(tf.io.read_file(img_file))
        return [image,image.shape]
    def readPNG2Bytes(self,img_file):
        image = tf.io.decode_png(tf.io.read_file(img_file))
        return [image, image.shape]

    def readBBox2List(self,label_file):
        bboxes = np.roll(np.loadtxt(fname=label_file, delimiter=" ", ndmin=2), 4, axis=1)#.tolist()
        # print(bboxes)
        if(len(bboxes)==0):
            return None
        class_id = [cls[-1] for cls in bboxes]
        return [bboxes,class_id]

    def function_Data2TFrecord(self):
        """
        path : path to text file which has image and bbox path
        
        """
        img_files = []
        label_files = []
        with open(self.path, "r") as file:
            img_files = [p.rstrip() for p in file.read().splitlines()]
            random.shuffle(img_files)
            img_files = img_files
        label_files = []
        for path in img_files:
            path = os.path.splitext(path)[0] + '.txt'
            path = path.replace('images','labels')
            label_files.append(path)  
        img_files = list(
            filter(lambda item: item is not None, img_files))
        label_files = list(
            filter(lambda item: item is not None, label_files))
        # print(len(img_files),len(label_files))
        self.chunk(img_files,label_files)

    def __call__(self) -> None:
        self.function_Data2TFrecord()

def main():
    opt = get_opt()
    # print(opt.path)
    _ = Data2TFrecord(opt.path,opt.data_name,num_samples=opt.num_samples)
    _()


if __name__ == "__main__":
    main()