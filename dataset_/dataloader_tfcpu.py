'''
code provides a method to take a existing tfrecord dataset created using Data2TFrecord.py for object detection.
Decodes the binary data for faster processing.
'''


from tfrecord_utils import TFRecordDataLoader

def dataloader_tfcpu(files,batch_size,cache=False, train=True, repeat=False):
    return TFRecordDataLoader(files,batch_size,cache=False, train=True, repeat=False)