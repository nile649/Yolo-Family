import math
import torch
from tfrecord_utils import parse_tfrecord_fn

def get_dataset(files, batch_size=128, repeat=False, cache=False, train=False):
    AUTO = tf.data.experimental.AUTOTUNE
    ds = tf.data.TFRecordDataset(filenames=files,num_parallel_reads=AUTOTUNE)
    if cache:
        ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    ds = ds.map(lambda x: parse_tfrecord_fn(x, (224,224)), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return tfds.as_numpy(ds)

class TFRecordDataLoader:
    def __init__(self, files, batch_size=128, cache=False, train=True, repeat=False):
        self.ds = get_dataset(
            files, 
            batch_size=batch_size,
            cache=cache,
            repeat=repeat,train=train)  # Use the `train` parameter here
        
        self.num_examples = count_data_items(files)

        self.batch_size = batch_size
        self._iterator = None
#     returns the self._iterator which will iterate over the numpy data. Not a recommendation    
#     def __iter__(self):
#         if self._iterator is None:
#             self._iterator = iter(self.ds)
#         else:
#             self._reset()
#         return self._iterator
    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self


    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        images = torch.tensor(batch["image"], dtype=torch.float32).permute(0, 3, 1, 2)
        bboxes = torch.tensor(batch["bbox"], dtype=torch.float32)
        labels = torch.tensor(batch["category_id"], dtype=torch.int64)
        return images, bboxes, labels

    def __len__(self):
        return math.ceil(self.num_examples / self.batch_size)  # Use integer division and math.ceil() here
