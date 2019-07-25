import os
import numpy as np
import random

from utils import sparse_tuple_from, resize_image, label_to_array
from PIL import Image

class DataLoader(object):
    def __init__(self, examples_path, batch_size, max_image_width, max_char_count):
        self.examples_path = examples_path
        self.batch_size = batch_size
        self.max_image_width = max_image_width
        self.max_char_count = max_char_count

    def __iter__(self):
        examples = []
        for f in os.listdir(self.examples_path):
            label, _ = f.split('_')
            if len(f.split('_')[0]) > self.max_char_count:
                continue
            arr, _ = resize_image(
                os.path.join(self.examples_path, f),
                self.max_image_width
            )
            # to lower
            label_lower = label.lower()
            examples.append(
                (
                    arr,
                    label_lower,
                    label_to_array(label_lower)
                )
            )

            if len(examples) == self.batch_size:
                raw_batch_x, raw_batch_y, raw_batch_la = zip(*examples)
                batch_y = np.reshape(
                    np.array(raw_batch_y),
                    (-1)
                )

                batch_dt = sparse_tuple_from(
                    np.array(raw_batch_la)
                )

                raw_batch_x = np.swapaxes(raw_batch_x, 1, 2)

                batch_x = np.reshape(
                    np.array(raw_batch_x),
                    (len(raw_batch_x), self.max_image_width, 32, 1)
                )
                yield (batch_y, batch_dt, batch_x)
                examples = []