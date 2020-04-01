"""
Python package to conduct EDA of test data
"""

import pandas as pd
import plain_rnn_utils as utils
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import numpy as np


object_detection_counts = dict()

test_meta_df = pd.read_csv(utils.test_meta_filepath)

groups_meta = test_meta_df.groupby('true_target')

classes = list()
freqs = list()

for g in groups_meta:

    classes.append(int(g[0]))
    freqs.append(g[1].shape[0])

classes.append(99)
freqs.append(0)

y_pos = np.arange(len(classes))

plt.bar(y_pos, freqs, align='center', alpha=0.5)
plt.xticks(y_pos, classes)
plt.ylabel('Number of astronomical sources')
plt.xlabel('Class of astronomical source')
plt.title('Training Data Class Distribution')

# plt.show()
plt.savefig('test_class_dist.png')

# for i in range(1, 12):

for chunk_test_data in pd.read_csv(utils.test_data_filepath, chunksize=utils.chunksize, compression='zip'):
    groups_ts = chunk_test_data.groupby('object_id')
    for g in groups_ts:
        key = int(g[0])
        if object_detection_counts.get(key, None) is None:
            object_detection_counts[key] = g[1].shape[0]
        else:
            object_detection_counts[key] = object_detection_counts[key] + g[1].shape[0]


objects = list()
num_time_steps = list()

for i in sorted(object_detection_counts):
    objects.append(i)
    num_time_steps.append(object_detection_counts[i])

y_pos = np.arange(len(objects))

ymax = max(num_time_steps)*1.1

plt.bar(y_pos, num_time_steps, align='center', alpha=0.5)
# plt.xticks(y_pos, classes)
plt.ylabel('Number of detections per astronomical source')
plt.ylim(0, ymax)
plt.xlabel('Astronomical source id')
plt.title('Number of detections vs Astronomical Source')

# plt.show()
plt.savefig('test_detections_count.png')
