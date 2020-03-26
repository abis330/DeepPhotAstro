"""
Python package to conduct EDA of test data
"""

import pandas as pd
import plain_rnn_utils as utils
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import numpy as np


object_detection_counts = dict()

for i in range(1, 12):

    for chunk_test_data in pd.read_csv('test_set_batch{}.csv.zip'.format(i), chunksize=utils.chunksize,
                                       compression='zip'):
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
