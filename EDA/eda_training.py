"""
Python package to conduct EDA of training data
"""

import pandas as pd
import plain_rnn_utils as utils
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import numpy as np

train_meta_df = pd.read_csv(utils.train_meta_filepath)
train_ts_df = pd.read_csv(utils.train_filepath)

groups_meta = train_meta_df.groupby('target')
groups_ts = train_ts_df.groupby('object_id')


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
plt.savefig('training_class_dist.png')


objects = list()
num_time_steps = list()

for g in groups_ts:
    objects.append(int(g[0]))
    num_time_steps.append(g[1].shape[0])

y_pos = np.arange(len(objects))

ymax = max(num_time_steps)*1.1

# Set the y limits making the maximum 5% greater


plt.bar(y_pos, num_time_steps, align='center', alpha=0.5)
# plt.xticks(y_pos, classes)
plt.ylabel('Number of detections per astronomical source')
plt.ylim(0, ymax)
plt.xlabel('Astronomical source id')
plt.title('Number of detections vs Astronomical Source')

# plt.show()
plt.savefig('training_detections_count.png')
