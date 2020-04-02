"""
Python package to conduct EDA of test data
"""

import pandas as pd
import data_utils as utils
import gc
import plotly
import plotly.graph_objs as go

counts_key_classes = dict()

classes = list()
counts_by_classes = list()

test_meta_df = pd.read_csv(utils.train_meta_filepath, usecols=['target'])

groups_meta = test_meta_df['target'].value_counts()
groups_meta = groups_meta.sort_index()

for i in groups_meta.index:
    classes.append(i)
    counts_by_classes.append(groups_meta[i])

classes.append(99)
counts_by_classes.append(0)

classes_counts_df = pd.DataFrame()
classes_counts_df['target'] = classes
classes_counts_df['num_objects'] = counts_by_classes

classes_counts_df.to_csv('training_target_objects_count.csv', index=False)


data = go.Bar(x=['_{}_'.format(target) for target in classes], y=counts_by_classes, text=counts_by_classes,
              textposition='outside')

layout = go.Layout(
    title=go.layout.Title(
        text='Number of Astronomical Objects per Object Type',
        xref='paper',
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='Object Types',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='Number of Astronomical Objects',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
)

plotly.offline.plot({
                "data": [data],
                "layout": layout
            }, auto_open=False, filename='training_target_objects_count.html')

del counts_key_classes, counts_by_classes, classes
gc.collect()


objects = list()
num_time_steps = list()

chunk_test_data = pd.read_csv(utils.train_filepath, usecols=['object_id'])
groups_ts = chunk_test_data['object_id'].value_counts()

groups_ts = groups_ts.sort_index()

for key in groups_ts.index:
    objects.append(key)
    num_time_steps.append(groups_ts[key])

detections_counts_df = pd.DataFrame()
detections_counts_df['object_id'] = objects
detections_counts_df['num_detections'] = num_time_steps

detections_counts_df.to_csv('training_object_detections_count.csv', index=False)

data = go.Bar(x=['_{}_'.format(target) for target in objects], y=num_time_steps, marker_color='black')

layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)',
    title=go.layout.Title(
        text='Number of Detections per Object',
        xref='paper',
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='Object ID',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='Number of Detections',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
)

plotly.offline.plot({
                "data": [data],
                "layout": layout
            }, auto_open=False, filename='training_object_detections_count.html')
