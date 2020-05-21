"""
Utility to pre-process data in training_set.csv after which now there are values of all six passbands (zero value if
there was no flux value for that passband) on given day. Now, time granularity is on day basis after pre-processing

Also, this is used to plot few of light curves.
"""

import argparse
import numpy as np
import pandas as pd
from itertools import groupby
import matplotlib
import matplotlib.pyplot as plt
import data_utils as utils

matplotlib.use('agg')


def index_min(values):
    return min(range(len(values)), key=values.__getitem__)


def time_collector(arr, frac=1):  # makes values on same day to be together
    bestclustering = True
    while bestclustering:
        a = []
        for key, group in groupby(arr, key=lambda n: n // (1. / frac)):
            s = sorted(group)
            a.append(np.sum(s) / len(s))
        ind = []
        i = 0
        for key, group in groupby(arr, key=lambda n: n // (1. / frac)):
            ind.append([])
            for j in group:
                ind[i].append(index_min(abs(j - np.array(arr))))
            i += 1
        if len([len(i) for i in ind if len(i) > 6]) != 0:
            frac += 0.1
        else:
            bestclustering = False
    return a, ind, frac


def create_colourband_array(ind, arr, err_arr, temp_arr, err_temp_arr):
    temp = [arr[ind[i]] for i in range(len(ind)) if arr[ind[i]] != 0]
    err_temp = [err_arr[ind[i]] for i in range(len(ind)) if err_arr[ind[i]] != 0]
    if len(temp) == 0:
        temp_arr.append(0)
        err_temp_arr.append(0)
        out = True
    elif len(temp) > 1:
        out = False
    else:
        temp_arr.append(temp[0])
        err_temp_arr.append(err_temp[0])
        out = True
    return temp_arr, err_temp_arr, out


def preprocess(filename, grouping=1):
    lightcurves_df = pd.read_csv(filename)

    # group by the object id
    lightcurves = lightcurves_df.groupby('object_id')
    num_lightcurves = 0
    object_id_arr = []
    all_t = []
    all_u_temp_arr = []
    all_u_err_temp_arr = []
    all_g_temp_arr = []
    all_g_err_temp_arr = []
    all_r_temp_arr = []
    all_r_err_temp_arr = []
    all_i_temp_arr = []
    all_i_err_temp_arr = []
    all_z_temp_arr = []
    all_z_err_temp_arr = []
    all_Y_temp_arr = []
    all_Y_err_temp_arr = []
    for lightcurve in lightcurves:
        num_lightcurves += 1
        obs = []
        first_obs = None
        for idx, row in lightcurve[1].iterrows():
            u = g = r = i = z = Y = 0
            u_error = g_error = r_error = i_error = z_error = Y_error = 0
            if first_obs is None:
                first_obs = float(row['mjd'])
            if row['passband'] == 0:  # u
                u = float(row['flux'])  # flux_val
                u_error = float(row['flux_err'])  # flux_err
            elif row['passband'] == 1:  # g
                g = float(row['flux'])  # flux_val
                g_error = float(row['flux_err'])  # flux_err
            elif row['passband'] == 2:  # r
                r = float(row['flux'])  # flux_val
                r_error = float(row['flux_err'])  # flux_err
            elif row['passband'] == 3:  # i
                i = float(row['flux'])  # flux_val
                i_error = float(row['flux_err'])  # flux_err
            elif row['passband'] == 4:  # z
                z = float(row['flux'])  # flux_val
                z_error = float(row['flux_err'])  # flux_err
            elif row['passband'] == 5:  # Y
                Y = float(row['flux'])  # flux_val
                Y_error = float(row['flux_err'])  # flux_err
            else:
                raise ValueError('Invalid passband value!')
            obs.append(
                [float(row['mjd'])] + [u, g, r, i, z, Y] + [u_error, g_error, r_error, i_error, z_error, Y_error])

        t_arr = [obs[i][0] for i in range(len(obs))]  # time values in lightcurve
        u_arr = [obs[i][1] for i in range(len(obs))]  # g flux values in lightcurve at each time point
        u_err_arr = [obs[i][7] for i in range(len(obs))]
        g_arr = [obs[i][2] for i in range(len(obs))]  # g flux values in lightcurve at each time point
        g_err_arr = [obs[i][8] for i in range(len(obs))]  # # g flux_err values in lightcurve at each time point
        r_arr = [obs[i][3] for i in range(len(obs))]
        r_err_arr = [obs[i][9] for i in range(len(obs))]
        i_arr = [obs[i][4] for i in range(len(obs))]
        i_err_arr = [obs[i][10] for i in range(len(obs))]
        z_arr = [obs[i][5] for i in range(len(obs))]
        z_err_arr = [obs[i][11] for i in range(len(obs))]
        Y_arr = [obs[i][6] for i in range(len(obs))]
        Y_err_arr = [obs[i][12] for i in range(len(obs))]
        correctplacement = True
        frac = grouping
        t = []
        while correctplacement:
            t, index, frac = time_collector(t_arr, frac)
            u_temp_arr = []
            u_err_temp_arr = []
            g_temp_arr = []
            g_err_temp_arr = []
            r_temp_arr = []
            r_err_temp_arr = []
            i_temp_arr = []
            i_err_temp_arr = []
            z_temp_arr = []
            z_err_temp_arr = []
            Y_temp_arr = []
            Y_err_temp_arr = []
            tot = []
            for i in range(len(index)):
                u_temp_arr, u_err_temp_arr, ufail = create_colourband_array(index[i], u_arr, u_err_arr, u_temp_arr,
                                                                            u_err_temp_arr)
                g_temp_arr, g_err_temp_arr, gfail = create_colourband_array(index[i], g_arr, g_err_arr, g_temp_arr,
                                                                            g_err_temp_arr)
                r_temp_arr, r_err_temp_arr, rfail = create_colourband_array(index[i], r_arr, r_err_arr, r_temp_arr,
                                                                            r_err_temp_arr)
                i_temp_arr, i_err_temp_arr, ifail = create_colourband_array(index[i], i_arr, i_err_arr, i_temp_arr,
                                                                            i_err_temp_arr)
                z_temp_arr, z_err_temp_arr, zfail = create_colourband_array(index[i], z_arr, z_err_arr, z_temp_arr,
                                                                            z_err_temp_arr)
                Y_temp_arr, Y_err_temp_arr, Yfail = create_colourband_array(index[i], Y_arr, Y_err_arr, Y_temp_arr,
                                                                            Y_err_temp_arr)
                tot.append(ufail * gfail * rfail * ifail * zfail * Yfail)
            if all(tot):
                correctplacement = False
            else:
                frac += 0.1
        print('{} Lightcurve for Object ID {} done'.format(num_lightcurves, lightcurve[0]))
        all_t = all_t + t
        all_u_temp_arr = all_u_temp_arr + u_temp_arr
        all_u_err_temp_arr = all_u_err_temp_arr + u_err_temp_arr
        all_g_temp_arr = all_g_temp_arr + g_temp_arr
        all_g_err_temp_arr = all_g_err_temp_arr + g_err_temp_arr
        all_r_temp_arr = all_r_temp_arr + r_temp_arr
        all_r_err_temp_arr = all_r_err_temp_arr + r_err_temp_arr
        all_i_temp_arr = all_i_temp_arr + i_temp_arr
        all_i_err_temp_arr = all_i_err_temp_arr + i_err_temp_arr
        all_z_temp_arr = all_z_temp_arr + z_temp_arr
        all_z_err_temp_arr = all_z_err_temp_arr + z_err_temp_arr
        all_Y_temp_arr = all_Y_temp_arr + Y_temp_arr
        all_Y_err_temp_arr = all_Y_err_temp_arr + Y_err_temp_arr

        object_id_arr = object_id_arr + [lightcurve[0]] * len(t)

    new_lightcurves_df = pd.DataFrame()
    new_lightcurves_df['object_id'] = object_id_arr
    new_lightcurves_df['mjd'] = all_t
    new_lightcurves_df['u_flux'] = all_u_temp_arr
    new_lightcurves_df['u_flux_err'] = all_u_err_temp_arr
    new_lightcurves_df['g_flux'] = all_g_temp_arr
    new_lightcurves_df['g_flux_err'] = all_g_err_temp_arr
    new_lightcurves_df['r_flux'] = all_r_temp_arr
    new_lightcurves_df['r_flux_err'] = all_r_err_temp_arr
    new_lightcurves_df['i_flux'] = all_i_temp_arr
    new_lightcurves_df['i_flux_err'] = all_i_err_temp_arr
    new_lightcurves_df['z_flux'] = all_z_temp_arr
    new_lightcurves_df['z_flux_err'] = all_z_err_temp_arr
    new_lightcurves_df['Y_flux'] = all_Y_temp_arr
    new_lightcurves_df['Y_flux_err'] = all_Y_err_temp_arr

    new_lightcurves_df.to_csv(utils.modified_train_filepath, index=False)


def plot_lightcurves(filepath):
    labels = []
    u_data, g_data, r_data, i_data, z_data, Y_data = [], [], [], [], [], []
    ids = []
    print('Loading light curves data')
    lightcurves_df = pd.read_csv(filepath)
    meta_df = pd.read_csv(utils.train_meta_filepath)
    # group by the object id
    lightcurves = lightcurves_df.groupby('object_id')
    for lightcurve in lightcurves:
        id = lightcurve[0]
        ids.append(id)
        meta = meta_df.loc[meta_df['object_id'] == id]
        u, g, r, i, z, Y = [], [], [], [], [], []
        first_obs = None
        for idx, row in lightcurve[1].iterrows():
            if first_obs is None:
                first_obs = float(row['mjd'])
            obs = [float(row['mjd']) - first_obs, float(row['flux']), float(row['flux_err'])]
            if row['passband'] == 0:  # u
                u.append(obs)
            elif row['passband'] == 1:  # g
                g.append(obs)
            elif row['passband'] == 2:  # r
                r.append(obs)
            elif row['passband'] == 3:  # i
                i.append(obs)
            elif row['passband'] == 4:  # z
                z.append(obs)
            elif row['passband'] == 5:  # Y
                Y.append(obs)

        labels.append(meta['target'].iloc[0])
        u_data.append(u)
        g_data.append(g)
        r_data.append(r)
        i_data.append(i)
        z_data.append(z)
        Y_data.append(Y)

    labels = np.array(labels)
    nrows = 1
    ncolumns = 2
    for ip in range(0, 2):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncolumns, sharex=True, sharey=True, figsize=(7, 4))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        for row in range(0, nrows):
            for column in range(0, ncolumns):
                ax = axes[column]
                ax.set_xticks([0, 100, 200, 300])
                ax.set_yticks([0, 250, 500, 750, 1000])
                ax.set_xlabel('Time (days)', fontsize=7)
                if column == 0:
                    ax.set_ylabel('Flux', fontsize=7)
                i = ncolumns * row + column
                data = np.array(u_data[i + ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='violet', linewidth=1)
                data = np.array(g_data[i + ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='green', linewidth=1)
                data = np.array(r_data[i + ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='red', linewidth=1)
                data = np.array(i_data[i + ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='black', linewidth=1)
                data = np.array(z_data[i + ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='blue', linewidth=1)
                data = np.array(Y_data[i + ip])
                ax.errorbar(data[:, 0], data[:, 1], data[:, 2], color='orange', linewidth=1)
                ax.text(75.0, 60.0, 'Type: %s' % labels[i + 16 * ip], fontsize=7)
                ax.text(75.0, 80.0, 'ID: %s' % str(ids[i + 16 * ip]).zfill(3), fontsize=7)
                ax.set_xlim([0, 375])
                ax.set_ylim([-100, 1500])
                ax.tick_params(axis='both', which='major', labelsize=7)
        plt.savefig(utils.result_dir_path + 'plots/%s.pdf' % ip)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-preprocess', action='store_true', default=True)
    parser.add_argument('-lightcurves', action='store_true', default=True)
    args = parser.parse_args()

    if args.preprocess:
        preprocess(utils.train_filepath)

    if args.lightcurves:
        plot_lightcurves(utils.train_filepath)
