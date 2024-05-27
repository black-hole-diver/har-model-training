from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import pickle
import re
import sys
import tarfile
import tempfile
import time
from datetime import datetime as dt
from datetime import timedelta as td

# from chainer.datasets import get_cifar10, get_mnist
# from keras.utils.np_utils import to_categorical
# from chainer.datasets import get_cifar10, get_mnist
# from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import pandas as pd
import scipy
import scipy.spatial as sp
from six.moves import urllib
from sklearn import linear_model as lin
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC, LinearSVC

# from keras.datasets import cifar10,mnist
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential, Model
# from keras.layers import Dense, Input, Dropout, Activation, Flatten, LSTM, SimpleRNN, GRU
# from keras.layers import Convolution2D, MaxPooling2D, normalization
# from keras.utils import np_utils
# import keras.backend as K


# ---------------------------------------------------------------------------
# -------- make_labels_dogids -----------------------------------------------


def make_labels_dogids(ts_raw, dog_names, labels_dict_fname="labels_2020_sep_30.dms"):
    label_list = pd.read_csv(labels_dict_fname, sep=" ")
    label_dict = {}

    for k in range(len(label_list)):
        label_dict[label_list.values[k, 1]] = label_list.values[k, 0]

    labels = np.zeros(len(ts_raw)).astype(np.int32)
    print(labels)
    ts_lab = ts_raw["label"].fillna(value=-1).values
    print(ts_lab[0:10])

    for k in range(len(ts_lab)):

        if ts_lab[k] != -1:
            labels[k] = label_dict.get(ts_lab[k], -1)
        else:
            labels[k] = -1
    del label_list
    del ts_lab

    dog_id = np.zeros(len(ts_raw)).astype(np.int32)

    # dog_list = pd.read_csv(dog_dict_fname,sep=" ")
    dog_dict = {dog_names[i]: i for i in range(0, len(dog_names))}
    print(dog_dict)
    dog_df = pd.DataFrame(dog_dict.items(), columns=["dog_name", "dog_id"])
    print(dog_df)

    # for k in range(len(dog_names)):
    #     dog_dict[dog_names.values[k,2]] = dog_names.values[k,0]

    # dog_names = ts_raw['name'].values

    print(dog_names)

    for k in range(len(dog_id)):
        dog_id[k] = dog_dict[ts_raw.at[k, "name"]]

    return labels, label_dict, dog_id, dog_df


def clearNaN(fname_data):
    data_filt = fname_data.fillna("unknown")
    return data_filt


# ---------------------------------------------------------------------------
# -------- get_raw_ts -------------------------------------------------------


def get_raw_ts(
    fname_labels="labels_2020_sep_30.dms",
    fname_data="sensedog_2020_feb03_ts_lgb_transition_final.csv",
):
    labels = pd.read_csv(fname_labels, delimiter=" ")
    # labels.columns=['ID','Name']
    # labels = labels.append({'ID' : 0,'Name':'unknown'},ignore_index=True)
    cat_inv = {}
    for k in range(len(labels)):
        cat_inv[labels.values[k, 1]] = labels.values[k, 0]

    data = pd.read_csv(fname_data, index_col=0)
    label_nan = data["label"].isnull()

    # data_filt = data[label_nan[:]==False]
    data_filt = data.fillna("unknown")

    dog_occ = data_filt["name"].value_counts()
    dog_fold = np.zeros(len(dog_occ)).astype(np.int32)
    dog_inv = {}
    dog_fold_inv = {}
    for i in range(len(dog_occ)):
        dog_fold[i] = int(i % 3)
        dog_inv[dog_occ.index[i]] = i
        dog_fold_inv[dog_occ.index[i]] = int(i % 3)
    data_filt["user_acceleration_x/z"] = (
        data_filt["user_acceleration_x"] / data_filt["user_acceleration_z"]
    )
    data_filt["user_acceleration_abs"] = np.linalg.norm(
        data_filt[
            ["user_acceleration_x", "user_acceleration_y", "user_acceleration_z"]
        ].values,
        axis=1,
    )
    data_filt["rotation_rate_abs"] = np.linalg.norm(
        data_filt[["rotation_rate_x", "rotation_rate_y", "rotation_rate_z"]].values,
        axis=1,
    )
    del data
    return data_filt, labels, cat_inv, dog_fold, dog_occ, dog_inv, dog_fold_inv


# ---------------------------------------------------------------------------
# -------- convert_to_ts ----------------------------------------------------


def convert_to_ts(filename, path, conv=""):
    df = pd.read_csv(path + filename + ".csv")
    with open(path + filename + conv + ".json") as data_file:
        json_data = json.load(data_file)

    # Sync extraction
    comments = df["Comments"]
    comments = comments[comments.notnull()]
    sync_num = 0
    if len(comments) > 1:
        for anytext in comments:
            if "SYNC" in anytext:
                sync_num = sync_num + 1
                print(anytext)
                if sync_num > 1:
                    print("More than 1 syncs")
                    break

    syncpoint = float(comments.iloc[0].split(" ")[1])
    syncpoint = dt.fromtimestamp(syncpoint)

    # ### Time series labelling
    def getvalues(json_data, feature, sep="     "):
        tmp = json_data[feature]
        df_temp = pd.DataFrame([sub.split() for sub in tmp])
        #         print(df_ua)
        # tmp_array = list(map(lambda x: x.split('     '), tmp))
        #         tmp_array = list(map(lambda x: x.split(sep), tmp))
        #         print("tmp_array =",*tmp_array[0:10],sep = "\n")
        #         return tmp_array
        return df_temp

    #     attitude = getvalues(json_data, 'attitude')
    #     print("attitude =",*attitude[0:10],sep = "\n")
    #     gravity = getvalues(json_data, 'gravity')
    #     rotation_rate = getvalues(json_data, 'rotation_rate')
    #     user_acceleration = getvalues(json_data, 'user_acceleration')

    #     df_ua = pd.DataFrame(user_acceleration)
    df_ua = getvalues(json_data, "user_acceleration")
    df_ua.head(5)
    #     print(df_ua)
    #     print(len(df_ua.columns))
    #     print(df_ua.info())
    #     if len(df_ua.columns) !=4:
    #         attitude = getvalues(json_data, 'attitude',sep=' ')
    #         gravity = getvalues(json_data, 'gravity',sep=' ')
    #         rotation_rate = getvalues(json_data, 'rotation_rate',sep=' ')
    #         user_acceleration = getvalues(json_data, 'user_acceleration',sep=' ')
    #         df_ua = pd.DataFrame(user_acceleration)
    #         print(len(df_ua.columns))
    #         print(df_ua)
    df_ua.columns = [
        "timestamp",
        "user_acceleration_x",
        "user_acceleration_y",
        "user_acceleration_z",
    ]
    #     print(df_ua.info())

    df_at = getvalues(json_data, "attitude")
    #     df_at = pd.DataFrame(attitude)
    df_at.head(5)
    df_at.columns = [
        "timestamp",
        "attitude_1",
        "attitude_2",
        "attitude_3",
        "attitude_4",
    ]
    #     print(df_at.info())

    df_gr = getvalues(json_data, "gravity")
    #     df_gr = pd.DataFrame(gravity)
    df_gr.head(5)
    df_gr.columns = ["timestamp", "gravity_x", "gravity_y", "gravity_z"]

    df_rr = getvalues(json_data, "rotation_rate")
    #     df_rr = pd.DataFrame(rotation_rate)
    df_rr.head(5)
    df_rr.columns = [
        "timestamp",
        "rotation_rate_x",
        "rotation_rate_y",
        "rotation_rate_z",
    ]

    label_time = df["Time"].apply(lambda x: syncpoint + td(seconds=x))
    for index, row in df.iterrows():
        if (
            row["Forelimb(s)"] in ["fore_rasp", "dig"]
            or row["Hindlimb(s)"] in ["hind_rasp", "hind_bend"]
            or row["Body"] in ["roll"]
        ):
            df.loc[index, "Posture/Gait"] = np.NaN
        if row["Nose/Mouth"] == "eat":
            df.loc[index, "Posture/Gait"] = "eat"
        if row["Nose/Mouth"] == "drink":
            df.loc[index, "Posture/Gait"] = "drink"
        if row["Nose/Mouth"] == "chew":
            df.loc[index, "Posture/Gait"] = "chew"
        if row["Body"] == "body_shake":
            df.loc[index, "Posture/Gait"] = "bodyshake"
        if row["Hindlimb(s)"] == "scratch":
            df.loc[index, "Posture/Gait"] = "scratch"
        if row["Body"] == "turn":
            df.loc[index, "Posture/Gait"] = "turn"
        if row["Posture/Gait"] in [
            "lay_side",
            "lay_rightside",
            "lay_leftside",
            "lay_curled",
            "lay_back",
        ]:
            df.loc[index, "Posture/Gait"] = "lay"

    #     for index, row in df.iterrows():
    #         if row["Posture/Gait"] in ['sit','lay','lay_side', 'lay_rightside', 'lay_leftside', 'lay_curled', 'lay_back']:
    #             df.loc[index,'Posture/Gait'] = 'passive'
    #         else:
    #         	df.loc[index,'Posture/Gait'] = 'active'

    label_label = df["Posture/Gait"]
    #     label_label = df['Movement'] KH_version

    ts_time = df_ua["timestamp"].apply(lambda x: dt.fromtimestamp(float(x)))
    labels = []

    label_ind = 0

    for ind in range(len(ts_time)):
        label_start = label_time[label_ind]
        label_end = label_start + td(seconds=0.1)

        if ts_time[ind] < label_start:
            labels.append(np.NaN)
        elif (ts_time[ind] >= label_start) & (ts_time[ind] < label_end):
            labels.append(label_label[label_ind])
        else:
            if label_ind < len(label_label) - 1:
                label_ind += 1
            # WHAT IF THE 50 Hz sampling is not continuous
            labels.append(label_label[label_ind])

    df_full = pd.concat([df_at, df_gr, df_rr, df_ua], axis=1)
    df_full["label"] = labels
    df_full = df_full.drop("timestamp", 1)
    df_full["timestamp"] = df_ua["timestamp"]
    df_full["name"] = [filename] * len(df_full)

    cols = df_full.columns.tolist()

    return df_full[["name", "timestamp", "label"] + cols[0:13]]


# ---------------------------------------------------------------------------
# -------- convert_to_ts_conv -----------------------------------------------


def convert_to_ts_conv(filename, path, conv=""):
    df = pd.read_csv(path + filename + ".csv")
    with open(path + filename + conv + ".json") as data_file:
        json_data = json.load(data_file)

    # Sync extraction
    comments = df["Comments"]
    comments = comments[comments.notnull()]
    sync_num = 0
    if len(comments) > 1:
        for anytext in comments:
            if "SYNC" in anytext:
                sync_num = sync_num + 1
                print(anytext)
                if sync_num > 1:
                    print("More than 1 syncs")
                    break

    syncpoint = float(comments.iloc[0].split(" ")[1])
    syncpoint = dt.fromtimestamp(syncpoint)

    # ### Time series labelling
    def getvalues(json_data, feature):
        tmp = json_data[feature]
        tmp_array = list(map(lambda x: x.split(" "), tmp))
        return tmp_array

    attitude = getvalues(json_data, "attitude")
    gravity = getvalues(json_data, "gravity")
    rotation_rate = getvalues(json_data, "rotation_rate")
    user_acceleration = getvalues(json_data, "user_acceleration")

    df_ua = pd.DataFrame(user_acceleration)
    df_ua.columns = [
        "timestamp",
        "user_acceleration_x",
        "user_acceleration_y",
        "user_acceleration_z",
    ]

    df_at = pd.DataFrame(attitude)
    df_at.columns = [
        "timestamp",
        "attitude_1",
        "attitude_2",
        "attitude_3",
        "attitude_4",
    ]

    df_gr = pd.DataFrame(gravity)
    df_gr.columns = ["timestamp", "gravity_x", "gravity_y", "gravity_z"]

    df_rr = pd.DataFrame(rotation_rate)
    df_rr.columns = [
        "timestamp",
        "rotation_rate_x",
        "rotation_rate_y",
        "rotation_rate_z",
    ]

    label_time = df["Time"].apply(lambda x: syncpoint + td(seconds=x))
    for index, row in df.iterrows():
        if (
            row["Forelimb(s)"] in ["fore_rasp", "dig"]
            or row["Hindlimb(s)"] in ["scratch", "hind_rasp", "hind_bend"]
            or row["Body"] in ["body_shake", "roll"]
            or row["Nose/Mouth"] in ["chew"]
        ):
            df.loc[index, "Posture/Gait"] = np.NaN
        if row["Nose/Mouth"] == "eat":
            df.loc[index, "Posture/Gait"] = "eat"
        if row["Nose/Mouth"] == "drink":
            df.loc[index, "Posture/Gait"] = "drink"

    label_label = df["Posture/Gait"]

    ts_time = df_ua["timestamp"].apply(lambda x: dt.fromtimestamp(float(x)))
    labels = []

    label_ind = 0

    for ind in range(len(ts_time)):
        label_start = label_time[label_ind]
        label_end = label_start + td(seconds=0.1)

        if ts_time[ind] < label_start:
            labels.append(np.NaN)
        elif (ts_time[ind] >= label_start) & (ts_time[ind] < label_end):
            labels.append(label_label[label_ind])
        else:
            if label_ind < len(label_label) - 1:
                label_ind += 1
            # WHAT IF THE 50 Hz sampling is not continuous
            labels.append(label_label[label_ind])

    df_full = pd.concat([df_at, df_gr, df_rr, df_ua], axis=1)
    df_full["label"] = labels
    df_full = df_full.drop("timestamp", 1)
    df_full["timestamp"] = df_ua["timestamp"]
    df_full["name"] = [filename] * len(df_full)

    cols = df_full.columns.tolist()

    return df_full[["name", "timestamp", "label"] + cols[0:16]]


# ---------------------------------------------------------------------------
# -------- convert_to_ts_KH -------------------------------------------------


def convert_to_ts_KH(filename, path, conv=""):
    df = pd.read_csv(path + filename + ".csv")
    with open(path + filename + conv + ".json") as data_file:
        json_data = json.load(data_file)

    # Sync extraction
    comments = df["Comments"]
    comments = comments[comments.notnull()]
    sync_num = 0
    if len(comments) > 1:
        for anytext in comments:
            if "SYNC" in anytext:
                sync_num = sync_num + 1
                print(anytext)
                if sync_num > 1:
                    print("More than 1 syncs")
                    break

    syncpoint = float(comments.iloc[0].split(" ")[1])
    syncpoint = dt.fromtimestamp(syncpoint)

    # ### Time series labelling
    def getvalues(json_data, feature, sep="     "):
        tmp = json_data["data"][feature]
        df_temp = pd.DataFrame([sub.split() for sub in tmp])
        #         print(df_ua)
        # tmp_array = list(map(lambda x: x.split('     '), tmp))
        #         tmp_array = list(map(lambda x: x.split(sep), tmp))
        #         print("tmp_array =",*tmp_array[0:10],sep = "\n")
        #         return tmp_array
        return df_temp

    #     attitude = getvalues(json_data, 'attitude')
    #     print("attitude =",*attitude[0:10],sep = "\n")
    #     gravity = getvalues(json_data, 'gravity')
    #     rotation_rate = getvalues(json_data, 'rotation_rate')
    #     user_acceleration = getvalues(json_data, 'user_acceleration')

    #     df_ua = pd.DataFrame(user_acceleration)
    df_ua = getvalues(json_data, "user_acceleration")
    df_ua.head(5)
    #     print(df_ua)
    #     print(len(df_ua.columns))
    #     print(df_ua.info())
    #     if len(df_ua.columns) !=4:
    #         attitude = getvalues(json_data, 'attitude',sep=' ')
    #         gravity = getvalues(json_data, 'gravity',sep=' ')
    #         rotation_rate = getvalues(json_data, 'rotation_rate',sep=' ')
    #         user_acceleration = getvalues(json_data, 'user_acceleration',sep=' ')
    #         df_ua = pd.DataFrame(user_acceleration)
    #         print(len(df_ua.columns))
    #         print(df_ua)
    df_ua.columns = [
        "timestamp",
        "user_acceleration_x",
        "user_acceleration_y",
        "user_acceleration_z",
    ]
    #     print(df_ua.info())

    df_at = getvalues(json_data, "attitude")
    #     df_at = pd.DataFrame(attitude)
    df_at.head(5)
    df_at.columns = [
        "timestamp",
        "attitude_1",
        "attitude_2",
        "attitude_3",
        "attitude_4",
    ]
    #     print(df_at.info())

    df_gr = getvalues(json_data, "gravity")
    #     df_gr = pd.DataFrame(gravity)
    df_gr.head(5)
    df_gr.columns = ["timestamp", "gravity_x", "gravity_y", "gravity_z"]

    df_rr = getvalues(json_data, "rotation_rate")
    #     df_rr = pd.DataFrame(rotation_rate)
    df_rr.head(5)
    df_rr.columns = [
        "timestamp",
        "rotation_rate_x",
        "rotation_rate_y",
        "rotation_rate_z",
    ]

    label_time = df["Time"].apply(lambda x: syncpoint + td(seconds=x))
    #    for index, row in df.iterrows():

    label_label = df["Movement"]

    ts_time = df_ua["timestamp"].apply(lambda x: dt.fromtimestamp(float(x)))
    labels = []

    label_ind = 0

    for ind in range(len(ts_time)):
        label_start = label_time[label_ind]
        # label_end = label_start + td(seconds=0.1)
        label_end = label_time[label_ind + 1]

        if ts_time[ind] < label_start:
            labels.append(np.NaN)
        elif (ts_time[ind] >= label_start) & (ts_time[ind] < label_end):
            labels.append(label_label[label_ind])
        else:

            if label_ind < len(label_label) - 2:
                label_ind += 1
                labels.append(label_label[label_ind])
            # WHAT IF THE 50 Hz sampling is not continuous
            else:
                labels.append(np.NaN)

    df_full = pd.concat([df_at, df_gr, df_rr, df_ua], axis=1)
    df_full["label"] = labels
    df_full = df_full.drop("timestamp", 1)
    df_full["timestamp"] = df_ua["timestamp"]
    df_full["name"] = [filename] * len(df_full)

    cols = df_full.columns.tolist()

    return df_full[["name", "timestamp", "label"] + cols[0:13]]


# ---------------------------------------------------------------------------
# -------- stat_features ----------------------------------------------------


def stat_features(
    data_filt, cat_inv, dog_inv, cat_pol="last", window=32, shift=16, verbose=False
):
    ndogs = data_filt["name"].value_counts().index
    x_all = []
    y_all = []
    x_t_all = []
    # cat_hist = np.zeros((ndogs,len(cat_inv))).astype(np.int32)
    cat_hist_all = []
    ts_str = []
    ts_end = []

    for i in range(len(ndogs)):
        data = data_filt[data_filt["name"] == ndogs[i]]
        dogid = dog_inv[ndogs[i]]
        nstat_feat = 2 * 5
        nts = len(data.head(1).values[0, 4:])
        dim = nstat_feat * nts
        ts_len = int((len(data) - window) / shift + 1)

        if verbose == True:
            print(
                "%dth dog samples: %d dimension: %d (timeseries: %d) name: %s"
                % (i, ts_len, dim, nts, ndogs[i])
            )

        c = dogid * np.ones((ts_len, 1)).astype(np.int32)
        c_str = []
        y = np.zeros((ts_len, 1)).astype(np.int32)
        x = np.zeros((ts_len, dim)).astype(np.float32)
        x_t = np.zeros((ts_len, 2)).astype(np.int32)

        if cat_pol == "last":
            cat_idx = window - 1
        if cat_pol == "first":
            cat_idx = 0

        act_y = data["label"].values
        act_x = data.values[:, 4:].astype(np.float32)
        ts_act = data["timestamp"].values
        act_grad_x = np.zeros((len(act_x), len(act_x[0, :])))
        act_grad_x[1:] = act_x[1:] - act_x[:-1]
        # print("diff: %f %s and %s" %(np.linalg.norm(act_grad_x-act_x),act_x.shape,act_grad_x.shape))

        cat_hist = np.zeros((ts_len, len(cat_inv) + 1)).astype(np.int32)

        for k in range(ts_len):
            str_x = k * shift
            act = act_x[str_x : str_x + window]
            act_grad = act_grad_x[str_x : str_x + window]
            y[k, 0] = int(cat_inv[act_y[str_x + cat_idx]])
            x_t[k, 0] = ts_act[str_x : str_x + window][0]
            x_t[k, 1] = ts_act[str_x : str_x + window][-1]

            # print("\tact diff:  %f %s vs %s" %(np.linalg.norm(act_grad-act),act.shape,act_grad.shape))

            for l in range(window):
                act_cat = int(cat_inv[act_y[str_x + cat_idx]])
                cat_hist[k, act_cat] += 1
            x[k, :nts] = np.min(act, axis=0)
            x[k, nts : 2 * nts] = np.max(act, axis=0)
            x[k, 2 * nts : 3 * nts] = np.mean(act, axis=0)
            x[k, 3 * nts : 4 * nts] = np.std(act, axis=0)
            x[k, 4 * nts : 5 * nts] = scipy.stats.skew(act, axis=0)
            x[k, 5 * nts : 6 * nts] = np.min(act_grad, axis=0)
            x[k, 6 * nts : 7 * nts] = np.max(act_grad, axis=0)
            x[k, 7 * nts : 8 * nts] = np.mean(act_grad, axis=0)
            x[k, 8 * nts : 9 * nts] = np.std(act_grad, axis=0)
            x[k, 9 * nts : 10 * nts] = scipy.stats.skew(act_grad, axis=0)
            # print("\t %dth dog %dth td diff mean: %f" % (i,k,np.linalg.norm(x[k,7*nts:8*nts]-x[k,2*nts:3*nts])))
            # print("\t\tafter diff: %f %s and %s" %(np.linalg.norm(act_grad_x-act_x),act_x.shape,act_grad_x.shape))
        x_all.append(np.concatenate((x_t, c, y, x), axis=1))

        cat_hist_all.append(cat_hist)
    full = x_all[0]
    for k in range(len(x_all) - 1):
        full = np.concatenate((full, x_all[k + 1]), axis=0).astype(np.float32)
    full = np.nan_to_num(np.array(full).astype(np.float32))
    return full, cat_hist_all


# ---------------------------------------------------------------------------
# -------- feat_to_file -----------------------------------------------------


def feat_to_file(full, data_filt, dog_occ, labels, fname):
    f = open(fname, "w")
    num = 0

    print("sample_id|sample_seq_id|dog|dog_id|label|label_id", file=f, end="")
    j = 0
    for s2 in ["raw", "grad"]:
        for s in ["min", "max", "mean", "std", "skew"]:
            for a in data_filt.columns[4:]:
                print("|feat_%d_%s_%s_%s" % (j, a, s2, s), end="", file=f)
                j += 1
    print("", end="\n", file=f)
    f.flush()

    for i in range(len(full)):
        if i > 0:
            if int(full[i, 2]) != last:
                num = 0
            else:
                num += 1
        last = int(full[i, 2])
        print(
            "%d|%d|%s|%d|%s|%d"
            % (
                i,
                num,
                dog_occ.index[int(full[i, 2])],
                int(full[i, 2]),
                labels[labels.values[:, 0] == int(full[i, 3])].values[0, 1],
                int(full[i, 3]),
            ),
            end="",
            file=f,
        )
        for j in range(len(full[0, 4:])):
            print("|%f" % (full[i, j + 4]), end="", file=f)
        print("", end="\n", file=f)
        f.flush()
    return


def rbf_kernel(x_te, x_tr, gamma):
    return np.exp(-gamma * sp.distance.cdist(x_te, x_tr, metric="sqeuclidean")).astype(
        np.float16
    )


def rbf_pre_kernel(x_te, x_tr):
    return sp.distance.cdist(x_te, x_tr, metric="sqeuclidean").astype(np.float32)


#    return gamma*sp.distance.cdist(x_te,x_tr,metric='sqeuclidean').astype(np.float32)


def rbf_pre_kernel_tr(x_tr):
    return sp.distance.pdist(x_tr, metric="sqeuclidean").astype(np.float16)


#    return gamma*sp.distance.cdist(x_te,x_tr,metric='sqeuclidean').astype(np.float32)


def dot_kernel(x_te, x_tr, gamma):
    return (gamma * x_te.dot(x_tr.T)).astype(np.float32)


def dot_rbf_kernel(x_te, x_tr, gamma=1):
    base = 2.0 * gamma * (x_te.astype(np.float32)).dot(x_tr.T.astype(np.float32))
    for k in range(len(base[:, 0])):
        base[k] -= gamma * (x_te[k].astype(np.float32)).dot(x_te[k].astype(np.float32))
    for k in range(len(base[0, :])):
        base[:, k] -= gamma * (x_tr[k].astype(np.float32)).dot(
            x_tr[k].astype(np.float32)
        )
    # np.fill_diagonal(base,0.0)
    return base  # .astype(np.float32)


def dot_rbf_kernel_tr(x_tr, gamma=1):
    base = 2.0 * gamma * (x_tr.astype(np.float32)).dot(x_tr.T.astype(np.float32))
    for k in range(len(base[:, 0])):
        act = gamma * (x_tr[k].astype(np.float32)).dot(x_tr[k].astype(np.float32))
        base[k, :] -= act
        base[:, k] -= act
    np.fill_diagonal(base, 0.0)
    return base  # (0.5*(base+base.T)).astype(np.float32)


def l1_norm(x):
    x_n = np.copy(x)
    for i in range(x[:, 0].size):
        n = np.sum(np.abs(x[i, :]))
        if n > 0:
            x_n[i, :] = x[i, :] / n
    return x_n


def l2_norm(x):
    x_n = np.copy(x)
    for i in range(x[:, 0].size):
        n = np.linalg.norm(x[i, :])
        if n > 0:
            x_n[i, :] = x[i, :] / n
    return x_n


def pow_norm(x, alpha):
    x_n = np.copy(x)
    for i in range(x[:, 0].size):
        for j in range(x[i, :].size):
            val = np.power(np.abs(x[i, j]), alpha)
            s = 1
            if x[i, j] < 0:
                s = -1
            x_n[i, j] = s * val
    return x_n


def scale_norm(x, y):
    x_n = np.copy(x)
    x_max = np.max(y, axis=0)
    x_min = np.min(y, axis=0)
    x_range = x_max - x_min
    for j in range(x_range.size):
        if x_range[j] == 0:
            x_range[j] = 1
    for i in range(x[:, 0].size):
        x_n[i] = (x[i] - x_min) / x_range
    return x_n


def scale_norm_min_max(x, x_max, x_min):
    x_n = np.copy(x)
    # x_max  = np.max(y,axis=0)
    # x_min  = np.min(y,axis=0)
    x_range = x_max - x_min
    for j in range(x_range.size):
        if x_range[j] == 0:
            x_range[j] = 1
    for i in range(x[:, 0].size):
        x_n[i] = (x[i] - x_min) / x_range
    return x_n


def std_norm_mean_std(x, x_mean, x_std):
    x_n = np.copy(x)
    # x_mean  = np.mean(y,axis=0)
    # x_std  = np.std(y,axis=0)
    for j in range(x_std.size):
        if x_std[j] == 0:
            x_std[j] = 1
    for i in range(x[:, 0].size):
        x_n[i] = (x[i] - x_mean) / x_std
    return np.nan_to_num(x_n)


def std_norm(x, y):
    x_n = np.copy(x)
    x_mean = np.mean(y, axis=0)
    x_std = np.std(y, axis=0)
    for j in range(x_std.size):
        if x_std[j] == 0:
            x_std[j] = 1
    for i in range(x[:, 0].size):
        x_n[i] = (x[i] - x_mean) / x_std
    return x_n


def prob_context(x, cx=2, cy=2):
    ncat = x[0].size
    x_out = np.zeros((len(x), (cx + cy + 1) * ncat)).astype(np.float32)
    for k in range(len(x)):
        x_out[k, cx * ncat : (cx + 1) * ncat] = x[k]
        if k < cx:
            for l in range(cx):
                x_out[k, l * ncat : (l + 1) * ncat] = x[k]
        else:
            x_out[k, : cx * ncat] = x[k - cx : k].flatten()
        if len(x) - 1 - k < cy:
            for l in range(cy):
                x_out[k, (cx + l) * ncat : (cx + l + 1) * ncat] = x[k]
        else:
            x_out[k, (cx + 1) * ncat :] = x[k + 1 : k + 1 + cy].flatten()
    return x_out


def full_context(new_meta, cx=2, cy=2, old=False):
    dogs = new_meta["name"].value_counts()
    dog_names = dogs.index
    n_dogs = len(dog_names)
    ncat = new_meta.values[0, 3:].size
    cont_feat = np.zeros((len(new_meta), (cx + cy + 1) * ncat)).astype(np.float32)
    cont_y = np.zeros(len(new_meta)).astype(np.int32)
    act_pos = 0
    for k in range(n_dogs):
        cont_base = (
            new_meta[new_meta["name"] == dog_names[k]].values[:, 3:].astype(np.float32)
        )
        cont_feat[act_pos : act_pos + len(cont_base)] = prob_context(cont_base, cx, cy)
        cont_y[act_pos : act_pos + len(cont_base)] = (
            new_meta[new_meta["name"] == dog_names[k]].values[:, 2].astype(np.float32)
        )
        act_pos = act_pos + len(cont_base)
    cont_dog_id = np.zeros(len(new_meta))
    dog_name_raw = new_meta["name"].values
    for i in range(len(cont_dog_id)):
        for j in range(len(dog_names)):
            if dog_names[j] == dog_name_raw[i]:
                cont_dog_id[i] = j
    if old == False:
        return cont_feat, cont_y, cont_dog_id
    else:
        # cont_feat_old= cont_feat[(cont_y[:]==1) | (cont_y[:]==2) | (cont_y[:]==3) | (cont_y[:]==8) | (cont_y[:]==9) | (cont_y[:]==10)]
        # cont_y_old= cont_y[(cont_y[:]==1) | (cont_y[:]==2) | (cont_y[:]==3) | (cont_y[:]==8) | (cont_y[:]==9) | (cont_y[:]==10)]
        # cont_dog_id_old= cont_dog_id[(cont_y[:]==1) | (cont_y[:]==2) | (cont_y[:]==3) | (cont_y[:]==8) | (cont_y[:]==9) | (cont_y[:]==10)]
        cont_feat_old = cont_feat[
            (cont_y[:] == 2)
            | (cont_y[:] == 3)
            | (cont_y[:] == 4)
            | (cont_y[:] == 9)
            | (cont_y[:] == 10)
            | (cont_y[:] == 11)
        ]
        cont_y_old = cont_y[
            (cont_y[:] == 2)
            | (cont_y[:] == 3)
            | (cont_y[:] == 4)
            | (cont_y[:] == 9)
            | (cont_y[:] == 10)
            | (cont_y[:] == 11)
        ]
        cont_dog_id_old = cont_dog_id[
            (cont_y[:] == 2)
            | (cont_y[:] == 3)
            | (cont_y[:] == 4)
            | (cont_y[:] == 9)
            | (cont_y[:] == 10)
            | (cont_y[:] == 11)
        ]
        return cont_feat_old, cont_y_old, cont_dog_id_old


def transform_for_rnn(data, recurrence):
    res = []
    for ind in range(len(data) - recurrence):
        res.append(data[ind : ind + recurrence, :])
    res = np.dstack(res)
    res = np.rollaxis(res, -1)

    return res


def filter_old(x, y, cat=[2, 3, 4, 9, 10, 11]):
    return x[np.isin(y[:], cat)], y[np.isin(y[:], cat)]


def transform_y_for_rnn(data, recurrence):
    res = []
    for ind in range(len(data) - recurrence):
        res.append(data[ind : ind + recurrence])
    res = np.dstack(res)
    res = np.rollaxis(res, -1)

    return res


def onehot_encode(l):
    res = np.zeros(len(l))
    for item in l:
        res[np.argmax(item)] += 1.0 / len(l)

    return res


def keras_train(
    model, x_train, y_train, batch_size=32, nb_epoch=1, data_augmentation=False
):
    if not data_augmentation:
        print("w/o data augmentation.")
        model.fit(
            x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True
        )
    else:
        print("with real-time data augmentation.")
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
        )  # randomly flip images
        datagen.fit(x_train)
        print("augmentation: %s" % (str(x_train.shape[0])))
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            samples_per_epoch=x_train.shape[0],
            nb_epoch=nb_epoch,
        )


def keras_predict(model, x_test, batch_size=32):
    return model.predict(
        x=x_test,
        batch_size=batch_size,
    )


def keras_evaluate(model, x_test, y_test, batch_size=32):
    return model.evaluate(x=x_test, y=y_test, batch_size=batch_size)


def keras_mlp(input_shape=(56), n_cla=28, hidd=32):
    model = Sequential()
    model.add(Dense(hidd, input_dim=input_shape))
    model.add(Activation("relu"))
    model.add(Dense(n_cla, name="fc1"))
    model.add(Activation("softmax"))
    return model


def keras_cnn():
    model = Sequential()
    model.add(
        Convolution2D(32, 3, 3, border_mode="same", input_shape=X_train.shape[1:])
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def keras_eto_rnn(input_dim, input_len, n_cla=28, rec_layer=16, layer_type="gru"):
    model = Sequential()

    if layer_type == "lstm":
        model.add(LSTM(rec_layer, input_dim=input_dim, input_length=input_len))
    elif layer_type == "gru":
        model.add(GRU(rec_layer, input_dim=input_dim, input_length=input_len))
    elif layer_type == "simplernn":
        model.add(SimpleRNN(rec_layer, input_dim=input_dim, input_length=input_len))
    model.add(Dense(n_cla, name="fc1"))
    model.add(Activation("softmax"))
    return model


def keras_eto_rnn_mlp(
    input_dim, input_len, n_cla=28, rec_layer=16, hidd=32, layer_type="gru"
):
    model = Sequential()

    if layer_type == "lstm":
        model.add(LSTM(rec_layer, input_dim=input_dim, input_length=input_len))
    elif layer_type == "gru":
        model.add(GRU(rec_layer, input_dim=input_dim, input_length=input_len))
    elif layer_type == "simplernn":
        model.add(SimpleRNN(rec_layer, input_dim=input_dim, input_length=input_len))
    model.add(Dense(hidd, name="fc1"))
    model.add(Activation("relu"))
    model.add(Dense(n_cla, name="fc2"))
    model.add(Activation("softmax"))
    return model
