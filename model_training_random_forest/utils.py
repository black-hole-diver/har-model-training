# import libraries

import bz2
from copy import deepcopy
import glob
import os
import pickle
# from dtw import dtw

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import matplotlib.lines as mlines
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
sns.set_theme()



# read .pbz2 files into a tuple of metadata + smartwatch readings

def read_pbz2(path):
    """
    Read a .pbz2 file from the file path

    Parameters:
    path (str) -- Path of the file to read

    Returns:
    tuple (metadata, smartwatch data) -- Tuple of metadata and smartwatch readings
    """

    # reading the file to the file handle using with statement and bz function
    with bz2.BZ2File(path, 'rb') as f:
        # read the data into data
        data = f.read()
    # uncompress the read data to the tuple
    decomp_data = pickle.loads(data)


    return decomp_data


# get a list of empty .csv files
def get_empty_csv(directory):
    """
    Get a list of empty .csv files (to avoid errors during data reading)

    Parameters:
    directory (str) -- Path of the directory containing the files

    Returns:
    list ([file_1, file_2, ...]) -- List containing names of csv files that are empty
    """

    # initialize empty list
    empty_files = []
    # list all .csv files in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    # loop over all .csv files
    for file_name in csv_files:
        # check if the file can be read
        try:
            pd.read_csv(file_name)
        except:
            # if not, append to the list
            empty_files.append(file_name)


    return empty_files


# get a dictionary of all files
# keys as experiment id, and values as a list of files
def get_files_dict():
    """
    Organize all files based on experiment ID for ease of use

    Parameters:
    directory (str) -- Path of the directory containing the files

    Returns:
    dict -- Dictionary of experiment ID and relevant files
    """

    # specify the directory
    directory = '/project/data/'
    # list all .pbz2 files
    pbz2_files = glob.glob(os.path.join(directory, '*.pbz2'))
    # list all .csv files
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    # list of empty csv files
    empty_files = get_empty_csv(directory)
    # empty dictionary
    files_dict = {}
    # empty list to contain non labeled files
    unlabeled_data = []
    # empty list for incorrect .csv files
    incorrect_csvs = []

    # loop over .pbz2 files
    for file_name in pbz2_files:
        # get the short name of the file (without the directory)
        file_name_short = os.path.basename(file_name)
        # split the name
        k = os.path.splitext(file_name_short)[0]
        # append the file to the files dictionary
        files_dict[k] = [file_name]

    # loop over .csv files
    for file_name in csv_files:
        # exempt empty .csv files
        if file_name not in empty_files:
            # get the file name and add it to the dict similar to .pbz2 files
            file_name_short = os.path.basename(file_name)
            k = file_name_short.split('_')[0]
            files_dict[k].append(file_name)
            
    # loop over the dictionary and remove keys that don't include .csv files
    for k, v in files_dict.items():
        if len(v) == 1:
            no_labels = k
            unlabeled_data.append(no_labels)
        # if there are two .csv files remove the incorrect one
        if len(v) == 3:
            # loop over the list of files for each experiment
            for i, f in enumerate(v):
                # if the file name ends with movements.csv, get the key and index
                if f.endswith('movements.csv'):
                    incorrect_csv = (k, i)
                    # append them to the list
                    incorrect_csvs.append(incorrect_csv)

    # loop over list of incorrect csvs and remove the files
    for csv in incorrect_csvs:
        files_dict[csv[0]].pop(csv[1])
                
    # loop over the list of unlabeled data and remove them
    for n in unlabeled_data:
        del files_dict[n]

    
    return files_dict


# helper function for labeling smartwatch readings
def merge_dfs(df_1, df_2):
    """
    Merge smartwatch data with labels on timestamp

    Parameters:
    df_1 (DataFrame) -- Dataframe containing smartwatch data
    df_2 (DataFrame) -- Dataframe containing labels

    Returns:
    DataFrame -- Dataframe with smartwatch readings and labels
    """

    readings = deepcopy(df_1)
    readings = readings.drop(['att_w', 'att_x', 'att_y', 'att_z'], axis=1)
    labels = deepcopy(df_2)
    

    # get the timestamp fo the first reading from the labels
    ts = float(labels.Comments[0].split(' ')[1])
    # align the time with the readings
    readings['t'] = readings['t'] - ts
    readings = readings[readings['t'] >= 0]
    # drop unnecessary columns
    labels = labels.drop(['Comments', 'Handedness'], axis=1)
    
    # set the time as the index for both the readings and the labels
    readings = readings.set_index('t')
    labels = labels.set_index('Time')

    # first align the time by creating a common time window
    # time_start = max(readings.index.min(), labels.index.min())
    time_stop = min(readings.index.max(), labels.index.max())

    # trim the start and end of the dataframes to match the time span
    # readings = readings.loc[readings.index >= time_start]
    readings = readings.loc[readings.index <= time_stop]
    # labels = labels.loc[labels.index >= time_start]
    labels = labels.loc[labels.index <= time_stop]

    # merge the readings and labels on timestamp
    # using backward direction to label readings with values where
    # time is less than or equal to the labels time
    df = pd.merge_asof(readings, labels, left_index=True, right_index=True,
                       allow_exact_matches=True, direction='backward')
    
    df['Movement'] = df['Movement'].replace(np.nan, 'No Movement')
    df['Movement'] = df['Movement'].replace('Glas_grabbing', 'Glass_grabbing')
    
    return df


# plot the movements activity over time
def plot_move(df_1, df_2, start, end):
    """
    Plot movement activity over time

    Parameters:
    df (DataFrame) -- Labeled readings data
    labels (DataFrame) -- Original Labels data
    start (int) -- Index of the number of seconds from the start to begin from
    end (int) -- Index of the number of seconds from the start to end at

    Returns:
    None
    """

    df = deepcopy(df_1)
    labels = deepcopy(df_2)

    # starting timestamp
    # TS = df.index[0]

    # indexing into the data
    # freq = 's'
    df = df.loc[(df.index >= start) & (df.index < end)]
    ts_1 = df.index
    # 0 when no movement, 1 for any movement
    move_1 = np.where(df['Movement'] == 'No Movement', 0, 1)

    # get the timestamp fo the first reading from the labels
    # ts = float(labels.Comments[0].split(' ')[1])
    # change the time column by adding the time stamp to the time
    # labels['Time'] = labels.Time.apply(lambda x: x + ts)
    # labels['Time'] = pd.to_datetime(labels['Time'], unit='s')
    labels = labels.set_index('Time')

    labels = labels.loc[(labels.index >= start) & (labels.index < end)]
    ts_2 = labels.index
    move_2 = np.where(labels.Movement.isna(), 0, 1)

    # specify plot size
    fig, ax = plt.subplots(figsize=(12, 3))
    # line plot for original labels and labeled readings
    sns.lineplot(x=ts_1, y=move_1, color='b', ax=ax)
    sns.lineplot(x=ts_2, y=move_2, color='r', ax=ax)
    # labeling plot
    ax.set_title('Movement Activity over Time')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Movement Activity')
    # setting the ticks
    ax.set_yticks([0, 1])
    # setting the legend
    blue_line = mlines.Line2D([], [], color='b', marker='_',
                              markersize=15, label='Labeled Readings')
    red_line = mlines.Line2D([], [], color='r', marker='_',
                             markersize=15, label='Original Labels')
    ax.legend(handles=[blue_line, red_line], loc='upper left')

    plt.show()


# plot signal data
def plot_signal(df, start, end):
    """
    Plot signals from sensor readings

    Parameters:
    df (DataFrame) -- DataFrame containing signals data from various sensors
    start (int) -- Index of the number of seconds from the start to begin from
    end (int) -- Index of the number of seconds from the start to end at

    Returns:
    None
    """

    # starting timestamp
    # TS = df.index[0]
    # freq = 's'

    # indexing into the data
    df = df.loc[(df.index >= start) & (df.index < end)]

    # get the number of columns
    cols = len(df.columns)

    # specify number of rows and columns for the plot
    n_rows = cols
    n_cols = 1

    # create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))

    # loop over the columns and plot them
    for col in range(cols):
        # specify the ax
        ax = axes[col]
    
        # line plot for original labels and labeled readings
        sns.lineplot(x=df.index, y=df.iloc[:, col], ax=ax)
        ax.set_xlabel('')
        
    fig.supxlabel('Timestamp')
    fig.supylabel('Signal Readings')
    # Adjust spacing between subplots if needed
    plt.tight_layout()
    # show the plots
    plt.show()


# print & plot multivariate warping distance
def dtw_multi(csv_1, csv_2, start=0, end=-1):
    """
    Print and plot the warping distance of multivariate time series

    Parameters:
    csv_1 (str) -- File path of the first .csv file
    csv_2 (str) -- File path of the second .csv file

    Returns:
    None
    """

    # import the data
    df_1 = pd.read_csv(csv_1, index_col=[0]) \
        .reset_index(drop=True)
    df_2 = pd.read_csv(csv_2, index_col=[0]) \
        .reset_index(drop=True)

    # align the two time series to match in length
    df_1, df_2 = df_1.align(df_2, fill_value=0)


    # index a section of the time series
    ts_1 = df_1.iloc[start:end].values
    ts_2 = df_2.iloc[start:end].values

    # get the distance matrix between the two time series
    dist_mat = cdist(ts_1, ts_2, metric='euclidean')

    # get the DTW object
    da = dtw(dist_mat, step_pattern='asymmetric', keep_internals=True)
    # plot and print the warping distance
    dist = da.distance
    plt.figure(figsize=(10, 6))
    plt.plot(da.index1, da.index2)
    plt.xlabel('Time Series 1')
    plt.ylabel('Time Series 2')
    plt.title('Warping Distance Curve')
    plt.text(len(ts_1)-10, 0.05, f'Distance = {dist:.2f}', fontsize=11)
    plt.show()


# print & plot dynamic time warping for univariate time series per variable
def dtw_uni(csv_1, csv_2, start=0, end=-1):
    """
    Print and plot the warping distance of univariate time series per variable

    Parameters:
    csv_1 (str) -- File path of the first .csv file
    csv_2 (str) -- File path of the second .csv file

    Returns:
    None
    """

    # import the data
    df_1 = pd.read_csv(csv_1, index_col=[0]) \
        .reset_index(drop=True)
    df_2 = pd.read_csv(csv_2, index_col=[0]) \
        .reset_index(drop=True)

    # align the two time series to match in length
    df_1, df_2 = df_1.align(df_2, fill_value=0)

    # index a section of the time series
    ts_1 = df_1.iloc[start:end].values
    ts_2 = df_2.iloc[start:end].values

    # get the number of columns
    cols = min(len(df_1.columns), len(df_2.columns))

    # specify number of rows and columns for the plot
    n_rows = cols
    n_cols = 1

    # create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6 * n_rows))

    # loop over the columns and plot them
    for col in range(cols):
        # specify the ax
        ax = axes[col]
        # get the query and template to align & plot
        query = ts_1[:, col]
        template = ts_2[:, col]
        alignment = dtw(query, template, step_pattern='asymmetric',
                        keep_internals=True)
        ax.plot(query, label='df_1')
        ax.plot(template, label='df_2')
        ax.legend()
        ax.set_title(f'{df_1.columns[col]}')
        dist = alignment.distance
        ax.annotate(f'Distance = {dist:.2f}', xy=(0.8, 0.05), xycoords='axes fraction',
                    fontsize=9, color='black')

    fig.suptitle('Signal Readings Comparison')
    fig.supxlabel('Timestep')
    fig.supylabel('Signal Readings')
    # Adjust spacing between subplots if needed
    plt.tight_layout()
    # show the plots
    plt.show()


def generate_windows(X, y, window_size, stride):
    """
    Generate windows of time series data

    Parameters:
    X (ndarray) -- Numpy array containing the features
    y (ndarray) -- Numpy array containing the respective targets
    window_size (int) -- Integer specifying the window length/size
    stride (int) -- Integer specifying the start of the next window

    Returns:
    tup (ndarray, ndarray) -- Tuple of windowed numpy arrays
    """
    # encode y for mode
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = sliding_window_view(X, window_size, axis=0)[::stride]
    y = sliding_window_view(y, window_size)[::stride]

    # getting the y value based on the mode
    y = mode(y, axis=1, keepdims=False)[0]

    # get back the original values
    y = le.inverse_transform(y)

    return X, y


# helper function to select child ids for train, validation, and test groups
def get_test_id(ids):
    """
    Split one child from the experiment for the test set.

    Parameters:
    ids (list) -- List of children IDs

    Returns:
    tuple(lists) -- Tuple of lists containing IDs for three groups
    """
    # change the list to a unique one
    ids_unique = list(set(ids))
    # sample the validation and test IDs
    test_id = np.random.choice(ids_unique, 1)
    
    return test_id


# get the instances of each movement class of a dataset
def get_instances(df):
    """
    Retrieve the instances of each movement class in a Dataframe

    Parameters:
    df (pd.DataFrame) -- DataFrame of sensor readings and their labels

    Returns:
    tuple (list(ndarray), list(ndarray)) -- Get a tuple of x and y
        as a list of numpy arrays
    """
    # condition to set to true when the value of movement changes
    mask = df['Movement'] != df['Movement'].shift(1)
    # get the indices of the condition
    idx = df.index[mask].to_list()
    # split the dataframe on the indices
    dfs = np.split(df, idx, axis=0)[1: ]
    # get the numpy arrays of x and y
    x = [i.drop('Movement', axis=1).values for i in dfs]
    y = [i['Movement'].values[0] for i in dfs]

    return x, y

# get the maximum length of instances
def max_len(x):
    max_len = max([len(i) for i in x])
    return max_len
    

def transform_to_same_length(x, max_length):
    # get the number of datasets/instances
    n = len(x)
    # get the number of variables in each set
    n_var = x[0].shape[1]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each instance
    for i in range(n):
        mts = x[i]
        # get the length of the instance
        curr_length = mts.shape[0]
        # create a padding width equal to the maximum length minus the current length
        pad_width = max_length - curr_length
        # create left and right pads
        left = pad_width // 2
        right = pad_width - left
        # loop over the variables
        for j in range(n_var):
            ts = mts[:, j]
            # pad with wrap mode from left and right of the instance
            new_ts = np.pad(ts, (left, right), mode='wrap')
            # replace the zero array with the new ts
            ucr_x[i, :, j] = new_ts

    return ucr_x

def get_data():
    """
    Concatenate all labeled smartwatch data with their labels to create two sets

    Returns:
    Tuple (ndarray) -- Tuple of numpy arrays for x_train, y_train, x_test, y_test
    """
    # specify the directory and get the file_dict
    file_dict = get_files_dict()

    # initialize data and ids
    data = []
    ids = []

    features = []
    labels = []

    # features_train = []
    # features_test = []
    # labels_train = []
    # labels_test = []

    # loop over the experiments
    for experiment in file_dict.keys():
        meta_data = read_pbz2(file_dict[experiment][0])[0]
        animal_name = meta_data['animal']['animalName']

        df_1 = read_pbz2(file_dict[experiment][0])[1]
        df_2 = pd.read_csv(file_dict[experiment][1])
        df_exp = merge_dfs(df_1, df_2)
        id = animal_name

        # check if the data is not empty before appending to the list
        if len(df_exp) != 0:
            data.append(df_exp)
            ids.append(id)

    # test_id = get_test_id(ids)

    # train_data = [df for id, df in zip(ids, data) if id != test_id]
    # test_data = [df for id, df in zip(ids, data) if id == test_id]

    # standardize the data
    # first get the mean and standard deviation by dataframe in train data list
    train_means = [train_df.mean(numeric_only=True) for train_df in data]
    train_stds = [train_df.std(numeric_only=True) for train_df in data]
    # aggregate the mean and standard deviation
    train_mean = np.mean(train_means, axis=0)
    train_std = np.mean(train_stds, axis=0)

    for df in data:
        # z-normalization
        df.iloc[:, :-1] = (df.iloc[:, :-1] - train_mean) / train_std

        # retrieve instances of features and labels
        x, y = get_instances(df)
        features.extend(x)
        labels.extend(y)

    del data
    
    # for df in test_data:
    #     # z-normalization
    #     df.iloc[:, :-1] = (df.iloc[:, :-1] - train_mean) / train_std
        
    #     # retrieve instances of features and labels
    #     x, y = get_instances(df)
    #     features_test.extend(x)
    #     labels_test.extend(y)

    # calculate the maximum length of an instance in the data
    max_length = max_len(features)

    # transform each instance to the same length as the maximum using padding/wrapping
    X = transform_to_same_length(features, max_length)
    # X_test = transform_to_same_length(features_test, max_length)

    # stack the labels to form one set
    y = np.hstack(labels)
    # y_test = np.hstack(labels_test)

    del features, labels
    
    return X, y


# save data in a .npz file    
def save_data(X_train, X_valid, X_test, y_train, y_valid, y_test):
    np.savez_compressed('data.npz',
    X_train=X_train, X_valid=X_valid, X_test=X_test,
    y_train=y_train, y_valid=y_valid, y_test=y_test)


def load_data(path):
    with np.load(path, allow_pickle=True) as data:
        X_train = data['X_train']
        X_valid = data['X_valid']
        X_test = data['X_test']
        y_train = data['y_train']
        y_valid = data['y_valid']
        y_test = data['y_test']

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_dataset, valid_dataset, test_dataset
