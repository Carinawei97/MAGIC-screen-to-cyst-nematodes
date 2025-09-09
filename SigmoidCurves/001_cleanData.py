#!/usr/bin/env python

# coding: utf-8



import os

import glob

import warnings

from concurrent.futures import ThreadPoolExecutor

from datetime import datetime



import pandas as pd

import numpy as np



warnings.simplefilter('always', category=UserWarning)



# ------------------------------

# Paths and working directory

# ------------------------------

path = r'/media/o/LabBookMirror2/backup/MainOld/Main1/Siyuan/AnalysisJi/Analysis divided by Day 38/38dpi/Denoised traits'

os.chdir(path)

cwd = os.getcwd()

subfolders = [folder for folder in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, folder))]



# ------------------------------

# Placeholder plotting functions

# ------------------------------

def displot(df, xax, outname):

    pass



def ScatterDish(merged_df_loc, outname, name):

    pass



def CurvePlot(plot_data, name, outname):

    pass



def ViolinplotHueID(plot_data, name, outname):

    pass



def distPerHue(plot_data, name, outname):

    pass



# ------------------------------

# CSV reading functions

# ------------------------------

def read_csv_loc(file_path):

    df = pd.read_csv(file_path)

    df_long = df.melt(id_vars=["Date"], var_name="ID", value_name="Value")

    df_long[['X', 'Y']] = df_long['Value'].str.extract(r'\(([^,]+), ([^,]+)\)')

    df_long['X'] = df_long['X'].combine_first(df_long['Value'])

    df_long['Y'] = df_long['Y'].combine_first(df_long['Value'])

    df_long['Value'].replace('OUT', np.nan, inplace=True)

    df_long['Date'] = pd.to_datetime(df_long['Date'])

    min_date = df_long['Date'].min()

    df_long['Relative_Days'] = (df_long['Date'] - min_date).dt.days

    df_long.drop(columns=['Value'], inplace=True)

    df_long['name'] = os.path.basename(os.path.dirname(file_path))

    df_long['X'] = pd.to_numeric(df_long['X'], errors='coerce')

    df_long['Y'] = pd.to_numeric(df_long['Y'], errors='coerce')

    return df_long



def read_csv_size(file_path):

    df = pd.read_csv(file_path)

    df_long = df.melt(id_vars=["Date"], var_name="ID", value_name="Value")

    df_long['Value'].replace('OUT', np.nan, inplace=True)

    df_long['Date'] = pd.to_datetime(df_long['Date'])

    min_date = df_long['Date'].min()

    df_long['Relative_Days'] = (df_long['Date'] - min_date).dt.days

    df_long['name'] = os.path.basename(os.path.dirname(file_path))

    df_long['Size'] = pd.to_numeric(df_long['Value'], errors='coerce')

    df_long.drop(columns=['Value'], inplace=True)

    return df_long



# ------------------------------

# Read CSVs using multithreading

# ------------------------------

def read_folder_csvs(folder):

    loc_path = os.path.join(cwd, folder, 'Location.csv')

    size_path = os.path.join(cwd, folder, 'Size.csv')

    loc_df = read_csv_loc(loc_path)

    size_df = read_csv_size(size_path)

    return pd.merge(loc_df, size_df, on=['Date', 'ID', 'name', 'Relative_Days'])



with ThreadPoolExecutor() as executor:

    dfs = list(executor.map(read_folder_csvs, subfolders))



merged_df = pd.concat(dfs, ignore_index=True)



merged_df.loc[merged_df['Date'] == '2022-04-07', 'Date'] = '2022-04-08'

merged_df = merged_df[merged_df.Date != '2022-05-06']



center_x, center_y = merged_df['X'].mean(), merged_df['Y'].mean()



# change to cleaning data path

path = r'/media/o/LabBookMirror2/backup/MainOld/Main1/Siyuan/AnalysisJi/CleaningData'

os.chdir(path)



warnings.filterwarnings('ignore')

displot(merged_df, 'Size', '000DistBeforeMessingWithData')



# ------------------------------

# Filtering helper functions (signature: repName, df, threshold, fulldf)

# ------------------------------

def removeIfManyNANinARow(repName, df, numberOfNANThreshold, fulldf):

    IndexList = []

    toads = df.ID.unique()

    for nem in toads:

        OneToad = df[df.ID == nem].sort_values(by='Relative_Days')

        DaySevenOnwards = OneToad[OneToad.Relative_Days >= 7]

        try:

            nan_mask = DaySevenOnwards['Size'].isna()

            max_consecutive_nans = 0

            current_length = 0

            for value in nan_mask:

                if value:

                    current_length += 1

                else:

                    max_consecutive_nans = max(max_consecutive_nans, current_length)

                    current_length = 0

            max_consecutive_nans = max(max_consecutive_nans, current_length)

        except:

            df_filteredIndex = fulldf[(fulldf['ID'] == nem) & (fulldf['name'] == repName)].index

            IndexList.append(df_filteredIndex)

            continue

        if max_consecutive_nans > numberOfNANThreshold:

            df_filteredIndex = fulldf[(fulldf['ID'] == nem) & (fulldf['name'] == repName)].index

            IndexList.append(df_filteredIndex)

    return IndexList



def removeProportionNAN(repName, df, numberOfNANThreshold, fulldf):

    IndexList = []

    toads = df.ID.unique()

    for nem in toads:

        OneToad = df[df.ID == nem].sort_values(by='Relative_Days')

        DaySevenOnwards = OneToad[OneToad.Relative_Days >= 7]

        try:

            max_index_for_ID = DaySevenOnwards.index.max()

            first_valid_index = DaySevenOnwards['Size'].first_valid_index()

            FilterOutBelowIndex = DaySevenOnwards.loc[first_valid_index:max_index_for_ID]

            num_nans_ID = FilterOutBelowIndex['Size'].isna().sum() / DaySevenOnwards.shape[0]

        except:

            df_filteredIndex = fulldf[(fulldf['ID'] == nem) & (fulldf['name'] == repName)].index

            IndexList.append(df_filteredIndex)

            continue

        if num_nans_ID > numberOfNANThreshold:

            df_filteredIndex = fulldf[(fulldf['ID'] == nem) & (fulldf['name'] == repName)].index

            IndexList.append(df_filteredIndex)

    return IndexList



def RMNoise(repName, df, numberOfRepsThreshold, fulldf):

    IndexList = []

    toads = df.ID.unique()

    for nem in toads:

        changed = True

        templist = []

        count_above_threshold = 0

        while changed:

            changed = False

            OneToad = df[df.ID == nem].sort_values(by='Relative_Days')

            non_nan_indices = OneToad['Size'].dropna().index

            for i in range(1, len(non_nan_indices) - 1):

                prev_index = non_nan_indices[i - 1]

                curr_index = non_nan_indices[i]

                next_index = non_nan_indices[i + 1]

                prev_value = OneToad.loc[prev_index, 'Size']

                curr_value = OneToad.loc[curr_index, 'Size']

                next_value = OneToad.loc[next_index, 'Size']

                if curr_value > prev_value and curr_value < next_value:

                    continue

                avg_before_after = (prev_value + next_value) / 2

                if (curr_value > 1.5 * avg_before_after) or (1.5 * curr_value < avg_before_after):

                    df.drop(curr_index, inplace=True)

                    templist.append(curr_index)

                    changed = True

                    count_above_threshold += 1

                    if count_above_threshold > 6:

                        df_filteredIndex = fulldf[(fulldf['ID'] == nem) & (fulldf['name'] == repName)].index

                        IndexList.append(df_filteredIndex)

                        changed = False

        if templist:

            IndexList += templist

    return IndexList



# ------------------------------

# Apply filters in parallel (per subfolder)

# ------------------------------

def apply_filter(func, df, threshold):

    results = []

    with ThreadPoolExecutor() as executor:

        futures = [executor.submit(func, sub, df[df.name == sub], threshold, df) for sub in subfolders]

        for f in futures:

            results += f.result()

    return results



print(len(merged_df))



indexlist = apply_filter(removeIfManyNANinARow, merged_df, 5)

for i in indexlist:

    merged_df.loc[i, 'Size'] = np.nan

merged_df.reset_index(drop=True, inplace=True)

displot(merged_df, 'Size', '001DistAfterFilteringConsecutiveNAN')



indexlist = apply_filter(removeProportionNAN, merged_df, 0.3)

for i in indexlist:

    merged_df.loc[i, 'Size'] = np.nan

merged_df.reset_index(drop=True, inplace=True)

displot(merged_df, 'Size', '002DistAfterFilteringpropotionNAN')



# ------------------------------

# Noise removal

# ------------------------------

print(len(merged_df))

passListTemp = []

names = (merged_df.name.unique())

lennames = len(merged_df.name.unique())

IDs = [len(merged_df[merged_df.name == sub].ID.unique()) for sub in names]

IDssum = sum(IDs)

print(IDssum)



indexlist = apply_filter(RMNoise, merged_df, 25)

for i in indexlist:

    merged_df.loc[i, 'Size'] = np.nan

merged_df.reset_index(drop=True, inplace=True)



names = (merged_df.name.unique())

lennames = len(merged_df.name.unique())

IDs = [len(merged_df[merged_df.name == sub].ID.unique()) for sub in names]

IDssum = sum(IDs)

print(IDssum)

displot(merged_df, 'Size', '003DistAfterAfterRemovingOutliers')



# ------------------------------

# Final adjustments and save

# ------------------------------

merged_df['name_ID'] = merged_df['name'] + ' (' + merged_df['ID'].astype(str) + ')'

merged_df = merged_df[(merged_df['ID'] != 'average') & (merged_df['ID'] != 'median')]



df = merged_df

resultmean = df.groupby(['Date'])['Size'].mean().reset_index()

resultmed = df.groupby(['Date'])['Size'].median().reset_index()



path = r'/media/o/LabBookMirror2/backup/MainOld/Main1/Siyuan/AnalysisJi/scriptsOlaf'

os.chdir(path)

merged_df.to_csv('Merged_Cleaned_outlier_data.csv')



# preserve some original debug/check prints

tempdf = merged_df.dropna(inplace=False, subset=['Size'])

tempdf2 = tempdf[tempdf.Date == '2022-05-06']

names = (tempdf2.name.unique())

lennames = len(tempdf2.name.unique())

IDs = [len(tempdf2[tempdf2.name == sub].ID.unique()) for sub in names]

IDssum = sum(IDs)

print(IDssum)



print(resultmean)



# extra checks from original script

IDssum = sum(IDs)

print(IDssum - (lennames * 2))

print(len(merged_df.name_ID.unique()) - (lennames * 2))



# sample filtering example

id_counts = merged_df['ID'].value_counts()

filtered_ids = id_counts[id_counts > 5].index

filtered_df = merged_df[merged_df['ID'].isin(filtered_ids)]

print(filtered_df)

