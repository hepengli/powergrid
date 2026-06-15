"""
Oct 16, 2020
Created by Hepeng Li

Read uncertainty data
"""
import os, tree
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle

# def read_data(train=True):
#     price_path = '/home/lihepeng/Documents/Github/tmp/MG/data/price'
#     load_path = '/home/lihepeng/Documents/Github/tmp/MG/data/load'
#     renewable_path = '/home/lihepeng/Documents/Github/tmp/MG/data/renewable'
#     tdays = 21
#     if train:
#         price_files = [os.path.join(price_path, f) for f in os.listdir(price_path) if re.match(r'^2016\d+.mat$', f)]
#         price_data = [loadmat(f)['price'].transpose()[:tdays,:].ravel() for f in price_files]
#         price_data = np.maximum(np.hstack(price_data).ravel() * 0.2, 1)
#         price_data = np.minimum(price_data, 18.0)
#         price_data = np.round(price_data, 2)

#         load_files = [os.path.join(load_path, f) for f in os.listdir(load_path) if re.match(r'^2016\d+.mat$', f)]
#         load_data = [loadmat(f)['demand'].transpose()[:tdays,:].ravel() for f in load_files]
#         load_data = np.hstack(load_data).ravel() * 3.0

#         renew_files = [os.path.join(renewable_path, f) for f in os.listdir(renewable_path) if re.match(r'^2016\d+.mat$', f)]
#         solar_data = [loadmat(f)['solar_power'].transpose()[:tdays,:].ravel() for f in renew_files]
#         wind_data = [loadmat(f)['wind_power'].transpose()[:tdays,:].ravel() for f in renew_files]
#         solar_data = np.hstack(solar_data).ravel() * 6 / 1000
#         wind_data = np.hstack(wind_data).ravel() * 6 / 1000
#     else:
#         price_files = [os.path.join(price_path, f) for f in os.listdir(price_path) if re.match(r'^2016\d+.mat$', f)]
#         price_data = [loadmat(f)['price'].transpose()[tdays:,:].ravel() for f in price_files]
#         price_data = np.maximum(np.hstack(price_data).ravel() * 0.2, 1)
#         price_data = np.minimum(price_data, 18.0)
#         price_data = np.round(price_data, 3)

#         load_files = [os.path.join(load_path, f) for f in os.listdir(load_path) if re.match(r'^2016\d+.mat$', f)]
#         load_data = [loadmat(f)['demand'].transpose()[tdays:,:].ravel() for f in load_files]
#         load_data = np.hstack(load_data).ravel() * 3.0

#         renew_files = [os.path.join(renewable_path, f) for f in os.listdir(renewable_path) if re.match(r'^2016\d+.mat$', f)]
#         solar_data = [loadmat(f)['solar_power'].transpose()[tdays:,:].ravel() for f in renew_files]
#         wind_data = [loadmat(f)['wind_power'].transpose()[tdays:,:].ravel() for f in renew_files]
#         solar_data = np.hstack(solar_data).ravel() * 6 / 1000
#         wind_data = np.hstack(wind_data).ravel() * 6 / 1000

#     size = price_data.size
#     days = price_data.size // 24

#     return {'load': load_data, 'solar': solar_data, 'wind': wind_data, 'price':price_data, 'days':days, 'size':size}

# def read_pickle_data():
#     import pickle, os
#     dir_path = '/Users/hepengli/Library/CloudStorage/OneDrive-Personal/Github/powergrid/data/data2018-2020.pkl'
#     f = open(dir_path, 'rb')
#     data = pickle.load(f)
#     f.close()
#     return data


# Load
dir = os.path.join(os.getcwd(), "2023-2024", "load")
file_names = [name for name in os.listdir(dir)]
file_names.sort()

all_data = []
for file in file_names:
    df = pd.read_csv(os.path.join(dir, file))
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_all['INTERVALSTARTTIME_GMT'] = pd.to_datetime(df_all['INTERVALSTARTTIME_GMT'], utc=True)

load = {}
full_index = pd.date_range(start='2023-01-01 08:00:00', end='2025-01-01 07:00:00', freq='h', tz='UTC')
areas = df_all['TAC_AREA_NAME'].unique()
for area in areas:
    df_area = df_all[df_all['TAC_AREA_NAME'] == area].copy()
    df_area = df_area.sort_values('INTERVALSTARTTIME_GMT').set_index('INTERVALSTARTTIME_GMT')
    # Reindex to full hourly timeline and interpolate
    df_area = df_area.reindex(full_index)
    df_area['MW'] = pd.to_numeric(df_area['MW'], errors='coerce')  # Ensure numeric for interpolation
    df_area['MW'] = df_area['MW'].interpolate(method='time')
    df_area['MW'] = df_area['MW'].ffill().bfill()  # Edge fill
    mw_values = df_area['MW'].values.astype("float32")
    max_val = np.nanmax(mw_values)
    if max_val > 0 and not np.isnan(max_val):
        load[area] = mw_values / max_val
    else:
        load[area] = mw_values  # Leave unnormalized if invalid


# Solar and Wind
dir = os.path.join(os.getcwd(), "2023-2024", "solar_wind")
file_names = [name for name in os.listdir(dir)]
file_names.sort()

all_data = []
for file in file_names:
    df = pd.read_csv(os.path.join(dir, file))
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_all['INTERVALSTARTTIME_GMT'] = pd.to_datetime(df_all['INTERVALSTARTTIME_GMT'], utc=True)

solar, wind = {}, {}
full_index = pd.date_range(start='2023-01-01 08:00:00', end='2025-01-01 07:00:00', freq='h', tz='UTC')
areas = df_all['TRADING_HUB'].unique()
for area in areas:
    df_area = df_all[df_all['TRADING_HUB'] == area].copy()
    for source, target_dict in [('Solar', solar), ('Wind', wind)]:
        df_source = df_area[df_area['RENEWABLE_TYPE'] == source].copy()
        if df_source.empty:
            continue
        df_source = df_source.sort_values('INTERVALSTARTTIME_GMT').set_index('INTERVALSTARTTIME_GMT')
        df_source = df_source.reindex(full_index)
        df_source['MW'] = pd.to_numeric(df_source['MW'], errors='coerce')
        df_source['MW'] = df_source['MW'].interpolate(method='time').ffill().bfill()
        # Normalize safely
        mw_values = df_source['MW'].values.astype("float32")
        max_val = np.nanmax(mw_values)
        if max_val > 0 and not np.isnan(max_val):
            mw_values /= max_val  # normalize
        # store
        target_dict[area] = np.clip(mw_values, 0, None)



# Price
dir = os.path.join(os.getcwd(), "2023-2024", "price", "0096WD_7_N001")
file_names = [name for name in os.listdir(dir)]
file_names.sort()

all_data = []
for file in file_names:
    df = pd.read_csv(os.path.join(dir, file))
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_all['INTERVALSTARTTIME_GMT'] = pd.to_datetime(df_all['INTERVALSTARTTIME_GMT'], utc=True)

# Filter only for LMP type rows before interpolation
df_lmp = df_all[df_all['LMP_TYPE'] == 'LMP'].copy()

# Set index and reindex to full hourly time range
full_index = pd.date_range(start='2023-01-01 08:00:00', end='2025-01-01 07:00:00', freq='h', tz='UTC')
df_lmp = df_lmp.sort_values('INTERVALSTARTTIME_GMT').set_index('INTERVALSTARTTIME_GMT')
df_lmp = df_lmp.reindex(full_index)

# Clean and interpolate
df_lmp['MW'] = pd.to_numeric(df_lmp['MW'], errors='coerce')
df_lmp['MW'] = df_lmp['MW'].interpolate(method='time').ffill().bfill()

# Store in dict if needed
price_0096WD_7_N001 = df_lmp['MW'].values.astype("float32")

price = {"0096WD_7_N001": price_0096WD_7_N001}

a = {'load': load, 'solar': solar, 'wind': wind, 'price':price}
split_index = 365 * 24 # 2023: 

# Apply split to entire tree
train_data = tree.map_structure(lambda x: x[:split_index], a)
test_data = tree.map_structure(lambda x: x[split_index:], a)
dataset = {"train": train_data, "test": test_data}

with open('data2023-2024.pkl', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data2023-2024.pkl', 'rb') as file:
    d = pickle.load(file)
