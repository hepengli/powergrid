"""
Oct 16, 2020
Created by Hepeng Li

Read uncertainty data
"""
import os, re
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle

def read_data(train=True):
    price_path = '/home/lihepeng/Documents/Github/tmp/MG/data/price'
    load_path = '/home/lihepeng/Documents/Github/tmp/MG/data/load'
    renewable_path = '/home/lihepeng/Documents/Github/tmp/MG/data/renewable'
    tdays = 21
    if train:
        price_files = [os.path.join(price_path, f) for f in os.listdir(price_path) if re.match(r'^2016\d+.mat$', f)]
        price_data = [loadmat(f)['price'].transpose()[:tdays,:].ravel() for f in price_files]
        price_data = np.maximum(np.hstack(price_data).ravel() * 0.2, 1)
        price_data = np.minimum(price_data, 18.0)
        price_data = np.round(price_data, 2)

        load_files = [os.path.join(load_path, f) for f in os.listdir(load_path) if re.match(r'^2016\d+.mat$', f)]
        load_data = [loadmat(f)['demand'].transpose()[:tdays,:].ravel() for f in load_files]
        load_data = np.hstack(load_data).ravel() * 3.0

        renew_files = [os.path.join(renewable_path, f) for f in os.listdir(renewable_path) if re.match(r'^2016\d+.mat$', f)]
        solar_data = [loadmat(f)['solar_power'].transpose()[:tdays,:].ravel() for f in renew_files]
        wind_data = [loadmat(f)['wind_power'].transpose()[:tdays,:].ravel() for f in renew_files]
        solar_data = np.hstack(solar_data).ravel() * 6 / 1000
        wind_data = np.hstack(wind_data).ravel() * 6 / 1000
    else:
        price_files = [os.path.join(price_path, f) for f in os.listdir(price_path) if re.match(r'^2016\d+.mat$', f)]
        price_data = [loadmat(f)['price'].transpose()[tdays:,:].ravel() for f in price_files]
        price_data = np.maximum(np.hstack(price_data).ravel() * 0.2, 1)
        price_data = np.minimum(price_data, 18.0)
        price_data = np.round(price_data, 3)

        load_files = [os.path.join(load_path, f) for f in os.listdir(load_path) if re.match(r'^2016\d+.mat$', f)]
        load_data = [loadmat(f)['demand'].transpose()[tdays:,:].ravel() for f in load_files]
        load_data = np.hstack(load_data).ravel() * 3.0

        renew_files = [os.path.join(renewable_path, f) for f in os.listdir(renewable_path) if re.match(r'^2016\d+.mat$', f)]
        solar_data = [loadmat(f)['solar_power'].transpose()[tdays:,:].ravel() for f in renew_files]
        wind_data = [loadmat(f)['wind_power'].transpose()[tdays:,:].ravel() for f in renew_files]
        solar_data = np.hstack(solar_data).ravel() * 6 / 1000
        wind_data = np.hstack(wind_data).ravel() * 6 / 1000

    size = price_data.size
    days = price_data.size // 24

    return {'load': load_data, 'solar': solar_data, 'wind': wind_data, 'price':price_data, 'days':days, 'size':size}

def read_pickle_data():
    import pickle, os
    dir_path = '/Users/hepengli/Library/CloudStorage/OneDrive-Personal/Github/powergrid/data/data2018-2020.pkl'
    f = open(dir_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


# df_load = pd.read_csv('./2024/load_Jan.csv')
# load = {}
# areas = df_load['TAC_AREA_NAME'].unique()
# for area in areas:
#     df = df_load[df_load['TAC_AREA_NAME'] == area]
#     df = df.sort_values('INTERVALSTARTTIME_GMT')
#     df = df[df['INTERVALSTARTTIME_GMT'] >= '2024-01-02T00:00:00-00:00']
#     df = df[df['INTERVALSTARTTIME_GMT'] < '2024-02-01T00:00:00-00:00']
#     load[area] = df['MW'].values / df['MW'].values.max()

# df_res = pd.read_csv('./2024/renewable_Jan.csv')
# solar, wind = {}, {}
# areas = df_res['TRADING_HUB'].unique()
# for area in areas:
#     df = df_res[df_res['TRADING_HUB'] == area]
#     df_solar = df[df['RENEWABLE_TYPE'] == 'Solar']
#     df_solar = df_solar.sort_values('INTERVALSTARTTIME_GMT')
#     df_solar = df_solar[df_solar['INTERVALSTARTTIME_GMT'] >= '2024-01-02T00:00:00-00:00']
#     df_solar = df_solar[df_solar['INTERVALSTARTTIME_GMT'] <= '2024-02-01T00:00:00-00:00']
#     solar[area] = df_solar['MW'].values / df_solar['MW'].values.max()
#     df = df_res[df_res['TRADING_HUB'] == area]
#     df_wind = df[df['RENEWABLE_TYPE'] == 'Wind']
#     df_wind = df_wind.sort_values('INTERVALSTARTTIME_GMT')
#     df_wind = df_wind[df_wind['INTERVALSTARTTIME_GMT'] >= '2024-01-02T00:00:00-00:00']
#     df_wind = df_wind[df_wind['INTERVALSTARTTIME_GMT'] <= '2024-02-01T00:00:00-00:00']
#     try:
#         wind[area] = df_wind['MW'].values / df_wind['MW'].values.max()
#     except:
#         pass

# df_price = pd.read_csv('./2024/price_Jan.csv')
# price = {}
# df = df_price[df_price['LMP_TYPE'] == 'LMP']
# df = df.sort_values('INTERVALSTARTTIME_GMT')
# df = df[df['INTERVALSTARTTIME_GMT'] >= '2024-01-02T00:00:00-00:00']
# df = df[df['INTERVALSTARTTIME_GMT'] < '2024-02-01T00:00:00-00:00']
# price['LMP'] = df['MW'].values

# a = {'load': load, 'solar': solar, 'wind': wind, 'price':price}
# with open('data2024.pkl', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data2024.pkl', 'rb') as file:
#     d = pickle.load(file)
