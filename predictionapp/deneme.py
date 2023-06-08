from datetime import datetime, timedelta

from darts import TimeSeries
import numpy as np
import pandas as pd
import warnings


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from transparency_epias.markets import dayaheadClient
from transparency_epias.consumption import consumptionClient
from transparency_epias.production import productionClient
from transparency_epias.gas import gasTraClients
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.linear_model import LinearRegression
import catboost as cat
import xgboost as xgb
from meteostat import Hourly,Point
from geopy.geocoders import Nominatim

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


class Project492:
    holiday_path = 'C:/Users/Osman/Desktop/ptfprediction/Calendar.csv'
    usdtry_path = 'C:/Users/Osman/Desktop/ptfprediction/USD_TRY Historical Data (5).csv'
    read_from_path = True
    target = 'PTF'

    production_features = False
    consumption_features = False
    weather_features = False
    holiday_features = True
    seasonality_features = True
    seasonality_spline_features = True
    ptf_lag_variables = False
    production_lag_features = False
    consumption_lag_features = False
    rolling_shift_features = False
    meteostat_weather_features = False
    weather_lag_features = False

    weather_lag_range = np.arange(1, 21, 10)
    rolling_range = np.arange(0, 30, 10)[1:]
    roll_types = ['mean', 'std']

    production_base_features = ['fueloil', 'blackCoal', 'lignite', 'geothermal', 'naturalGas', 'river', 'dammedHydro',
                            'biomass', 'importCoal', 'asphaltiteCoal', 'wind', 'sun', 'importExport', 'wasteheat']

    # teklif_features = ['Alış Teklif Miktarı (MWh)','Satış Teklif Miktarı (MWh)']

    consumption_base_features = ['Consumption']

    ptf_base_feature = ['PTF']

    usd_try_features = ['USD_TRY_Price']

    nat_gas_features = ['Nat_Gas_Ref_Price']

    #   weather_base_features=['temp','pres']
    weather_base_features = ['temp']
    # weather_base_features=['temp','wspd','pres']

def lag_features(df_temp, columns, lags):

    """
    Lag feature üreten fonksiyon, ilgili dataframe, lag uzunlukluları ve sütun isimleri verilerek lag'li featurelar üretilir.
    """
    for col in columns:
        for lag in lags:
            df_temp[f'lag_{lag}_{col}'] = df_temp[col].shift(lag)

    return df_temp

def ptf_retriever(start_date, end_date):

    ptf_time = list(dayaheadClient.dayahead.mcp(startDate=start_date, endDate=end_date)[0])
    ptf_price = list(dayaheadClient.dayahead.mcp(startDate=start_date, endDate=end_date)[1])

    df = pd.DataFrame(list(zip(ptf_time, ptf_price)))
    df.columns = ['Time', 'PTF']

    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    df['Time'] = df['Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    df['Time'] = pd.to_datetime(df['Time'])
    df['data'] = 'train'

    return df

def ptf_predictor(start_date, end_date):

    ptf_time = list(dayaheadClient.dayahead.mcp(startDate=start_date, endDate=end_date)[0])
    # ptf_price =list(dayaheadClient.dayahead.mcp(startDate=start_date,endDate=end_date)[1])

    df = pd.DataFrame(ptf_time)
    df.columns = ['Time']
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    df['Time'] = df['Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    df['Time'] = pd.to_datetime(df['Time'])

    df['PTF'] = np.nan
    df['data'] = 'pred'

    return df

def natural_gas_price_retriever(start_date, end_date):

    gas = gasTraClients.gasClient()

    nat_gas_time = gas.price_daily(start_date, end_date)[0]
    gas_ref_price = gas.price_daily(start_date, end_date)[6]

    gasp = pd.DataFrame(list(zip(nat_gas_time, gas_ref_price)))
    gasp = gasp.rename(columns={0: 'Time', 1: 'Nat_Gas_Ref_Price'})
    gasp['Time'] = pd.to_datetime(gasp['Time'])
    gasp['Tarih_daily'] = gasp['Time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    gasp['Tarih_daily'] = pd.to_datetime(gasp['Tarih_daily'])
    gasp = gasp[['Tarih_daily', 'Nat_Gas_Ref_Price']]

    # gasp = lag_features(gasp,columns = Project492.nat_gas_features, lags = [1,2,3,4,5])

    # gasp.drop(Project492.nat_gas_features ,axis=1,inplace=True)

    return gasp

def meteostat_weather_data(point, start_date, end_date):
    """
    Retrieves weather data using meteostat api based on the parameters point (latitude, longitude),
    start date and end date.
    """

    weather = Hourly(point, start_date, end_date, timezone="Europe/Istanbul")
    weather = weather.fetch()
    weather = weather.reset_index().rename(columns={'time': 'Time'})
    weather = weather.drop(['snow', 'wpgt', 'tsun', 'coco', 'dwpt', 'wdir', 'rhum', 'prcp', 'wspd'], axis=1)
    weather['Time'] = pd.to_datetime(weather['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    return weather

def weather_data_retriever(start_date, end_date):

    # cities = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Konya', 'Gaziantep', 'Kocaeli', 'Mersin', 'Kayseri']

    cities = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Konya', 'Gaziantep', 'Kocaeli', 'Kayseri']

    start_dated = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_dated = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    merged_data = pd.DataFrame()

    # Create an instance of the geocoder
    geolocator = Nominatim(user_agent="492")

    for city in cities:
        # Retrieve the latitude and longitude values using geocode
        location = geolocator.geocode(city)
        latitude = location.latitude
        longitude = location.longitude

        # Create a meteostat Point object with latitude and longitude
        point = Point(latitude, longitude)

        # Call the meteostat_weather_data function for each city
        city_weather = meteostat_weather_data(point, start_dated, end_dated)

        # Add city prefix to each column name except 'Time'
        city_weather = city_weather.rename(columns=lambda x: f'{city}_{x}' if x != 'Time' else x)

        # Merge the city_weather DataFrame with the merged_data DataFrame on the 'Time' column
        if merged_data.empty:
            merged_data = city_weather
        else:
            merged_data = merged_data.merge(city_weather, on='Time', how='left')

    merged_data['Time'] = pd.to_datetime(merged_data['Time'])

    return merged_data

def usdtry_data_retriever():

    data = pd.read_csv(Project492.usdtry_path)
    df = pd.DataFrame(data)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Tarih_daily'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['Tarih_daily'] = pd.to_datetime(df['Tarih_daily'])
    df = df[['Tarih_daily', 'Price']]
    df = df.rename(columns={'Price': 'USD_TRY_Price'})

    df = df.sort_values(by='Tarih_daily', ascending=True)
    df = df.reset_index(drop=True)

    #       df_cons = lag_features(df,columns = Project492.usd_try_features ,lags = [])

    #       df_cons.drop(Project492.usd_try_features,axis=1,inplace=True)

    return df

def consumption_retriever(start_date, end_date):

    consumption_time = list(
        consumptionClient.consumption.consumption_realtime(startDate=start_date, endDate=end_date)[0])
    consumption_amount = list(
        consumptionClient.consumption.consumption_realtime(startDate=start_date, endDate=end_date)[1])

    df = pd.DataFrame(list(zip(consumption_time, consumption_amount)))
    df.columns = ['Time', 'Consumption']

    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    df['Time'] = df['Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    df['Time'] = pd.to_datetime(df['Time'])
    #        df['data'] = 'train'

    return df

def production_retriever(start_date, end_date):

    proddf = pd.DataFrame(productionClient.production.real_time_gen(startDate=start_date, endDate=end_date))
    proddf['Time'] = pd.to_datetime(proddf['date'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    proddf['Time'] = proddf['Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    proddf['Time'] = pd.to_datetime(proddf['Time'])
    proddf = proddf.drop(['date', 'naphta', 'nucklear', 'lng', 'gasOil', 'total'], axis=1)

    return proddf

def holiday_features(path):
    """
    Tatil günlerinden türetilmiş featureları üretir.

    """

    hol = pd.read_csv(path, parse_dates=['CALENDAR_DATE'])
    hol = hol[['CALENDAR_DATE', 'WEEKEND_FLAG', 'RAMADAN_FLAG', 'RELIGIOUS_DAY_FLAG_SK', 'NATIONAL_DAY_FLAG_SK',
               'PUBLIC_HOLIDAY_FLAG']].rename(columns={'CALENDAR_DATE': 'Tarih_daily'})
    hol = hol.sort_values('Tarih_daily').reset_index(drop=True)
    hol.loc[hol.RELIGIOUS_DAY_FLAG_SK != 100, 'RELIGIOUS_DAY_FLAG_SK'] = 1
    hol.loc[hol.RELIGIOUS_DAY_FLAG_SK == 100, 'RELIGIOUS_DAY_FLAG_SK'] = 0

    hol.loc[hol.NATIONAL_DAY_FLAG_SK != 200, 'NATIONAL_DAY_FLAG_SK'] = 1
    hol.loc[hol.NATIONAL_DAY_FLAG_SK == 200, 'NATIONAL_DAY_FLAG_SK'] = 0

    hol.loc[hol.RAMADAN_FLAG == 'N', 'RAMADAN_FLAG'] = 0
    hol.loc[hol.RAMADAN_FLAG == 'Y', 'RAMADAN_FLAG'] = 1

    hol.loc[hol.PUBLIC_HOLIDAY_FLAG == 'N', 'PUBLIC_HOLIDAY_FLAG'] = 0
    hol.loc[hol.PUBLIC_HOLIDAY_FLAG == 'Y', 'PUBLIC_HOLIDAY_FLAG'] = 1

    hol.loc[hol.WEEKEND_FLAG == 'N', 'WEEKEND_FLAG'] = 0
    hol.loc[hol.WEEKEND_FLAG == 'Y', 'WEEKEND_FLAG'] = 1

    hol['WEEKEND_FLAG'] = hol['WEEKEND_FLAG'].astype(int)
    hol['RAMADAN_FLAG'] = hol['RAMADAN_FLAG'].astype(int)
    hol['PUBLIC_HOLIDAY_FLAG'] = hol['PUBLIC_HOLIDAY_FLAG'].astype(int)
    # Resmi/Dini/Milli Bayram ve Tatilleri önceden bildiren featurelar
    #    is_next_days_cols = ['RAMADAN_FLAG','PUBLIC_HOLIDAY_FLAG','RELIGIOUS_DAY_FLAG_SK','NATIONAL_DAY_FLAG_SK']
    #    for i in [3,7]:
    #       for col in is_next_days_cols:
    #          hol[f"is_{col}_in_next_{i}_days"] = hol[col].rolling(i).sum().shift(-i)

    #    hol_features = hol.columns
    return hol

def datetime_features(df_temp):
    """
    Datetime feature üretir.
    """
    df_temp['month'] = df_temp['Time'].dt.month
    df_temp['hour'] = df_temp['Time'].dt.hour
    df_temp['year'] = df_temp['Time'].dt.year
    df_temp['dayofweek'] = df_temp['Time'].dt.dayofweek
    df_temp['quarter'] = df_temp['Time'].dt.quarter
    df_temp['day'] = df_temp['Time'].dt.day
    df_temp['weekofyear'] = df_temp['Time'].dt.weekofyear

    return df_temp

def periodic_spline_transformer(period, n_splines=None, degree=3):
    """
    Kaynak: https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
    """

    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True)

def seasonality_features(df_temp):

    df_temp['month_sin'] = np.sin(2 * np.pi * df_temp.month / 12)
    df_temp['month_cos'] = np.cos(2 * np.pi * df_temp.month / 12)
    df_temp['day_sin'] = np.sin(2 * np.pi * df_temp.hour / 24)
    df_temp['day_cos'] = np.cos(2 * np.pi * df_temp.hour / 24)
    return df_temp

def seasonality_spline_features(hours=np.arange(0, 24)):

    hour_df = pd.DataFrame(np.linspace(0, 24, 24).reshape(-1, 1), columns=["hour"])
    splines = periodic_spline_transformer(24, n_splines=12).fit_transform(hour_df)
    splines_df = pd.DataFrame(splines, columns=[f"spline_{i}" for i in range(splines.shape[1])])
    splines_df = pd.concat([pd.Series(hours, name='hour'), splines_df], axis="columns")

    return splines_df

# rolling shift modelde kullanilmadi
def rolling_shift_features(df_temp, columns, rolls, roll_types):
    """
    24 lagli rolling feature üreten fonksiyon, ilgili dataframe, rolling type'ları ve sütun isimleri verilerek rolling featurelar üretilir.
    """
    for col in columns:
        for roll in rolls:
            if 'mean' in roll_types:
                df_temp[f'rolling_shift_24_mean_{roll}_{col}'] = df_temp[col].shift(24).rolling(roll,
                                                                                                min_periods=1).mean().reset_index(
                    drop=True)
            # if 'max' in roll_types:
            #    df_temp[f'rolling_shift_24_max_{roll}_{col}'] = df_temp[col].shift(24).rolling(roll,min_periods=1).max().reset_index(drop=True)
            # if 'min' in roll_types:
            #    df_temp[f'rolling_shift_24_min_{roll}_{col}'] = df_temp[col].shift(24).rolling(roll,min_periods=1).min().reset_index(drop=True)
            if 'std' in roll_types:
                df_temp[f'rolling_shift_24_std_{roll}_{col}'] = df_temp[col].shift(24).rolling(roll,
                                                                                               min_periods=1).std().reset_index(
                    drop=True)

    return df_temp


def data_featuring(start_date_train, end_date_train):
    start_date = start_date_train[:10]
    end_date = end_date_train[:10]

    df = pd.DataFrame()

    df = ptf_retriever(start_date, end_date)

    if Project492.ptf_lag_variables:
        df = lag_features(df, columns=Project492.ptf_base_feature,
                          lags=[24])
        df['Time'] = pd.to_datetime(df['Time'])

    if Project492.production_features:

        proddf = production_retriever(start_date, end_date)

        if Project492.production_lag_features:
            proddf = lag_features(proddf,
                                  columns=Project492.production_base_features,
                                  lags=[24])
        proddf.drop(Project492.production_base_features, axis=1, inplace=True)

        proddf['Time'] = pd.to_datetime(proddf['Time'])

        df = df.merge(proddf, on='Time', how='left')

    if Project492.consumption_features:

        df_cons = consumption_retriever(start_date, end_date)

        if Project492.consumption_lag_features:
            df_cons = lag_features(df_cons,
                                   columns=Project492.consumption_base_features,
                                   lags=[24])
        df_cons.drop(Project492.consumption_base_features, axis=1, inplace=True)

        df_cons['Time'] = pd.to_datetime(df_cons['Time'])

        df = df.merge(df_cons, on='Time', how='left')

    df['Tarih_daily'] = df['Time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['Tarih_daily'] = pd.to_datetime(df['Tarih_daily'])

    df = datetime_features(df)

    if Project492.seasonality_features:
        #        df_w_season = seasonality_features(df_w_dt)
        df = seasonality_features(df)

    if Project492.seasonality_spline_features:
        splines_df = seasonality_spline_features()
        df = df.merge(splines_df, on='hour', how='left')

    if Project492.holiday_features:
        hol = holiday_features(Project492.holiday_path)
        df = pd.merge(df, hol, on='Tarih_daily')

    if Project492.meteostat_weather_features:

        merged_data = weather_data_retriever(start_date_train, end_date_train)

        if Project492.weather_lag_features:
            columns = merged_data.columns[1:]

            merged_data = lag_features(merged_data,
                                       columns=columns,
                                       lags=Project492.weather_lag_range)

        df = df.merge(merged_data, on='Time', how='left')

    return df


def data_featuring_predictor(start_date_train, end_date_train):
    time_range = pd.date_range(start=start_date_train, end=end_date_train, freq='H')

    df = pd.DataFrame({'Time': time_range})
    df['PTF'] = np.nan
    df['data'] = 'pred'

    start_date = start_date_train[:10]
    end_date = end_date_train[:10]

    if Project492.ptf_lag_variables:
        df = lag_features(df, columns=Project492.ptf_base_feature,
                          lags=[24])
        df['Time'] = pd.to_datetime(df['Time'])

    if Project492.production_features:

        proddf = production_retriever(start_date, end_date)

        if Project492.production_lag_features:
            proddf = lag_features(proddf,
                                  columns=Project492.production_base_features,
                                  lags=[24])
        proddf.drop(Project492.production_base_features, axis=1, inplace=True)

        proddf['Time'] = pd.to_datetime(proddf['Time'])

        df = df.merge(proddf, on='Time', how='left')

    if Project492.consumption_features:

        df_cons = consumption_retriever(start_date, end_date)

        if Project492.consumption_lag_features:
            df_cons = lag_features(df_cons,
                                   columns=Project492.consumption_base_features,
                                   lags=[24])
        df_cons.drop(Project492.consumption_base_features, axis=1, inplace=True)

        df_cons['Time'] = pd.to_datetime(df_cons['Time'])

        df = df.merge(df_cons, on='Time', how='left')

    df['Tarih_daily'] = df['Time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['Tarih_daily'] = pd.to_datetime(df['Tarih_daily'])

    df = datetime_features(df)

    if Project492.seasonality_features:
        #        df_w_season = seasonality_features(df_w_dt)
        df = seasonality_features(df)

    if Project492.seasonality_spline_features:
        splines_df = seasonality_spline_features()
        df = df.merge(splines_df, on='hour', how='left')

    if Project492.holiday_features:
        hol = holiday_features(Project492.holiday_path)
        df = pd.merge(df, hol, on='Tarih_daily')

    if Project492.meteostat_weather_features:

        merged_data = weather_data_retriever(start_date_train, end_date_train)

        if Project492.weather_lag_features:
            columns = merged_data.columns[1:]

            merged_data = lag_features(merged_data,
                                       columns=columns,
                                       lags=Project492.weather_lag_range)

        df = df.merge(merged_data, on='Time', how='left')

    return df


# You can change the duration of the overall dataset by adjusting start_date_train and end_date_train.Pay attention to type it in correct format '%Y-%m-%d %H:%M:%S'

today = datetime.now()

date_time = today.strftime("%Y-%m-%d 23:00:00")

date_time = str(date_time)

start_date_train = '2021-01-01 00:00:00'
end_date_train = date_time

# call data_featuring func to create historical dataframe

train_df_all = data_featuring(start_date_train,end_date_train)

# this block is for the creation of next 24 hourse prediction dataframe which has exactly same column structure only difference data = pred , PTF =NaN and Time is the dynamic next 24 hours from the end_date_train


#pred_next_hours = 24

end_date_train = end_date_train
hour = 23
end_date_train = pd.to_datetime(end_date_train)
starting_time = end_date_train + pd.DateOffset(hours=1)
starting_time = starting_time.replace(minute=0, second=0)
end_date_predict = starting_time + pd.DateOffset(hours=hour)
start_date_predict_str = starting_time.strftime('%Y-%m-%d %H:%M:%S')
end_date_predict_str = end_date_predict.strftime('%Y-%m-%d %H:%M:%S')

pred_df_all = data_featuring_predictor(start_date_predict_str,end_date_predict_str)

# NATGAS PRICES RETRIEVEL FROM EPIAS LIBRARY

start_date = start_date_train[:10]
end_date = end_date_predict_str[:10]

nat_gas_df = natural_gas_price_retriever(start_date,end_date)

# USDTRY PRICES RETRIEVEL FROM CSV FILE
usd_try_price = usdtry_data_retriever()

usd_try_price = usd_try_price.drop_duplicates(subset=['Tarih_daily'])

# CONCETANATION OF HISTORIC DATA and PREDICT df

dataframe_final = pd.concat([train_df_all,pred_df_all], ignore_index=True)

#NATGAS MERGING WITH MAIN DF
dataframe_final = dataframe_final.merge(nat_gas_df, on='Tarih_daily', how='left')

# USDTRY MERGING WITH MAIN DF
dataframe_final = dataframe_final.merge(usd_try_price, on='Tarih_daily', how='left')

# ADDITION OF 2 COLUMNS Nat_Gas_Ref_Usd_try_ratio AND is_business_time.Additionally the filling of the non existent 24 hours natgas price and usdtry price value in preddf with the recent value
#

dataframe_final['USD_TRY_Price'].fillna(method='ffill', inplace=True)
dataframe_final['Nat_Gas_Ref_Price'].fillna(method='ffill', inplace=True)

# dataframe_final['lag_1_Nat_Gas_Ref_Price'] = dataframe_final['lag_1_Nat_Gas_Ref_Price'].fillna(method='bfill')

dataframe_final['Nat_Gas_Ref_Usd_try_ratio'] = dataframe_final['Nat_Gas_Ref_Price'] / dataframe_final['USD_TRY_Price']
dataframe_final['is_business_time'] = np.where(
    (dataframe_final["Time"].dt.dayofweek < 5) & (dataframe_final["hour"] >= 9) & (dataframe_final["hour"] < 18), 1, 0)
# dataframe_final['season'] = dataframe_final['Time'].dt.month.apply(lambda x: 1 if x in [1, 2, 12] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4)


# dataframe_final.set_index('Time', inplace=True)

#CUT OF FOR NAN VALUES


start_date_train = pd.to_datetime(start_date_train)

# Add 2 days to the start_date_train
cutoffforNanvalues_date = start_date_train + pd.DateOffset(days=2)

# Convert the new_date back to string format
cutoffforNanvalues_date = cutoffforNanvalues_date.strftime('%Y-%m-%d %H:%M:%S')
#dataframe_final['Time'] = dataframe_final.index

dataframe_final = dataframe_final[dataframe_final['Time'] >= cutoffforNanvalues_date]
dataframe_final = dataframe_final.reset_index(drop=True)

train_df = dataframe_final[dataframe_final.data=='train']
pred_df = dataframe_final[dataframe_final.data=='pred']
exclude_cols = ['Time','data','year', 'Tarih_daily']

#train_df.dropna(inplace=True)

tarih = pred_df['Time']

pred_df = pred_df.drop(exclude_cols,axis=1)
y = train_df[Project492.target]
X = train_df.drop(exclude_cols,axis=1)

tss = TimeSeriesSplit(n_splits=3, test_size=24*5*1)

catboost_models = []
fold = 0
cat_scores = []

for train_idx, val_idx in tss.split(X):
    train_data = X.iloc[train_idx]
    test_data = X.iloc[val_idx]

    X_train, y_train = train_data.drop('PTF', axis=1), train_data['PTF']
    X_val, y_val = test_data.drop('PTF', axis=1), test_data['PTF']

    #    print(X_train)

    feature_cov_train = TimeSeries.from_series(X_train)
    y_train_ts = TimeSeries.from_series(y_train)
    feature_cov_test = TimeSeries.from_series(X_val)

    # features = ['PTF','Nat_Gas_Ref_Price']

    # future_cov = TimeSeries.from_dataframe(X[features])

    # model_catboost = RegressionModel(lags=[-36, -72, -128, -256, -400, - 512, -724, -1024],
    model_catboost = RegressionModel(lags=[-24, -48, -72, -96, -120, -144, -256, -400, -512, -724, -1024],
                                     lags_future_covariates=[0],
                                     model=cat.CatBoostRegressor(n_estimators=1000,
                                                                 verbose=200,
                                                                 random_state=42,
                                                                 eval_metric='MAPE'))

    # print(lags_future_covariates)

    print(f'{fold + 1}. Fold Training... ')
    fold += 1
    model_catboost.fit(y_train_ts, future_covariates=feature_cov_train)
    catboost_models.append(model_catboost)
    # pred_log = model_catboost.predict(n=X_val.shape[0], series=y_train_ts, future_covariates=feature_cov_test)
    # pred = np.exp(pred_log)  # Reverse the logarithm transformation

    pred = model_catboost.predict(n=X_val.shape[0], series=y_train_ts, future_covariates=feature_cov_test)
    score = mean_absolute_percentage_error(y_val, pred.values())
    cat_scores.append(score)
    avg_cat = np.average(cat_scores)
    del train_data, test_data, model_catboost, X_train, y_train, X_val, y_val

    print(f'Test score => {score}')

print(f'CATBOOST CV score => {avg_cat}')

test = pred_df.drop('PTF', axis=1)

y_ts = TimeSeries.from_series(X['PTF'])
feature_cov_test_data = TimeSeries.from_series(test)
cat_preds_dart = [(model.predict(n=pred_df.shape[0], series=y_ts, future_covariates=feature_cov_test_data)).values() for model in catboost_models]
cat_preds_dart = np.mean(cat_preds_dart, axis = 0)

pred_df['PTF'] = cat_preds_dart
pred_df['Tarih'] = tarih
pred_df


fig, ax = plt.subplots(figsize=(15, 5))
pred_df.set_index('Tarih')['PTF'].plot(ax=ax, title='Tahmin')
plt.show()
