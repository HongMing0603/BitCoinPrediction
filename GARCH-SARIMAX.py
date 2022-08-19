# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters 
import arch
register_matplotlib_converters()
# %matplotlib inline
import warnings
import requests
warnings.filterwarnings("ignore")

# # import dataset
# url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
# param = {"convert":"USD","slug":"bitcoin","time_end":"1601510400","time_start":"1367107200"}
# content = requests.get(url=url, params=param).json()
# df = pd.json_normalize(content['data']['quotes'])

# Fetching data from the server
url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
# param = {"convert":"USD","slug":"bitcoin","time_end":"1601510400","time_start":"1367107200"}
param = {"convert":"USD","slug":"bitcoin","time_end":"1658275200","time_start":"1367107200"}

content = requests.get(url=url, params=param).json()
df = pd.json_normalize(content['data']['quotes'])

# selecting useful columns
df['Date']=pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)
df['Low'] = df['quote.USD.low']
df['High'] = df['quote.USD.high']
df['Open'] = df['quote.USD.open']
df['Close'] = df['quote.USD.close']
df['Volume'] = df['quote.USD.volume']

# dropping unused columns
df=df.drop(columns=['time_open','time_close','time_high','time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open', 'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])

# feature creation for better representation of price on a day
df['Mean'] = (df['Low'] + df['High'])/2

# remove NaNs and Nones
df = df.dropna()

# data preview
print(df.head())


# making copy for making changes
dataset_for_prediction = df.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['Mean'].shift()
dataset_for_prediction=dataset_for_prediction.dropna()
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index= dataset_for_prediction['Date']


# normalizing exogeneous variables
from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Mean']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X= scaled_input
X.rename(columns={0:'Low', 1:'High', 2:'Open', 3:'Close', 4:'Volume', 5:'Mean'}, inplace=True)
print(X.head())

# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
y=scaler_output
y.rename(columns={0:'BTC Price next day'}, inplace= True)
y.index=dataset_for_prediction.index
print(y.head())


# train-test split
train_size=int(len(df) *0.9)
test_size = int(len(df)) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Init the model
predic_garch =[]
# garch + SARIMAX 就是 garch - SARIMAX
for i in range(test_size):
  # 要把測試集都跑過一遍，比較耗時
  model= SARIMAX(pd.concat([train_y,test_y.iloc[:i+1]]),
  # 從第1行開始取 (把train_y 跟 test_y 用concat混和?)
  exog=pd.concat([train_X,test_X.iloc[:i+1]]),
  order=(0,1,1),
  seasonal_order =(0, 0, 1, 12),
  enforce_invertibility=False, enforce_stationarity=False)
  # 不使用強制平穩與強制平均轉換
  results= model.fit()
  garch = arch.arch_model(results.resid, p=1, q=1,vol='GARCH')
  # resid是殘差
  garch_model = garch.fit(update_freq=1)
  garch_forecast = garch_model.forecast(start = train_size-1,horizon=1,method='simulation')
  predicted_et = garch_forecast.mean['h.1'].iloc[-1]
  predic_garch.append(predicted_et)
  print(predicted_et)


model= SARIMAX(train_y,
 exog=train_X,
 order=(0,1,1),
 seasonal_order =(0, 0, 1, 12),
 enforce_invertibility=False, enforce_stationarity=False)


# training the model
results= model.fit()

# plotting residuals
results.resid.plot()


# making preditions
predictions= results.predict(start =train_size, end=train_size+test_size-2,exog=test_X)
act= pd.DataFrame(scaler_output.iloc[train_size:, 0])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['BTC Price next day']
predictions.rename(columns={'predicted_mean':'Pred'}, inplace=True)
print(predictions)
for i in range(len(predictions)) : 
  predictions.iloc[i,0]= predictions.iloc[i,0]+predic_garch[i]
  # sarimax prediction + garch



# plotting the results
trainPredict = sc_out.inverse_transform(predictions[['Pred']])
testPredict = sc_out.inverse_transform(predictions[['Actual']])

plt.figure(figsize=(20,10))
plt.plot(predictions.index, trainPredict, label='Pred', color='blue')
plt.plot(predictions.index, testPredict, label='Actual', color='red')
plt.legend()
plt.show()

# RMSE
from statsmodels.tools.eval_measures import rmse
error=rmse(trainPredict, testPredict)
print("RMSE:",error)

# print MAPE
from index import mape
print("MAPE:",mape(trainPredict, testPredict))

#print SMAPE
from index import smape
print("SMAPE:",smape(trainPredict, testPredict))