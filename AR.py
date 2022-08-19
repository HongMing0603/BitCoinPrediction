import pandas as pd
import matplotlib.pyplot as plt
import requests
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore")




# Fetching data from the server
url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
# param = {"convert":"USD","slug":"bitcoin","time_end":"1601510400","time_start":"1367107200"}
param = {"convert":"USD","slug":"bitcoin","time_end":"1658275200","time_start":"1367107200"}

content = requests.get(url=url, params=param).json()
df = pd.json_normalize(content['data']['quotes'])

# Extracting and renaming the important variables
df['Date']=pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)
df['Low'] = df['quote.USD.low']
df['High'] = df['quote.USD.high']
df['Open'] = df['quote.USD.open']
df['Close'] = df['quote.USD.close']
df['Volume'] = df['quote.USD.volume']
# 把名字轉換後，刪除原本數據

# Drop original and redundant columns
df=df.drop(columns=['time_open','time_close','time_high','time_low', 'quote.USD.low', 'quote.USD.high', 'quote.USD.open', 'quote.USD.close', 'quote.USD.volume', 'quote.USD.market_cap', 'quote.USD.timestamp'])
# 換了名稱並且捨去原本名稱
# 捨去不要的數據

# Creating a new feature for better representing day-wise values
df['Mean'] = (df['Low'] + df['High'])/2

# Cleaning the data for any NaN or Null fields
# 除去空值
df = df.dropna()



# Creating a copy for making small changes
dataset_for_prediction = df.copy()
dataset_for_prediction['Actual']=dataset_for_prediction['Mean'].shift()
#新增一個欄位Actual 是mean的位移 

dataset_for_prediction=dataset_for_prediction.dropna()
# 去除預測值的空值

# date time typecast
dataset_for_prediction['Date'] =pd.to_datetime(dataset_for_prediction['Date'])
# 把Date時間轉換成時間pandas提供的格式
dataset_for_prediction.index= dataset_for_prediction['Date']
# 把Date設為index


# normalizing the exogeneous variables
from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range=(0, 1))
# print(dataset_for_prediction)

scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Mean']])
# https://blog.csdn.net/weixin_38278334/article/details/82971752
# fit_transform作用
# 經過sc_in 的fit_transform之後 將結果侷限在0~1之間

scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X=scaled_input
X.rename(columns={0:'Low', 1:'High', 2:'Open', 3:'Close', 4:'Volume', 5:'Mean'}, inplace=True)
# 重新命名
print("Normalized X")
print(X.head())


# normalizing the time series
sc_out = MinMaxScaler(feature_range=(0, 1))
scaler_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
# 正規化
# scaler_input裡面沒有Actual
# 為整體的Actual


scaler_output =pd.DataFrame(scaler_output, index=dataset_for_prediction.index)
# 用pandas將他放進資料框
y=scaler_output
y.rename(columns={0:'BTC Price next day'}, inplace= True)
# 將actual變成BTC Price next day
y.index=dataset_for_prediction.index
print("Normalized y")
print(y.head())


# train-test split (cannot shuffle in case of time series)
train_size=int(len(df) *0.9)
# 總資料長度之90%
test_size = int(len(df)) - train_size
# 剩下之10%
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
# x -> High Low Mean...
# y -> Actual
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()
# 訓練集跟測試集處理空值


# Init the best AR model

# import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


model = AutoReg(train_y,lags=2)
# 這裡有點奇怪，跟舊版不太一樣



# training the model
results = model.fit()

# get predictions
predictions = results.predict(start =train_size, end=train_size+test_size-2)
# -2減去交集
# 得到時間index跟一些數字(應該是預測值)

# setting up for plots
act = pd.DataFrame(scaler_output.iloc[train_size:, 0])
# 取最後10%
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
# 只剩下日期(要預測的)
predictions['Actual'] = act['BTC Price next day']
# 這些Actual 是後面百分之10%的
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)
# index不算在columns裡面所以0是pred的值

# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
# sc_out是MinMax解析器跟sc_in一樣，他把數據轉換成人類較為容易看懂的數據
# 0.679531 -> 45618.60144198
# 取得pred原始數據
testActual = sc_out.inverse_transform(predictions[['Actual']])
# 取得Actual 原始數據
# 將標準化的結果轉換成原始數據
# https://blog.csdn.net/qq_34840129/article/details/86257790

# prediction plots
plt.figure(figsize=(20,10))
plt.plot(predictions.index, testActual, label='Actual', color='blue')
plt.plot(predictions.index, testPredict, label='Pred', color='red')
plt.legend()
plt.show()

# print RMSE
from statsmodels.tools.eval_measures import rmse
print("RMSE:",rmse(testActual, testPredict))

# print MAPE
from index import mape
print("MAPE:",mape(testActual, testPredict))

#print SMAPE
from index import smape
print("SMAPE:",smape(testActual, testPredict))