import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda, LeakyReLU, BatchNormalization
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

pd.set_option('display.max_columns', None)

df = pd.read_csv('Panel/data/a_macro.csv', encoding='cp949')

df['date'] = pd.to_datetime(df['date'], format='%Y/%m')
df.set_index('date', inplace=True)

c_code = "A066570"  # 삼성전자 : A005930, SK하이닉스 : A000660, LG전자 : A066570
cdataList = glob("Panel/data/*.csv")[1:]
for cdata in cdataList:
    fname = cdata.split('\\')[1].split('.')[0]
    temp_cdf = pd.read_csv(cdata, encoding='utf-8')
    df[fname] = temp_cdf[c_code].values

for i in ['EPS_ms', 'PER_ms', '거래량_ms', '순매수매도_ms']:
    print(i)
    for j in range(len(df)):
        temp = df.loc[df.index[j], i]
        if (type(temp) is np.float64) | (type(temp) is np.float):
            continue
        else:
            df.loc[df.index[j], i] = float(df.loc[df.index[j], i].replace(",", ""))

df[['거래량_ms', '순매수매도_ms']] = df[['거래량_ms', '순매수매도_ms']].astype('float')

adf_df = pd.DataFrame(columns=['stat', 'p-value', 'stationary'], index=df.columns)
for feature in df.columns:
    print(feature)
    temp_result = adfuller(df[feature].dropna())
    temp_stat = temp_result[0]
    temp_pval = temp_result[1]
    temp_stationary = "o" if temp_pval <= 0.05 else "x"
    adf_df.loc[feature] = [temp_stat, temp_pval, temp_stationary]
#
non_staionary = adf_df[adf_df['stationary'] == 'x'].index
# xp, cpi, pis, kospi, dowjones, nasdaq, e_dollar, oil, PBR 에 대해 로그변환
for col in ['xp', 'cpi', 'esi', 'pis', 'kospi', 'dowjones', 'nasdaq', 'shcomp', 'e_dollar', 'oil', 'PER_ms', '거래량_ms', '매출액']:
    df[col] = np.log(df[col]+0.001)

adf_df2 = pd.DataFrame(columns=['stat', 'p-value', 'stationary'], index=non_staionary)
for feature in non_staionary:
    print(feature)
    temp_result = adfuller(df[feature].dropna().diff().dropna())
    temp_stat = temp_result[0]
    temp_pval = temp_result[1]
    temp_stationary = "o" if temp_pval <= 0.05 else "x"
    adf_df2.loc[feature] = [temp_stat, temp_pval, temp_stationary]

df[non_staionary] = df[non_staionary].diff()

n = 3
target_df = df[[
    'xp', 'cpi', 'ppi', 'esi', 'pis', 'dowjones', 'nasdaq', 'shcomp', 'e_dollar', 'oil', 'kospi',
    'PBR', 'PER', 'ROA(영업이익)', 'ROE(영업이익)', '당기순이익(손실)', '매출액', '매출총이익(손실)', '수익률', '영업이익률', 'EPS_ms', '거래량_ms', '순매수매도_ms']].iloc[55:-1, :]
# target_df['kospi'] = target_df['kospi'].fillna(method='bfill')
# target_df['pis'] = target_df['pis'].fillna(method='bfill')

# target_df에 수익률 전기 컬럼 추가 및 3개월 후 데이터 삽입
for i in range(3, len(target_df)):
    target_df.loc[target_df.index[i], '전기수익률'] = target_df.loc[target_df.index[i-1], '수익률']
    target_df.loc[target_df.index[i], '전전기수익률'] = target_df.loc[target_df.index[i-2], '수익률']
    target_df.loc[target_df.index[i], '전전전기수익률'] = target_df.loc[target_df.index[i-3], '수익률']
    # for cfeature in ['koribor', 'xp', 'cpi', 'ppi', 'esi', 'pis', 'dowjones', 'nasdaq', 'shcomp', 'e_dollar', 'oil','EPS', 'PBR', 'PER', 'ROA(영업이익)', 'ROE(영업이익)', '당기순이익(손실)', '매출액', '매출총이익(손실)', '영업이익률', 'EPS_ms', '거래량_ms', 'PER_ms', '순매수매도_ms']:
    #     print(cfeature)
    #     target_df.loc[target_df.index[i], '{}개월후{}'.format(n, cfeature)] = target_df.loc[target_df.index[i+n], cfeature]
    #     target_df.loc[target_df.index[i], '{}개월후{}'.format(n, cfeature)] = target_df.loc[target_df.index[i+n], cfeature]

# target_df = target_df.reset_index().iloc[:, 1:]
target_df = target_df.dropna()
target_df2 = target_df.copy()

target_df3 = target_df[[
    'kospi', '매출액',
    'xp', 'cpi', 'ppi', 'pis', 'nasdaq', 'e_dollar', 'oil',
    'esi',
    '수익률',
    'dowjones',
    'shcomp',
    'EPS_ms', '거래량_ms', '순매수매도_ms', 'PBR',
    '전기수익률',
    # '전전기수익률',
    # '전전전기수익률',
    # 'PBR', 'ROA(영업이익)', 'ROE(영업이익)', '당기순이익(손실)', '매출액', '매출총이익(손실)', '영업이익률',
    # '{}개월후EPS_ms'.format(n),
    # '{}개월후PBR'.format(n),
    # '{}개월후PER_ms'.format(n),
    # '{}개월후ROA(영업이익)'.format(n),
    # '{}개월후ROE(영업이익)'.format(n),
    # '{}개월후당기순이익(손실)'.format(n),
    # '{}개월후매출액'.format(n),
    # '{}개월후매출총이익(손실)'.format(n),
    # '{}개월후영업이익률'.format(n),
    # '{}개월후거래량_ms'.format(n),
    # '{}개월후순매수매도_ms'.format(n)
]]
X = target_df3.drop(['수익률'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X[:-1, :]
y = np.array(target_df3[['수익률']]).reshape(-1, 1)
# scaler = MinMaxScaler()
# y = scaler.fit_transform(y)
y = y[1:].reshape(1, -1)[0]

###
# timesteps = 3
#
# X_train_sequence = []
# for i in range(len(X_train) - timesteps + 1):
#     X_train_sequence.append(X_train[i:i + timesteps])
# X_train_sequence = np.array(X_train_sequence)
#
# X_test_sequence = []
# for i in range(len(X_test) - timesteps + 1):
#     X_test_sequence.append(X_test[i:i + timesteps])
# X_test_sequence = np.array(X_test_sequence)
#
# ###
# model = Sequential()
# model.add(LSTM(64, input_shape=(X_train_sequence.shape[1], X_train_sequence.shape[2])))
# model.add(Dense(1))
#
# ###
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X_train_sequence, y_train, epochs=100, batch_size=32, validation_split=0.2)
#
# ###
# y_pred_scaled = model.predict(X_test_sequence)
# y_pred = scaler.inverse_transform(y_pred_scaled)
#
# ###
# plt.figure(figsize=(12, 9))
# plt.plot(np.asarray(y_test)[3:], label='actual')
# plt.plot(y_pred, label='prediction')
# plt.legend()
# plt.show()

###
def train_make_dataset(data, label, train_window_size=3):
    feature_list = []
    label_list = []
    for j in range(len(data) - train_window_size):
        feature_list.append(np.array(data[j:j+train_window_size]))
        label_list.append(np.array(label[j+train_window_size]))
    return np.array(feature_list), np.array(label_list)

def test_make_dataset(data, label):
    feature_list = []
    label_list = []
    feature_list.append(np.array(data))
    label_list.append(np.array(label))
    return np.array(feature_list), np.array(label_list)


###
window_size = 1
predictions = []
true_values = []

def swish(x):
    return x * tf.keras.activations.sigmoid(x)

for i in range(len(X) - window_size):
    X_train = X[i:i+window_size, :]
    y_train = y[i:i+window_size]
    X_test = X[i+1:i+1+window_size, :]
    y_test = y[i+1:i+1+window_size]
    train_feature, train_label = test_make_dataset(X_train, y_train)
    test_feature, test_label = test_make_dataset(X_test, y_test)

    model = Sequential([
        Conv1D(filters=32, kernel_size=5,
               padding="causal",
               activation=swish,
               input_shape=[window_size, 17]),
        LSTM(32, activation=swish, return_sequences=True),
        LSTM(64, activation=swish, return_sequences=True),
        LSTM(32, activation=swish),
        Dense(16, activation=swish),
        Dense(1),
    ])

    loss = Huber()
    optimizer = Adam(0.001)
    model.compile(loss=loss, optimizer=optimizer, metrics=['mse'], run_eagerly=True)
    # model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    model_path = 'model'
    filename = os.path.join(model_path, 'tmp_checkpoint.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(train_feature, train_label,
                        epochs=200,
                        batch_size=8,
                        validation_data=(test_feature, test_label),
                        callbacks=[early_stop, checkpoint],
                        verbose=False)

    model.load_weights(filename)
    pred = model.predict(test_feature)
    print(pred)
    predictions.append(pred[0][0])

    true_values.append(y[i + window_size])

rmse = np.sqrt(mean_squared_error(true_values, predictions))
print("Root Mean Squared Error (RMSE): {}".format(rmse))

plt.figure(figsize=(12, 9))
plt.plot(true_values, label='actual')
plt.plot(predictions, label='prediction')
plt.xlabel('Date')
plt.ylabel('수익률')
plt.title("Stock Return Prediction using LSTM")
plt.legend()
plt.show()
###
print("Root Mean Squared Error (RMSE): {}".format(rmse))
# 5월 주식수익률 예측
print(target_df3.loc['2023-05-01', '수익률'])
print(predictions[-1])
# 6월 주식수익률 예측
X_6 = scaler.transform(target_df3.drop(['수익률'], axis=1))[-1]
X_6_feature_list = []
X_6_feature_list.append(np.array(X_6))
X_7_predict = model.predict(np.array([X_6_feature_list]))
print(df.loc['2023-06-01', '수익률'])
print(X_7_predict)

###
print(target_df3.loc['2023-05-01', '수익률'])
print(df.loc['2023-06-01', '수익률'])

###
true_pred_df = pd.DataFrame(columns=['true', 'pred'])
true_pred_df['true'] = true_values
true_pred_df['pred'] = predictions

from statsmodels.formula.api import ols
ols_model = ols('true ~ pred', data=true_pred_df).fit()
print(ols_model.summary())

###
non_staionary