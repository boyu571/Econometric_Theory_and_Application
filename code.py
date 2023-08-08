import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
###
plt.rc('font', family='NanumGothic')
pd.set_option('display.max_columns', None)

df = pd.read_csv('Panel/data/a_macro.csv', encoding='cp949')

df['date'] = pd.to_datetime(df['date'], format='%Y/%m')
df.set_index('date', inplace=True)

c_code = "A066570"  # 삼성전자
cdataList = glob("Panel/data/*.csv")[1:]
for cdata in cdataList:
    fname = cdata.split('\\')[1].split('.')[0]
    temp_cdf = pd.read_csv(cdata, encoding='utf-8')
    df[fname] = temp_cdf[c_code].values

###
df['PBR'].mean()  # 삼전 : 1.86 / 하이닉스 : 1.4 / LG전자 : 1.6

###
for i in ['EPS_ms', 'PER_ms', '거래량_ms', '순매수매도_ms']:
    print(i)
    for j in range(len(df)):
        temp = df.loc[df.index[j], i]
        if (type(temp) is np.float64) | (type(temp) is np.float):
            continue
        else:
            df.loc[df.index[j], i] = float(df.loc[df.index[j], i].replace(",", ""))

df[['거래량_ms', '순매수매도_ms']] = df[['거래량_ms', '순매수매도_ms']].astype('float')

### visualization for all columns
for feature in df.columns:
    plt.figure(figsize=(10, 8))
    plt.plot(df.index, df[feature])
    plt.xlabel('date')
    plt.ylabel(feature)
    plt.title('the change in {} over time'.format(feature))
    plt.show()
###
plt.figure(figsize=(10, 8))
plt.plot(df.index, df['수익률'])
plt.xlabel('date')
plt.ylabel('수익률')
plt.title('the change in {} over time'.format('수익률'))
plt.show()

###

# ADF test
adf_df = pd.DataFrame(columns=['stat', 'p-value', 'stationary'], index=df.columns)
for feature in df.columns:
    print(feature)
    temp_result = adfuller(df[feature].dropna())
    temp_stat = temp_result[0]
    temp_pval = temp_result[1]
    temp_stationary = "o" if temp_pval <= 0.05 else "x"
    adf_df.loc[feature] = [temp_stat, temp_pval, temp_stationary]
###
adf_df
###
# 1st differenciate and ADF test
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

###
adf_df2


### change non_stationary columns to stationary ones
# df[non_staionary] = np.log(df[non_staionary])
df[non_staionary] = df[non_staionary].diff()

### RE-visualization for all columns
for feature in df.columns:
    plt.figure(figsize=(10, 8))
    plt.plot(df.index, df[feature])
    plt.xlabel('date')
    plt.ylabel(feature)
    plt.title('the change in {} over time'.format(feature))
    plt.show()


###
n = 3
target_df = df[[
    'xp', 'cpi', 'ppi', 'esi', 'pis', 'dowjones', 'nasdaq', 'shcomp', 'e_dollar', 'oil',
    'EPS', 'PBR', 'PER', 'ROA(영업이익)', 'ROE(영업이익)', '당기순이익(손실)', '매출액', '매출총이익(손실)', '수익률', '영업이익률', 'EPS_ms', '거래량_ms', '순매수매도_ms']].iloc[55:-1, :]
# target_df['kospi'] = target_df['kospi'].fillna(method='bfill')
target_df['pis'] = target_df['pis'].fillna(method='bfill')

# target_df에 수익률 전기 컬럼 추가 및 3개월 후 데이터 삽입
for i in range(1, len(target_df)):
    target_df.loc[target_df.index[i], '전기수익률'] = target_df.loc[target_df.index[i-1], '수익률']
    # for cfeature in ['koribor', 'xp', 'cpi', 'ppi', 'esi', 'pis', 'dowjones', 'nasdaq', 'shcomp', 'e_dollar', 'oil','EPS', 'PBR', 'PER', 'ROA(영업이익)', 'ROE(영업이익)', '당기순이익(손실)', '매출액', '매출총이익(손실)', '영업이익률', 'EPS_ms', '거래량_ms', 'PER_ms', '순매수매도_ms']:
    #     print(cfeature)
    #     target_df.loc[target_df.index[i], '{}개월후{}'.format(n, cfeature)] = target_df.loc[target_df.index[i+n], cfeature]
    #     target_df.loc[target_df.index[i], '{}개월후{}'.format(n, cfeature)] = target_df.loc[target_df.index[i+n], cfeature]

# target_df = target_df.reset_index().iloc[:, 1:]
target_df = target_df.dropna()
target_df2 = target_df.copy()


###
for cfeature in ['koribor', 'xp', 'cpi', 'ppi', 'esi', 'pis', 'dowjones', 'nasdaq', 'shcomp', 'e_dollar', 'oil','EPS', 'PBR', 'PER', 'ROA(영업이익)', 'ROE(영업이익)', '당기순이익(손실)', '매출액', '매출총이익(손실)', '영업이익률', 'EPS_ms', '거래량_ms', 'PER_ms', '순매수매도_ms']:
    X = target_df.drop(['{}개월후{}'.format(n, cfeature)], axis=1)
    y = target_df['{}개월후{}'.format(n, cfeature)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = CatBoostRegressor(iterations=100, learning_rate=0.3, depth=9)
    model.fit(X_train, y_train)
    prediction = model.predict(X)

    target_df2['{}개월후{}'.format(n, cfeature)] = prediction
    print(cfeature)


###
target_df3 = target_df[[
    'xp', 'cpi', 'ppi', 'pis', 'nasdaq', 'e_dollar', 'oil',
    'esi',
    '전기수익률',
    'dowjones',
    'shcomp',
    '수익률',
    'EPS_ms', '거래량_ms', '순매수매도_ms',
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
df_X = X.iloc[:-1, :].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X[:-1, :]
y = target_df3['수익률']
df_y = y[1:].copy()
y = y[1:]

###
prediction

###
window_size = 3
predictions_svm = []
predictions_cat = []
predictions_lgbm = []
true_values = []
cat_fi = pd.DataFrame(index=df_X.columns)
lgbm_fi = pd.DataFrame(index=df_X.columns)
for i in range(len(X) - window_size):
    X_train = X[i:i+window_size, :]
    y_train = y.iloc[i:i+window_size].values
    X_test = X[i + window_size:i + window_size + 1, :]

    model = SVR()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    predictions_svm.append(prediction[0])

    model = CatBoostRegressor(iterations=100, learning_rate=0.3, depth=8)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    predictions_cat.append(prediction[0])
    fi = pd.Series(model.feature_importances_, index=df_X.columns)
    for f in df_X.columns:
        cat_fi.loc[f, df_y.index[i + window_size]] = fi[f]

    model = LGBMRegressor(max_depth=9, num_leaves=20)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    predictions_lgbm.append(prediction[0])

    true_values.append(y.iloc[i + window_size])

rmse_svm = np.sqrt(mean_squared_error(true_values, predictions_svm))
print("Root Mean Squared Error (RMSE): {}".format(rmse_svm))
rmse_lgbm = np.sqrt(mean_squared_error(true_values, predictions_lgbm))
print("Root Mean Squared Error (RMSE): {}".format(rmse_lgbm))
rmse_cat = np.sqrt(mean_squared_error(true_values, predictions_cat))
print("Root Mean Squared Error (RMSE): {}".format(rmse_cat))

sq_rmse_svm = np.square(rmse_svm)
sq_rmse_lgbm = np.square(rmse_lgbm)
sq_cat_lgbm = np.square(rmse_cat)

###

for f in cat_fi.index:
    plt.figure()
    plt.plot(cat_fi.columns, cat_fi.loc[f, :])
    plt.title("Feature Importance / {}".format(f))
    plt.show()

###
predictions = []
for i in range(len(predictions_lgbm)):
    temp = (sq_rmse_svm/(sq_rmse_svm + sq_rmse_lgbm+sq_cat_lgbm)) * predictions_lgbm[i] +\
           (sq_rmse_lgbm/(sq_rmse_svm + sq_rmse_lgbm+sq_cat_lgbm)) * predictions_lgbm[i] +\
           (sq_cat_lgbm/(sq_rmse_svm + sq_rmse_lgbm+sq_cat_lgbm)) * predictions_cat[i]
    predictions.append(temp)
rmse = np.sqrt(mean_squared_error(true_values, predictions))
print("Root Mean Squared Error (RMSE): {}".format(rmse))


###
plt.figure(figsize=(10, 6))
plt.plot(target_df3.index, target_df3['수익률'], label='True Values')
plt.plot(target_df3.index[window_size+1:], predictions, label='Predicted Values')
plt.xlabel("Date")
plt.ylabel("수익률")
plt.title("Rolling Window(size={}) Prediction using ML".format(window_size))
plt.legend()
plt.show()

###
pred_7 = model.predict(X[-3:,])
print(pred_7)

###
predictions[-3:]

