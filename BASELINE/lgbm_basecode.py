#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


# !pip install pandas
# !pip install numpy
# # install sklearn
# !pip install scikit-learn
# !pip install tqdm
# !pip install seaborn
# !pip install prophet
# !pip install lightgbm
# !pip install optuna


# In[1]:


import random
import pandas as pd
import numpy as np
import os
import re
import glob


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta, timezone
from itertools import combinations, product

import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
import gc
import warnings
warnings.filterwarnings(action='ignore') 


# # Device & Path

# In[2]:


# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
# torch.backends.mps.is_available()


# In[3]:


# PATH
DATA_PATH  = '../DATA'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TRAIN_CSV  = os.path.join(DATA_PATH, 'train.csv')
SAMPLE_PATH = os.path.join(DATA_PATH, 'sample_submission.csv')


# ## Hyperparameter Setting

# In[4]:


CFG = {
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'SEED':41
}


# ## Fixed RandomSeed

# In[5]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


seed_everything(CFG['SEED']) # Seed 고정


# ## Data Pre-processing

# In[6]:


df_train = pd.read_csv(TRAIN_CSV)
df_ss = pd.read_csv(SAMPLE_PATH)
df_train


# ## columns 
# 1. '일자': 이 컬럼은 특정 거래가 발생한 날짜를 나타냅니다. "YYYY-MM-DD" 형식으로 표현됩니다.
# 
# 2. '종목코드': 각각의 주식을 식별하는 고유한 코드입니다. 한국의 경우 종목코드는 대부분 6자리 숫자로 이루어져 있습니다. (ex - A060310)
# 
# 3. '종목명': 주식의 공식 이름을 나타냅니다. 이 이름은 주로 회사의 이름을 반영하며, 시장에서 해당 주식을 찾을 때 사용됩니다.
# 
# 4. '거래량': 특정 일자에 해당 주식이 거래된 총 주식 수를 나타냅니다. 거래량은 시장의 활동 수준과 관심도를 반영하는 중요한 지표입니다.
# 
# 5. '시가': 주식 시장이 개장했을 때의 첫 거래 가격을 의미합니다. 이는 해당 날의 시장 흐름을 이해하는데 도움이 됩니다.
# 
# 6. '고가': 특정 일자에 해당 주식이 거래된 가장 높은 가격을 나타냅니다.
# 
# 7. '저가': 특정 일자에 해당 주식이 거래된 가장 낮은 가격을 나타냅니다.
# 
# 8. '종가': 주식 시장이 마감했을 때의 마지막 거래 가격을 의미합니다. 종가는 해당 일의 주식 가격 변동을 반영하며, 이후의 시장 분석에 중요한 기준이 됩니다.

# In[7]:


# change columns name
df = df_train.rename(columns={
    '일자': 'date',
    '종목코드': 'code',
    '종목명': 'name',
    '거래량': 'volume',
    '시가': 'open',
    '고가': 'high',
    '저가': 'low',
    '종가': 'close'
})
# datetime
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df_price = df.pivot(index='date', columns='code', values='close')

df_processed = df.copy()
le = preprocessing.LabelEncoder()
df_processed['code'] = le.fit_transform(df_processed['code'])  
df_processed.drop(columns=['name'], inplace=True)

df_processed.set_index('date', inplace=True)


# In[8]:


# target 값 생성
df_processed['target'] = df_processed.groupby('code')['close'].pct_change()
df_processed.fillna(0, inplace=True)
print(df_processed.isnull().sum())


# In[10]:


# time series train test split
def ts_train_test_split(df, test_size=0.05):
    df_prices = df.iloc[:-int(len(df)*test_size)]
    prices = df.iloc[-int(len(df)*test_size):]
    return df_prices, prices
df_prices, prices = ts_train_test_split(df_processed)


# ## Sharpe ratio 측정 함수

# In[12]:


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['rank'].min() == 0
        assert df['rank'].max() == len(df['rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='rank')['target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='rank', ascending=False)['target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


# Utilities 

def calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
    weights_mean = weights.mean()
    df = df.sort_values(by='rank')
    purchase = (df['target'][:portfolio_size]  * weights).sum() / weights_mean
    short    = (df['target'][-portfolio_size:] * weights[::-1]).sum() / weights_mean
    return purchase - short
def calc_spread_return_sharpe(df, portfolio_size=200, toprank_weight_ratio=2):
    grp = df.groupby('date')
    min_size = grp["target"].count().min()
    if min_size<2*portfolio_size:
        portfolio_size=min_size//2
        if portfolio_size<1:
            return 0, None
    buf = grp.apply(calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio, buf

def add_rank(df, col_name="pred"):
    df["rank"] = df.groupby("date")[col_name].rank(ascending=False, method="first") - 1 
    df["rank"] = df["rank"].astype("int")
    return df


# In[14]:


def adjuster(df):
    def calc_pred(df, x, y, z):
        return df['target'].where(df['target'].abs() < x, df['target'] * y + np.sign(df['target']) * z)

    def objective(trial, df):
        x = trial.suggest_uniform('x', 0, 0.2)
        y = trial.suggest_uniform('y', 0, 0.05)
        z = trial.suggest_uniform('z', 0, 1e-3)
        df["rank"] = calc_pred(df, x, y, z).rank(ascending=False, method="first") - 1 
        return calc_spread_return_per_day(df, 200, 2)

    def predictor_per_day(df):
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SD))#5187
        study.optimize(lambda trial: abs(objective(trial, df) - 3), 3)
        return calc_pred(df, *study.best_params.values())

    return df.groupby("date").apply(predictor_per_day).reset_index(level=0, drop=True)

def _predictor_base(feature_df):
    return model.predict(feature_df[feats])

def _predictor_with_adjuster(feature_df):
    df_pred = feature_df.copy()
    df_pred["target"] = model.predict(feature_df[feats])
    return adjuster(df_pred).values.T


# In[15]:


# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm

# 1. 특성 엔지니어링
# 'close' 값만을 사용하는 것이 아니라 'open', 'high', 'low' 값을 이용해 추가 특성을 만듭니다.
df_prices['volatility'] = df_prices['high'] - df_prices['low']  # 일일 변동성
df_prices['daily_change'] = df_prices['close'] - df_prices['open']  # 일일 가격 변동

# 2. 파라미터 최적화를 위한 그리드 서치 설정
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

# 3. 랜덤 포레스트 모델을 설정하고 그리드 서치를 실행
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(df_prices[['close', 'volatility', 'daily_change']], df_prices['target'])

# 최적의 파라미터로 모델을 재학습
best_model = grid_search.best_estimator_

# 4. 예측 및 성능 평가
df_prices["pred"] = best_model.predict(df_prices[['close', 'volatility', 'daily_change']])
score, buf = calc_spread_return_sharpe(add_rank(df_prices))

print(f'Best parameters: {grid_search.best_params_}')
print(f'Sharpe Ratio Score -> {score}')


# prices
# 'close' 값만을 사용하는 것이 아니라 'open', 'high', 'low' 값을 이용해 추가 특성을 만듭니다.
prices['volatility'] = prices['high'] - prices['low']  # 일일 변동성
prices['daily_change'] = prices['close'] - prices['open']  # 일일 가격 변동
prices["pred"] = best_model.predict(prices[['close', 'volatility', 'daily_change']])
score, buf = calc_spread_return_sharpe(add_rank(prices))


# In[17]:


print(f'Best parameters: {grid_search.best_params_}')
print(f'Sharpe Ratio Score -> {score}')


# In[18]:


# 시간 고유값 
kst = timezone(timedelta(hours=9))        
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join(DATA_PATH, 'results', train_serial)

# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)    
RESULT_PATH = os.path.join(RECORDER_DIR, 'submission.csv')


# In[25]:


# # 코드, 평균 최종 수익률, 순위(rank)를 포함하는 데이터프레임 생성
# df_final = prices.groupby('code')['pred'].mean().reset_index()
# df_final = df_final.sort_values(by='pred', ascending=False)
# df_final.reset_index(drop=True, inplace=True)
# df_final['rank'] = df_final.index+1
# df_final['code']= le.inverse_transform(df_final['code'].astype(int))
# df_final = df_final[['code', 'rank']]
# df_final.rename(columns={'code': '종목코드', 'rank': '순위'}, inplace=True)
# df_final.to_csv(RESULT_PATH, index=False)
# df_final


# In[20]:


df_final = prices.copy()

# rolling average return and risk(standard deviation)
df_final['rolling_return'] = df_final.groupby('code')['pred'].rolling(2).mean().reset_index(0, drop=True)
df_final['rolling_risk'] = df_final.groupby('code')['pred'].rolling(2).std().reset_index(0, drop=True)

# risk adjusted return
df_final['risk_adjusted_return'] = df_final['rolling_return'] / df_final['rolling_risk']

# 15-day rolling average of risk adjusted return
df_final['average_risk_adjusted_return'] = df_final.groupby('code')['risk_adjusted_return'].rolling(15).mean().reset_index(0, drop=True)

# dropna
df_final = df_final.dropna()

# 코드, 평균 최종 수익률, 순위(rank)를 포함하는 데이터프레임 생성
df_final = df_final.groupby('code')['average_risk_adjusted_return'].last().reset_index()
df_final = df_final.sort_values(by='average_risk_adjusted_return', ascending=False)
df_final.reset_index(drop=True, inplace=True)
df_final['rank'] = df_final.index + 1
df_final['code'] = le.inverse_transform(df_final['code'].astype(int))
df_final = df_final[['code', 'rank']]
df_final.rename(columns={'code': '종목코드', 'rank': '순위'}, inplace=True)
df_final.to_csv(RESULT_PATH, index=False)
df_final


# In[23]:





# In[65]:


np.random.seed(0)
feats = ["close"]
max_score = 0
max_depth = 0
for md in tqdm(range(3,40)):
    model = DecisionTreeRegressor( max_depth=md ) # Controlling the overfit with max_depth parameter
    model.fit(df_prices[feats],df_prices["target"])
    predictor = _predictor_base
    prices["pred"] = predictor(prices)
    score, buf = calc_spread_return_sharpe(add_rank(prices))
    if score>max_score:
        max_score = score
        max_depth = md
        
model = DecisionTreeRegressor( max_depth=max_depth )
model.fit(df_prices[feats],df_prices["target"])
print(f'Max_deph={max_depth} : Sharpe Ratio Score base -> {max_score}')


# In[66]:


np.random.seed(0)
feats = ["close"]
max_score = 0
max_depth = 0
for md in tqdm(range(3,40)):
    model = DecisionTreeRegressor( max_depth=md ) # Controlling the overfit with max_depth parameter
    model.fit(df_prices[feats],df_prices["target"])
    predictor = _predictor_base
    prices["pred"] = predictor(prices)
    score, buf = calc_spread_return_sharpe(add_rank(prices))
    if score>max_score:
        max_score = score
        max_depth = md
        
model = DecisionTreeRegressor( max_depth=max_depth )
model.fit(df_prices[feats],df_prices["target"])
print(f'Max_deph={max_depth} : Sharpe Ratio Score base -> {max_score}')
# Controlling the Sharpe Ratio Score (≃3)
predictor = _predictor_with_adjuster
err = 1
maxSD = 2000
for SD in tqdm(range(maxSD,4000)):
    prices["pred"] = predictor(prices)
    score, buf = calc_spread_return_sharpe(add_rank(prices))
    if abs(score-3)<=err and score<3:
        err=abs(score-3)
        maxSD = SD
        print(f'{maxSD} Sharpe Ratio Score with adjuster -> {score}')
        
SD = maxSD


# In[67]:


prices


# In[68]:


# 시간 고유값 
kst = timezone(timedelta(hours=9))        
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join(DATA_PATH, 'results', train_serial)

# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)    
RESULT_PATH = os.path.join(RECORDER_DIR, 'submission.csv')


# In[69]:


import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from datetime import datetime

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['rank'].min() == 0
        assert df['rank'].max() == len(df['rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='rank')['target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='rank', ascending=False)['target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


# calc_spread_return_sharpe 함수를 사용하여 6월 1일부터 6월 15일까지 Sharpe Ratio 계산
sharpe_ratio = calc_spread_return_sharpe(prices, portfolio_size=20, toprank_weight_ratio=2)

print(f'The Sharpe Ratio from June 1 to June 15 is {sharpe_ratio:.2f}')

# calc_spread_return_sharpe 함수를 사용하여 6월 1일부터 6월 15일까지 Sharpe Ratio 계산
sharpe_ratio = calc_spread_return_sharpe(prices, portfolio_size=200, toprank_weight_ratio=2)

print(f'The Sharpe Ratio from June 1 to June 15 is {sharpe_ratio:.2f}')


# In[ ]:





# - test size 0.2
# - The Sharpe Ratio from June 1 to June 15 is 0.18 (20)
# - The Sharpe Ratio from June 1 to June 15 is 0.39 (200)

# In[51]:


df_prices


# In[70]:


prices


# In[71]:


# 코드, 평균 최종 수익률, 순위(rank)를 포함하는 데이터프레임 생성
df_final = prices.groupby('code')['pred'].mean().reset_index()
df_final = df_final.sort_values(by='pred', ascending=False)
df_final.reset_index(drop=True, inplace=True)
df_final['rank'] = df_final.index+1
df_final['code']= le.inverse_transform(df_final['code'].astype(int))
df_final = df_final[['code', 'rank']]
df_final.rename(columns={'code': '종목코드', 'rank': '순위'}, inplace=True)
df_final.to_csv(RESULT_PATH, index=False)
df_final


# In[45]:


df_final


# In[42]:


pd.read_csv('/Users/admin/Documents/GitHub/Dacon_stock_price_prediction/DATA/results/20230706_214757/submission.csv')


# In[28]:


# Controlling the Sharpe Ratio Score (≃3)
predictor = _predictor_with_adjuster
err = 1
maxSD = 3683
for SD in tqdm(range(maxSD,4000)):
    prices["pred"] = predictor(prices)
    score, buf = calc_spread_return_sharpe(add_rank(prices))
    if abs(score-3)<=err and score<3:
        err=abs(score-3)
        maxSD = SD
        print(f'{maxSD} Sharpe Ratio Score with adjuster -> {score}')
        
SD = maxSD
%%time
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for prices, options, financials, trades, secondary_prices, sample_prediction in iter_test:
    prices = fill_nans(prices)
    prices.loc[:,"pred"] = predictor(prices)
    prices = add_rank(prices)
    rank = prices.set_index('code')['rank'].to_dict()
    sample_prediction['rank'] = sample_prediction['code'].map(rank)
    env.predict(sample_prediction)


# In[160]:


tscv = TimeSeriesSplit(n_splits=5)

def timeseries_cv(df):
    results_df = pd.DataFrame(columns=['code', 'final_return'])

    for code in df['code'].unique():
        df_company = df[df['code'] == code]

        final_returns = []  # Store all final returns for this company

        for train_index, val_index in tscv.split(df_company):
            X_train, X_val = df_company[['volume', 'open', 'high', 'low', 'close_rolling_mean']].iloc[train_index], df_company[['volume', 'open', 'high', 'low', 'close_rolling_mean']].iloc[val_index]
            y_train, y_val = df_company['close'].iloc[train_index], df_company['close'].iloc[val_index]

            model = LGBMRegressor()
            model.fit(X_train, y_train)

            last_row = X_val.iloc[-1][['volume', 'open', 'high', 'low', 'close_rolling_mean']].copy()

            future_returns = []
            for i in range(15):
                future_date = last_row.name + timedelta(days=i+1)
                prediction = model.predict(last_row.values.reshape(1,-1))
                future_returns.append(prediction[0])

                # Update the last row with the predicted value
                last_row['close_rolling_mean'] = prediction[0]

            if future_returns[0] != 0:
                final_return = (future_returns[-1] - future_returns[0]) / future_returns[0]
            else:
                final_return = 0

            final_returns.append(final_return)

        # Compute the average final return for this company
        avg_final_return = sum(final_returns) / len(final_returns)

        results_df = results_df.append({'code': code, 'final_return': avg_final_return}, ignore_index=True)

    return results_df


# In[161]:


# df_processed.drop(columns=['index'], inplace=True)
df_processed.info()


# ## !!RUN!!

# In[116]:


results_df = timeseries_cv(df_processed)
results_df


# # Record path

# In[240]:


# 시간 고유값 
kst = timezone(timedelta(hours=9))        
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# 기록 경로
RECORDER_DIR = os.path.join(DATA_PATH, 'results', train_serial)

# 현재 시간 기준 폴더 생성
os.makedirs(RECORDER_DIR, exist_ok=True)    
RESULT_PATH = os.path.join(RECORDER_DIR, 'submission.csv')


# ## Submission

# In[125]:


results_df['rank'] = results_df['final_return'].rank(method='first', ascending=False).astype('int')
results_df['code']= le.inverse_transform(results_df['code'].astype(int))
results_df.sort_values(by=['rank'], inplace=True)
results_df = results_df[['code', 'rank']]
results_df.rename(columns={'code': '종목코드', 'rank': '순위'}, inplace=True)
results_df.to_csv(RESULT_PATH, index=False)
results_df


# 

# ## Diff calculation

# In[217]:





# In[238]:





# In[239]:





# In[213]:


# 각 코드별 최종 수익률 계산
final_returns = {}
for code, df in df_predictions.items():
    final_returns[code] = (df['trend'].iloc[-1] - df['trend'].iloc[0]) / df['trend'].iloc[0]

# 평균 최종 수익률 계산
average_final_returns = {}
for code, final_return in final_returns.items():
    average_final_returns[code] = final_return.mean()

# 평균 최종 수익률에 따라 코드 순위 매기기
ranked_codes = sorted(average_final_returns, key=average_final_returns.get, reverse=True)
ranks = {code: rank for rank, code in enumerate(ranked_codes, 1)}

# 각 분리된 데이터프레임에 순위 열 추가
for code, df in df_predictions.items():
    df['rank'] = ranks[code]

# 모든 분리된 데이터프레임 연결
df_predictions = pd.concat(df_predictions.values())
# 코드, 평균 최종 수익률, 순위(rank)를 포함하는 데이터프레임 생성
df_final = pd.DataFrame({
    'code': ranked_codes,
    'final_return': [average_final_returns[code] for code in ranked_codes],
    'rank': [ranks[code] for code in ranked_codes]
})


df_final['code']= le.inverse_transform(df_final['code'].astype(int))
df_final = df_final[['code', 'rank']]
df_final.rename(columns={'code': '종목코드', 'rank': '순위'}, inplace=True)
df_final.to_csv(RESULT_PATH, index=False)
df_final


# In[215]:


df_final = df_final[['code', 'rank']]
df_final.rename(columns={'code': '종목코드', 'rank': '순위'}, inplace=True)
df_final.to_csv(RESULT_PATH, index=False)
df_final


# In[127]:


def prep_prices(df):
    
    from decimal import ROUND_HALF_UP, Decimal
    
    pcols = ['open', 'high', 'low', 'close']

    #df.ExpectedDividend.fillna(0,inplace=True) # No column 'ExpectedDividend' in the given data
    
    def qround(x):
        return float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
    
    def adjust_prices(df_code):
        df_code = df_code.sort_values("date", ascending=False)
        #df_code.loc[:, "CumAdjust"] = df_code["AdjustmentFactor"].cumprod() # No column 'AdjustmentFactor' in the given data

        # generate adjusted prices
        for p in pcols:     
            df_code.loc[:, p] = df_code[p].apply(qround)
        df_code.loc[:, "volume"] = df_code["volume"]
        df_code.ffill(inplace=True)
        df_code.bfill(inplace=True)
        
        # generate and fill Targets
        #df_code.loc[:, "Target"] = df_code.Close.pct_change().shift(-2).fillna(df_code.Target).fillna(0) # No column 'Target' in the given data
        #df_code.Target.fillna(0,inplace=True)

        return df_code

    # generate Adjusted
    df = df.sort_values(["code", "date"])
    df = df.groupby("code").apply(adjust_prices).reset_index(drop=True)
    df = df.sort_values("date")
    return df


# In[ ]:





# In[128]:


df_processed


# - test를 위한 코드

# In[163]:


df_processed.reset_index(inplace=True)
# Process the data using the given function
df_train = prep_prices(df_processed)

# Select data from 2021-06-03 to 2023-05-30
mask = (df_train['date'] >= "2021-06-03") & (df_train['date'] <= "2023-05-30")
df_train = df_train.loc[mask]

# Prepare the data for Prophet
df_train = df_train.rename(columns={"date": "ds", "close": "y"})

# Separate features by 'code'
data_dict = {}
unique_codes = df_train['code'].unique()

for code in unique_codes:
    data_dict[code] = df_train[df_train['code'] == code][['ds', 'y']]


# In[168]:


data_dict[661]


# In[167]:


model = Prophet()
model.fit(data_dict[661])

# # Create a dataframe to hold the future dates
future = model.make_future_dataframe(periods=1)

# Predict on the future dates
forecast = model.predict(future)
predictions_dict = {}
# Save the predictions to the dictionary
predictions_dict[code] = forecast


# - 아마 미래기간에 대한 데이터 프레임을 따로 생성해줘야할 듯
# - In sample test 진행 후에 out of sample test를 진행한다
# - 밑에 코드는 예시코드 

# In[130]:


from prophet import Prophet
# Initialize a dictionary to hold prediction dataframes
predictions_dict = {}

# Loop over each code
for code in unique_codes:
    # Initialize Prophet and fit the data
    model = Prophet()
    model.fit(data_dict[code])

    # Create a dataframe to hold the future dates
    future = model.make_future_dataframe(periods=15)

    # Predict on the future dates
    forecast = model.predict(future)

    # Save the predictions to the dictionary
    predictions_dict[code] = forecast

# Concatenate all prediction dataframes
df_predictions = pd.concat(predictions_dict.values())


# In[ ]:


# read pkl
df_predictions = pd.read_pickle(os.path.join(DATA_PATH, 'predictions_dict.pkl'))
# df_processed
true_y = df_processed[['code','close']][df_processed['date']=='2023-05-30']

# Initialize the DataFrame
final_df = pd.DataFrame(columns=['code', 'final_return_mean'])

for item in true_y['code']:
    y_true = true_y[true_y['code']==item]['close']
    y_hat = df_predictions[item]['yhat'][df_predictions[item]['ds']=='2023-05-30']
    diff = y_true.values[0] - y_hat.values[0]
    results = df_predictions[item]['yhat']+diff
    results = results[-15:]

    # Calculate final return
    final_return = (results.iloc[-1] - results.iloc[0]) / results.iloc[0]

    # Append to DataFrame
    final_df = final_df.append({'code': item, 'final_return_mean': final_return}, ignore_index=True)

# Print the final DataFrame
print(final_df)

final_df['rank'] = final_df['final_return_mean'].rank(method='first', ascending=False).astype('int')
final_df['code']= le.inverse_transform(final_df['code'].astype(int))
final_df.sort_values(by=['rank'], inplace=True)
final_df = final_df[['code', 'rank']]
final_df.rename(columns={'code': '종목코드', 'rank': '순위'}, inplace=True)
final_df.to_csv(RESULT_PATH, index=False)
final_df


# ## In-sample Forecast
# 
# -   트레인셋과 테스트 셋으로 확인하는 과정
# 
# 
# 

# In[177]:


# train set 마지막 15일 
last_1month_in = list()

for i in range(15,31):
    last_1month_in.append(['2023-05-%02d' % i])

last_1month_in = pd.DataFrame(last_1month_in, columns = ['ds'])
last_1month_in['ds']= pd.to_datetime(last_1month_in['ds'])


# In[ ]:


m.fit(data_dict[661])


# In[180]:


data_dict[661]


# In[181]:


tt = data_dict[661]
tt = tt.reset_index(drop=True)
tt


# In[184]:


last_1month_in


# In[191]:


get_ipython().system('brew install tbb')


# In[199]:


get_ipython().system('export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH')


# In[200]:


get_ipython().system('pip uninstall -y prophet')
get_ipython().system('pip install prophet')


# In[201]:


from prophet import Prophet


# In[202]:


# tt.reset_index(inplace=True)
# example
m = Prophet()
m.fit(tt)
# 에측
forecast = m.predict(last_1month_in)


# ## 메모리 확인

# In[187]:


get_ipython().system('pip install psutil')


# In[189]:


import psutil

# Get the memory details
memory_info = psutil.virtual_memory()

print(f"Total memory: {memory_info.total / (1024.0 ** 3)} GB")
print(f"Available memory: {memory_info.available / (1024.0 ** 3)} GB")
print(f"Used memory: {memory_info.used / (1024.0 ** 3)} GB")
print(f"Memory percent used: {memory_info.percent}%")


# In[245]:


get_ipython().system('pip install finance-datareader')


# In[246]:


get_ipython().system('pip install pykrx')


# In[ ]:





# In[ ]:





# #### 앞으로 할것 
# - validation data 구축 -> 비교 분석
# - shape ratio 공식
# - prophet / lightgbm / transformer 비교

# In[ ]:





# In[ ]:





# In[ ]:




