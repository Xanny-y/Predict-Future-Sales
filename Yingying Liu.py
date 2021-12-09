#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:33:08 2021

@author: yingyingliu
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from itertools import product
import lightgbm as lgb

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


sns.set(style="darkgrid")

%matplotlib inline 

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)

# import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm_notebook

from itertools import product

def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df

# Data Loading 
path = '/Users/yingyingliu/Desktop/predict-future-sales/'

df_train = pd.read_csv(path + 'sales_train.csv')
df_train.head()
df_test = pd.read_csv(path + 'test.csv')
# df_test.head()
df_shops = pd.read_csv(path + 'shops.csv')
df_shops.head(15)
df_items = pd.read_csv(path + 'items.csv')
# df_items.head()
df_items_categories = pd.read_csv(path + 'item_categories.csv')
# df_items_categories.head()
df_submission = pd.read_csv(path + 'sample_submission.csv')
# df_submission.head()

# Number of NaNs for each columns
print('df_train shape', df_train.shape)
print('df_shops shape', df_train.shape)
df_shops.isnull().sum(axis=0).head()
df_train.isnull().sum(axis=0).head(15)


# Delete outlier from df_train
df_train.describe()

sns.boxplot(x = df_train.item_cnt_day)  # 1,000
sns.boxplot(x = df_train.item_price)    # 100,000

df_train = df_train[(df_train.item_price < 100000) & (df_train.item_price > 0)]
train = df_train[df_train.item_cnt_day < 1001]


# detect same shop_names with different shop_ids
print(df_shops[df_shops.shop_id.isin([0, 57])]['shop_name'])  # 57 = 0
print(df_shops[df_shops.shop_id.isin([1, 58])]['shop_name'])
print(df_shops[df_shops.shop_id.isin([39, 40])]['shop_name'])
print(df_shops[df_shops.shop_id.isin([10, 11])]['shop_name'])


# data cleaning #
''' train '''
df_shops['shop_id'].nunique()

train.loc[train.shop_id == 0, 'shop_id'] = 57
df_test.loc[df_test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
df_test.loc[df_test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 40, 'shop_id'] = 39
df_test.loc[df_test.shop_id == 40, 'shop_id'] = 39
train.loc[train.shop_id == 10, 'shop_id'] = 11
df_test.loc[df_test.shop_id == 10, 'shop_id'] = 11

# train["revenue"] = train["item_cnt_day"] * train["item_price"]

u_df_test_id = df_test['shop_id'].unique()
train = train[train['shop_id'].isin(u_df_test_id)]
train['shop_id'].nunique()

''' df_shops '''
df_shops["city"] = df_shops.shop_name.str.split(" ").map(lambda x: x[0])
df_shops["city"]
df_shops.loc[df_shops.city == "!Якутск", "city"] = "Якутск"
df_shops['category'] = df_shops.shop_name.str.split(" ").map(lambda x: x[1])  # df_shop['shop_name'][1] = category
df_shops.head()
df_shops['category']

# Only keep shop category if there are 5 or more shops of that category
category = []    
for cat in df_shops.category.unique():
    if len(df_shops[df_shops.category == cat]) >= 5:
           category.append(cat)
df_shops.category = df_shops.category.apply(lambda x: x if (x in category) 
                                            else "others")


label_encoder = LabelEncoder()
df_shops['city'] = label_encoder.fit_transform(df_shops['city'])
df_shops = df_shops.drop('shop_name', axis = 1)
df_shops['category'] = label_encoder.fit_transform(df_shops['category'])     
shops = df_shops
shops.head()


'''df_items_categories'''
df_items_categories.head()
df_items_categories['type'] = df_items_categories['item_category_name'].apply(lambda x: x.split()[0]).astype(str)
df_items_categories['type'].value_counts()
df_items_categories.head()

category = []
for cat in df_items_categories['type'].unique():
    if len(df_items_categories[df_items_categories.type == cat]) >= 5:
        category.append(cat)
df_items_categories.type = df_items_categories.type.apply(lambda x: x if (x in category) else 'etc')

df_items_categories.type = LabelEncoder().fit_transform(df_items_categories.type)
df_items_categories.head()

cats = df_items_categories[['type', 'item_category_id']]
cats.head()


'''items'''
df_items.head()
df_items = df_items.drop('item_name', axis = 1)
# Create the date the product was first sold as a feature
df_items['first_sale'] = train.groupby('item_id').agg({'date_block_num': 'min'})['date_block_num']
df_items[df_items['first_sale'].isna()]
df_items['first_sale'] = df_items['first_sale'].fillna(34)
df_items.head()
# items = pd.merge(cats, df_items, on = 'item_category_id')
# items = items.drop('item_category_name', axis = 1)



# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']
# For every month we create a grid from all shops/items combinations from that month

#### import itertools from product
grid = []
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[train['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = train.loc[train['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype = 'int32'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)


# groupby('index_cols')? 
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': 'sum'})
group = group.reset_index()
group = group.rename(columns = {'item_cnt_day': 'item_cnt_month'})

all_data = pd.merge(grid, group, on = index_cols, how = 'left').fillna(0)
all_data.avg_shop_price = (all_data.avg_shop_price.fillna(0).astype(np.float16))



# concate test data with training data which has been cleaned before
df_test.head()
df_test['date_block_num'] = 34
df_test['date_block_num'] = df_test['date_block_num'].astype(np.int8)
all_data = pd.concat([all_data, df_test.drop('ID', axis = 1)],
                     ignore_index = True,
                     keys = index_cols)
# Replace NaN with 0
all_data = all_data.fillna(0)


# concatenate shop, item, etc data
all_data = pd.merge(all_data, shops, on = 'shop_id', how = 'left')
all_data = pd.merge(all_data, df_items, on = 'item_id', how = 'left')
all_data = pd.merge(all_data, cats, on = 'item_category_id', how = 'left')

# Downcast dtypes from 64 to 32 bit to save memory
all_data = downcast_dtypes(all_data)
del grid, group 
group.collect()



# feature summary
def resumetable(df):
    print(f'Data Shape: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['Dtypes'])
    summary['Null'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First_values'] = df.loc[0].values
    summary['Second_values'] = df.loc[1].values
    summary['Third_values'] = df.loc[2].values
    
    return summary

resumetable(all_data)


# Basic comprehension of cnt
figure, ax= plt.subplots() 
figure.set_size_inches(11, 5)
# Total item sales/ shop_id
group_shop_sum = all_data.groupby('shop_id').agg({'item_cnt_month': 'sum'})
group_shop_sum = group_shop_sum.reset_index()

group_shop_sum = group_shop_sum[group_shop_sum['item_cnt_month'] > 10000]

sns.barplot(x='shop_id', y='item_cnt_month', data=group_shop_sum)
ax.set(title='Distribution of total item counts by shop id',
       xlabel='Shop ID', 
       ylabel='Total item counts')
ax.tick_params(axis='x', labelrotation=90)




# Target Lags
def lag_feature( df,lags, cols ):
    for col in cols:
        print(col)
        tmp = df[["date_block_num", "shop_id", "item_id", col ]]
        for i in lags:
            shifted = tmp.copy()
            shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_" + str(i)]
            shifted.date_block_num = shifted.date_block_num + i
            df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df


 # Add lag-'item_cnt_month' 
## all_data = lag_feature(all_data, [1,2,3], ["avg_shop_price"])   # Add lag-'avg_shop_price'
## all_data = lag_feature(all_data, [1,2,3], ['avg_item_price'])   # Add lag-'avg_item_price'
all_data = lag_feature(all_data, [1,2,3], ["item_cnt_month"])

# mean encoder based on n_splits
# previous month's avg_items
group = all_data.groupby(["date_block_num"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_avg_item"]
group.reset_index(inplace = True)

all_data =pd.merge(all_data, group, on = ['date_block_num'], how = 'left')
all_data.data_avg_item = all_data["date_avg_item"].astype(np.float16)

all_data = lag_feature(all_data, [1], ["date_avg_item"])
all_data.drop(['date_avg_item'], axis = 1, inplace = True) # lag = 1 bacause of correlation


# Add lag features for avg_item_month
group = all_data.groupby(["date_block_num", "item_id"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["avg_item_cnt_month"]
group.reset_index(inplace = True)

all_data = pd.merge(all_data, group, on = ['date_block_num', 'item_id'], how = 'left')
all_data["avg_item_cnt_month"] = all_data["avg_item_cnt_month"].astype(np.float16)

all_data = lag_feature(all_data, [1,2,3], ["avg_item_cnt_month"])
all_data.drop(["avg_item_cnt_month"], axis = 1, inplace = True)






# add lag features for month&shop
group = all_data.groupby(["date_block_num", "shop_id"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["avg_date_shop_item"]
group.reset_index(inplace = True)

all_data = pd.merge(all_data, group, on = ['date_block_num', 'shop_id'], how = 'left')
all_data["avg_date_shop_item"] = all_data["avg_date_shop_item"].astype(np.float16)

all_data = lag_feature(all_data, [1,2,3], ["avg_date_shop_item"])
all_data.drop(["avg_date_shop_item"], axis = 1, inplace = True)






# Add lag features for avg_item_month_cnt
group = all_data.groupby(["date_block_num", "shop_id", "item_id"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_shop_avg_item_cnt"]
group.reset_index(inplace = True)

all_data = pd.merge(all_data, group, on = ["date_block_num", "shop_id", "item_id"], how = 'left')
all_data["date_shop_avg_item_cnt"] = all_data["date_shop_avg_item_cnt"].astype(np.float16)

all_data = lag_feature(all_data, [1], ["date_shop_avg_item_cnt"])
all_data.drop(["date_shop_avg_item_cnt"], axis = 1, inplace = True)


# avg_date_shop_item
group = all_data.groupby(["date_block_num", "city"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_city_avg_item"]
group.reset_index(inplace = True)

all_data = pd.merge(all_data, group, on = ['date_block_num', 'city'], how = 'left')
all_data["date_city_avg_item"] = all_data["date_city_avg_item"].astype(np.float16)

all_data = lag_feature(all_data, [1], ["date_city_avg_item"])
all_data.drop(["date_city_avg_item"], axis = 1, inplace = True)



# revenue
all_data["revenue"] = train["item_cnt_day"]*train["item_price"]

group = all_data.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

all_data = pd.merge(all_data, group, on=['date_block_num','shop_id'], how='left')
all_data['date_shop_revenue'] = all_data['date_shop_revenue'].astype(np.float32)

### shop_avg_revenue
group = all_data.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

all_data = pd.merge(all_data, group, on=['shop_id'], how='left')
all_data['shop_avg_revenue'] = all_data['shop_avg_revenue'].astype(np.float32)



### delta_revenue denotes the difference between paticular avg_revenue and avg(all_shop_avg_revenue)
all_data['delta_revenue'] = (all_data['date_shop_revenue'] - all_data['shop_avg_revenue']) / all_data['shop_avg_revenue']
all_data['delta_revenue'] = all_data['delta_revenue'].astype(np.float32)

all_data = lag_feature(all_data, [1], ['delta_revenue'])
all_data.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)


### item based trends ###
group = train.groupby( ["item_id"] ).agg({"item_price": ["mean"]})
group.columns = ["item_avg_item_price"]
group.reset_index(inplace = True)

all_data = all_data.merge( group, on = ["item_id"], how = "left" )
all_data["item_avg_item_price"] = all_data.item_avg_item_price.astype(np.float32)


group = train.groupby( ["date_block_num","item_id"] ).agg( {"item_price": ["mean"]} )
group.columns = ["date_item_avg_item_price"]
group.reset_index(inplace = True)

all_data = all_data.merge(group, on = ["date_block_num","item_id"], how = "left")
all_data["date_item_avg_item_price"] = all_data.date_item_avg_item_price.astype(np.float32)

lags = [1, 2, 3]
all_data = lag_feature(all_data, lags, ["date_item_avg_item_price"] )
for i in lags:
    all_data["delta_price_lag_" + str(i) ] = (all_data["date_item_avg_item_price_lag_" + 
                                                       str(i)]- all_data["item_avg_item_price"] )/ all_data["item_avg_item_price"]
all_data.head()

def select_trends(row) :
    for i in lags:
        if row["delta_price_lag_" + str(i)]:
            return row["delta_price_lag_" + str(i)]
    return 0


all_data["delta_price_lag"] = all_data.apply(select_trends, axis = 1)
all_data["delta_price_lag"] = all_data.delta_price_lag.astype( np.float32)
all_data["delta_price_lag"].fillna(0 ,inplace = True)

features_to_drop = ["item_avg_item_price", "date_item_avg_item_price"]
for i in lags:
    features_to_drop.append("date_item_avg_item_price_lag_" + str(i) )
    features_to_drop.append("delta_price_lag_" + str(i) )
all_data.drop(features_to_drop, axis = 1, inplace = True)


# Special Feature
all_data.month = all_data["date_block_num"] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
all_data['days'] = all_data.month.map(days).astype(np.int8)


all_data = all_data[all_data["date_block_num"] > 3] # no lag_variables for first three months
all_data.head().T

# fill NA brought by lag_variables generation
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

all_data = fill_na(all_data)

all_data.columns
all_data.shape
all_data.info
all_data.describe


# all_data = preprocessing.scale(all_data)   # no need to normalize data
# all_data = pd.DataFrame(all_data)
# all_data.describe

# Training data 
X_train = all_data[all_data['date_block_num'] < 33].drop(["item_cnt_month"], axis = 1)
Y_train = all_data[all_data.date_block_num < 33]['item_cnt_month']
# Validation
X_valid = all_data[all_data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = all_data[all_data.date_block_num == 33]['item_cnt_month']

X_test = all_data[all_data.date_block_num == 34].drop(['item_cnt_month'], axis=1)



# Xgboost
model = XGBRegressor(
    max_depth=6,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

model.fit


Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": Y_test, 
    "item_cnt_month": Y_test
})
# submission.to_csv('xgb_submission.csv', index=False)


plot_features(model, (10,14))


# lightgbm
# lgb hyper-parameters
params = {'metric': 'rmse',
          'num_leaves': 255,
          'learning_rate': 0.005,
          'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'bagging_freq': 5,
          'force_col_wise' : True,
          'random_state': 10}

cat_features = ['shop_id', 'city', 'item_category_id', 'category', 'days']

# lgb train and valid dataset
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)
 
# Train LightGBM model
lgb_model = lgb.train(params=params,
                      train_set=dtrain,
                      num_boost_round=1500,
                      valid_sets=(dtrain, dvalid),
                      early_stopping_rounds=150,
                      categorical_feature=cat_features,
                      verbose_eval=100) 
# training's rmse: 2.22276	valid_1's rmse: 1.67866