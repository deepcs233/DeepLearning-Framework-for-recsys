# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import RandomizedSearchCV
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import time
from sklearn.metrics import log_loss

def time2cov(time_):
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))

def map_hour(x):
    if (x>=7)&(x<=12):
        return 1
    elif (x>=13)&(x<=20):
        return 2
    else:
        return 3

from conf import TEST_DATA_PATH
from conf import TRAIN_DATA_PATH
TRAIN_DATA_PATH = 'data/round1_ijcai_18_train_20180301.txt'
TEST_DATA_PATH = 'data/round1_ijcai_18_test_a_20180301.txt'
filepath1 = TRAIN_DATA_PATH
traindata = pd.read_csv(filepath1,sep = ' ')
trainlabel = traindata['is_trade']

filepath2 = TEST_DATA_PATH
testdata = pd.read_csv(filepath2, sep = ' ')
print(traindata.shape,trainlabel.shape,testdata.shape)

traindata['context_timestamp'] = traindata['context_timestamp'].apply(time2cov)
traindata['context_timestamp_tmp'] = pd.to_datetime(traindata['context_timestamp'])
traindata['hour'] = traindata.context_timestamp_tmp.dt.hour
traindata['day'] = traindata.context_timestamp_tmp.dt.day
testdata['context_timestamp'] = testdata['context_timestamp'].apply(time2cov)
testdata['context_timestamp_tmp'] = pd.to_datetime(testdata['context_timestamp'])
testdata['hour'] = testdata['context_timestamp_tmp'].dt.hour
testdata['day'] = testdata['context_timestamp_tmp'].dt.day
traindata['hour_seg'] = traindata['hour'].apply(map_hour)
testdata['hour_seg'] = testdata['hour'].apply(map_hour)

del traindata['context_timestamp_tmp']
del traindata['context_timestamp']
del testdata['context_timestamp_tmp']
del testdata['context_timestamp']
print(traindata.shape,trainlabel.shape,testdata.shape,0)



#=========================
#处理item_id列
#========================
counts = pd.DataFrame(pd.value_counts(traindata['item_id']))
counts1 = pd.DataFrame(pd.value_counts(testdata['item_id']))
traindata['item_id'] = traindata['item_id'].replace(counts.index.tolist(),counts['item_id'].tolist())
testdata['item_id'] = testdata['item_id'].replace(counts1.index.tolist(),counts1['item_id'].tolist())
print(4)

#========================
#处理item_brand_id列
#========================
counts = pd.DataFrame(pd.value_counts(traindata['item_brand_id']))
counts1 = pd.DataFrame(pd.value_counts(testdata['item_brand_id']))
traindata['item_brand_id'] = traindata['item_brand_id'].replace(counts.index.tolist(),counts['item_brand_id'].tolist())
testdata['item_brand_id'] = testdata['item_brand_id'].replace(counts1.index.tolist(),counts1['item_brand_id'].tolist())
print(6)

#=========================
#处理shop_id列
#========================
counts = pd.DataFrame(pd.value_counts(traindata['shop_id']))
counts1 = pd.DataFrame(pd.value_counts(testdata['shop_id']))
traindata['shop_id'] = traindata['shop_id'].replace(counts.index.tolist(),counts['shop_id'].tolist())
testdata['shop_id'] = testdata['shop_id'].replace(counts1.index.tolist(),counts1['shop_id'].tolist())
print(10)


#=========================
#处理city列
#========================
counts = pd.DataFrame(pd.value_counts(traindata['item_city_id']))
counts1 = pd.DataFrame(pd.value_counts(testdata['item_city_id']))
traindata['item_city_id'] = traindata['item_city_id'].replace(counts.index.tolist(),counts['item_city_id'].tolist())
testdata['item_city_id'] = testdata['item_city_id'].replace(counts1.index.tolist(),counts1['item_city_id'].tolist())
print(2)

del traindata['context_id']
del testdata['context_id']

counts = pd.DataFrame(pd.value_counts(traindata['user_id']))
traindata['user_id'] = traindata['user_id'].replace(counts.index.tolist(),counts['user_id'].tolist())
counts = pd.DataFrame(pd.value_counts(testdata['user_id']))
testdata['user_id'] = testdata['user_id'].replace(counts.index.tolist(),counts['user_id'].tolist())
print(traindata.shape,testdata.shape)

traindata['user_gender_id'] = abs(traindata['user_gender_id'])
testdata['user_gender_id'] = abs(testdata['user_gender_id'])





testdata['len_item_category'] = testdata['item_category_list'].map(lambda x: len(str(x).split(';')))
traindata['len_item_category'] = traindata['item_category_list'].map(lambda x: len(str(x).split(';')))
traindata['len_item_property'] = traindata['item_property_list'].map(lambda x: len(str(x).split(';')))
testdata['len_item_property'] = testdata['item_property_list'].map(lambda x: len(str(x).split(';')))
testdata['len_predict_category_property'] = testdata['predict_category_property'].map(lambda x: len(str(x).split(';')))
traindata['len_predict_category_property'] = traindata['predict_category_property'].map(lambda x: len(str(x).split(';')))
for i in range(8):
    traindata['property_%d'%(i)] = traindata['item_property_list'].apply(
        lambda x:x.split(";")[i] if len(x.split(";")) > i else " "
    )
for i in range(8):
    testdata['property_%d'%(i)] = testdata['item_property_list'].apply(
        lambda x:x.split(";")[i] if len(x.split(";")) > i else " "
    )

for i in range(8):
    counts = pd.DataFrame(pd.value_counts(traindata['property_'+str(i)]))
    counts1 = pd.DataFrame(pd.value_counts(testdata['property_'+str(i)]))
    traindata['property_'+str(i)] = traindata['property_'+str(i)].replace(counts.index.tolist(),counts['property_'+str(i)].tolist())
    testdata['property_'+str(i)] = testdata['property_'+str(i)].replace(counts1.index.tolist(),counts1['property_'+str(i)].tolist())

for i in range(8):
    traindata['predict_category_%d'%(i)] = traindata['predict_category_property'].apply(
        lambda x:str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "
    )
for i in range(8):
    testdata['predict_category_%d'%(i)] = testdata['predict_category_property'].apply(
        lambda x:str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "
    )

for i in range(8):
    counts = pd.DataFrame(pd.value_counts(traindata['predict_category_%d'%(i)]))
    counts1 = pd.DataFrame(pd.value_counts(testdata['predict_category_%d'%(i)]))
    traindata['predict_category_%d'%(i)] = traindata['predict_category_%d'%(i)].replace(counts.index.tolist(),counts['predict_category_%d'%(i)].tolist())
    testdata['predict_category_%d'%(i)] = testdata['predict_category_%d'%(i)].replace(counts1.index.tolist(),counts1['predict_category_%d'%(i)].tolist())
sub_data =  pd.DataFrame((x.split(';') for x in traindata.item_category_list),index = traindata.index,columns = ['item_category_list_1','item_category_list_2','item_category_list_3']  )
traindata = pd.concat([traindata,sub_data],axis = 1)
counts = pd.DataFrame(pd.value_counts(traindata['item_category_list_2']))
traindata['item_category_list_2'] = traindata['item_category_list_2'].replace(counts.index.tolist(),counts['item_category_list_2'].tolist())

sub_data =  pd.DataFrame((x.split(';') for x in testdata.item_category_list),index = testdata.index,columns = ['item_category_list_1','item_category_list_2','item_category_list_3']  )
testdata = pd.concat([testdata,sub_data],axis = 1)
testdata['item_category_list_2'] = testdata['item_category_list_2'].replace(counts.index.tolist(),counts['item_category_list_2'].tolist())
traindata['item_category_list_3'] = traindata['item_category_list_3'].fillna(1)
testdata['item_category_list_3'] = testdata['item_category_list_3'].fillna(1)
counts = pd.DataFrame(pd.value_counts(traindata['item_category_list_3']))
traindata['item_category_list_3'] = traindata['item_category_list_3'].replace(counts.index.tolist(),counts['item_category_list_3'].tolist())
testdata['item_category_list_3'] = testdata['item_category_list_3'].replace(counts.index.tolist(),counts['item_category_list_3'].tolist())

del traindata['predict_category_property']
del testdata['predict_category_property']
del traindata['item_property_list']
del testdata['item_property_list']
del traindata['item_category_list']
del traindata['item_category_list_1']
del testdata['item_category_list_1']
del testdata['item_category_list']
print(traindata.shape,trainlabel.shape,testdata.shape,3)



##=========================
##处理几个列
##========================
traindata['shop_star_level0'] = traindata['shop_star_level'].apply(lambda x: 2 if x > 5015 else x)
traindata['shop_star_level0'] = traindata['shop_star_level0'].apply(lambda x: 1 if 5015 >= x > 5011 else x)
traindata['shop_star_level0'] = traindata['shop_star_level0'].apply(lambda x: 0 if  x <= 5011 else x)

traindata['context_page_id0'] = traindata['context_page_id'].apply(lambda x: 2 if x > 4009 else x)
traindata['context_page_id0'] = traindata['context_page_id0'].apply(lambda x: 1 if 4009 >= x > 4001 else x)
traindata['context_page_id0'] = traindata['context_page_id0'].apply(lambda x: 0 if  x == 4001 else x)

traindata['user_star_level0'] = traindata['user_star_level'].apply(lambda x: 2 if x >= 3007 else x)
traindata['user_star_level0'] = traindata['user_star_level0'].apply(lambda x: 1 if 3007 > x > 3002 else x)
traindata['user_star_level0'] = traindata['user_star_level0'].apply(lambda x: 0 if  x <= 3002 else x)

traindata['user_occupation_id0'] = traindata['user_occupation_id'].apply(lambda x: 1 if x == 2005 else 0)

traindata['user_age_level0'] = traindata['user_age_level'].apply(lambda x: 2 if x >= 1006 else x)
traindata['user_age_level0'] = traindata['user_age_level0'].apply(lambda x: 1 if 1006 > x > 1002 else x)
traindata['user_age_level0'] = traindata['user_age_level0'].apply(lambda x: 0 if  x <= 1002 else x)

traindata['shop_review_num_level0'] = traindata['shop_review_num_level'].apply(lambda x: 2 if x > 17 else x)
traindata['shop_review_num_level0'] = traindata['shop_review_num_level0'].apply(lambda x: 1 if 17 >= x > 13 else x)
traindata['shop_review_num_level0'] = traindata['shop_review_num_level0'].apply(lambda x: 0 if  x <= 13 else x)


testdata['shop_star_level0'] = testdata['shop_star_level'].apply(lambda x: 2 if x > 5015 else x)
testdata['shop_star_level0'] = testdata['shop_star_level0'].apply(lambda x: 1 if 5015 >= x > 5011 else x)
testdata['shop_star_level0'] = testdata['shop_star_level0'].apply(lambda x: 0 if  x <= 5011 else x)

testdata['context_page_id0'] = testdata['context_page_id'].apply(lambda x: 2 if x > 4009 else x)
testdata['context_page_id0'] = testdata['context_page_id0'].apply(lambda x: 1 if 4009 >= x > 4001 else x)
testdata['context_page_id0'] = testdata['context_page_id0'].apply(lambda x: 0 if  x == 4001 else x)

testdata['user_star_level0'] = testdata['user_star_level'].apply(lambda x: 2 if x >= 3007 else x)
testdata['user_star_level0'] = testdata['user_star_level0'].apply(lambda x: 1 if 3007 > x > 3002 else x)
testdata['user_star_level0'] = testdata['user_star_level0'].apply(lambda x: 0 if  x <= 3002 else x)

testdata['user_occupation_id0'] = testdata['user_occupation_id'].apply(lambda x: 1 if x == 2005 else 0)

testdata['user_age_level0'] = testdata['user_age_level'].apply(lambda x: 2 if x >= 1006 else x)
testdata['user_age_level0'] = testdata['user_age_level0'].apply(lambda x: 1 if 1006 > x > 1002 else x)
testdata['user_age_level0'] = testdata['user_age_level0'].apply(lambda x: 0 if  x <= 1002 else x)

testdata['shop_review_num_level0'] = testdata['shop_review_num_level'].apply(lambda x: 2 if x > 17 else x)
testdata['shop_review_num_level0'] = testdata['shop_review_num_level0'].apply(lambda x: 1 if 17 >= x > 13 else x)
testdata['shop_review_num_level0'] = testdata['shop_review_num_level0'].apply(lambda x: 0 if  x <= 13 else x)


traindata['user_age_level'] = traindata['user_age_level'].replace(-1,traindata['user_age_level'].median())
traindata['user_age_level'] = traindata['user_age_level']-1000
testdata['user_age_level'] = testdata['user_age_level'].replace(-1,testdata['user_age_level'].median())
testdata['user_age_level'] = testdata['user_age_level']-1000

traindata['user_occupation_id'] = traindata['user_occupation_id'].replace(-1,traindata['user_occupation_id'].median())
traindata['user_occupation_id'] = traindata['user_occupation_id']-2000
testdata['user_occupation_id'] = testdata['user_occupation_id'].replace(-1,testdata['user_occupation_id'].median())
testdata['user_occupation_id'] = testdata['user_occupation_id']-2000


traindata['user_star_level'] = traindata['user_star_level'].replace(-1,traindata['user_star_level'].median())
traindata['user_star_level'] = traindata['user_star_level']-3000
testdata['user_star_level'] = testdata['user_star_level'].replace(-1,testdata['user_star_level'].median())
testdata['user_star_level'] = testdata['user_star_level']-3000

traindata['context_page_id'] = traindata['context_page_id'].replace(-1,traindata['context_page_id'].median())
traindata['context_page_id'] = traindata['context_page_id'] - 4000
testdata['context_page_id'] = testdata['context_page_id'].replace(-1,testdata['context_page_id'].median())
testdata['context_page_id'] = testdata['context_page_id'] - 4000

traindata['shop_star_level'] = traindata['shop_star_level'].replace(-1,traindata['context_page_id'].median())
traindata['shop_star_level'] = traindata['shop_star_level'] - 5000
testdata['shop_star_level'] = testdata['shop_star_level'].replace(-1,testdata['context_page_id'].median())
testdata['shop_star_level'] = testdata['shop_star_level'] - 5000



traindata['shop_score_description0'] = traindata['shop_score_description'].apply(lambda x: 2 if x > 0.984 else x)
traindata['shop_score_description0'] = traindata['shop_score_description0'].apply(lambda x: 1 if 0.984 >= x > 0.97 else x)
traindata['shop_score_description0'] = traindata['shop_score_description0'].apply(lambda x: 0 if  x <= 0.97 else x)

traindata['shop_score_delivery0'] = traindata['shop_score_delivery'].apply(lambda x: 2 if x > 0.979 else x)
traindata['shop_score_delivery0'] = traindata['shop_score_delivery0'].apply(lambda x: 1 if 0.979 >= x > 0.966 else x)
traindata['shop_score_delivery0'] = traindata['shop_score_delivery0'].apply(lambda x: 0 if  x <= 0.966 else x)

traindata['shop_score_service0'] = traindata['shop_score_service'].apply(lambda x: 2 if x > 0.979 else x)
traindata['shop_score_service0'] = traindata['shop_score_service0'].apply(lambda x: 1 if 0.979 >= x > 0.967 else x)
traindata['shop_score_service0'] = traindata['shop_score_service0'].apply(lambda x: 0 if  x <= 0.967 else x)

traindata['shop_review_positive_rate0'] = traindata['shop_review_positive_rate'].apply(lambda x: 2 if x == 1 else x)
traindata['shop_review_positive_rate0'] = traindata['shop_review_positive_rate0'].apply(lambda x: 1 if 1 > x > 0.98 else x)
traindata['shop_review_positive_rate0'] = traindata['shop_review_positive_rate0'].apply(lambda x: 0 if  x <= 0.98 else x)

traindata['item_price_level0'] = traindata['item_price_level'].apply(lambda x: 2 if x >= 9 else x)
traindata['item_price_level0'] = traindata['item_price_level0'].apply(lambda x: 1 if 9 > x > 5 else x)
traindata['item_price_level0'] = traindata['item_price_level0'].apply(lambda x: 0 if  x <= 5 else x)

traindata['item_sales_level0'] = traindata['item_sales_level'].apply(lambda x: 2 if x >= 14 else x)
traindata['item_sales_level0'] = traindata['item_sales_level0'].apply(lambda x: 1 if 14 > x > 9 else x)
traindata['item_sales_level0'] = traindata['item_sales_level0'].apply(lambda x: 0 if  x <= 9 else x)

traindata['item_collected_level0'] = traindata['item_collected_level'].apply(lambda x: 2 if x >= 15 else x)
traindata['item_collected_level0'] = traindata['item_collected_level0'].apply(lambda x: 1 if 15 > x > 10 else x)
traindata['item_collected_level0'] = traindata['item_collected_level0'].apply(lambda x: 0 if  x <= 10 else x)

traindata['item_collected_level0'] = traindata['item_collected_level'].apply(lambda x: 2 if x >= 15 else x)
traindata['item_collected_level0'] = traindata['item_collected_level0'].apply(lambda x: 1 if 15 > x > 10 else x)
traindata['item_collected_level0'] = traindata['item_collected_level0'].apply(lambda x: 0 if  x <= 10 else x)

traindata['item_pv_level0'] = traindata['item_pv_level'].apply(lambda x: 2 if x >= 20 else x)
traindata['item_pv_level0'] = traindata['item_pv_level0'].apply(lambda x: 1 if 20 > x > 14 else x)
traindata['item_pv_level0'] = traindata['item_pv_level0'].apply(lambda x: 0 if  x <= 14 else x)



testdata['shop_score_description0'] = testdata['shop_score_description'].apply(lambda x: 2 if x > 0.984 else x)
testdata['shop_score_description0'] = testdata['shop_score_description0'].apply(lambda x: 1 if 0.984 >= x > 0.97 else x)
testdata['shop_score_description0'] = testdata['shop_score_description0'].apply(lambda x: 0 if  x <= 0.97 else x)

testdata['shop_score_delivery0'] = testdata['shop_score_delivery'].apply(lambda x: 2 if x > 0.979 else x)
testdata['shop_score_delivery0'] = testdata['shop_score_delivery0'].apply(lambda x: 1 if 0.979 >= x > 0.966 else x)
testdata['shop_score_delivery0'] = testdata['shop_score_delivery0'].apply(lambda x: 0 if  x <= 0.966 else x)

testdata['shop_score_service0'] = testdata['shop_score_service'].apply(lambda x: 2 if x > 0.979 else x)
testdata['shop_score_service0'] = testdata['shop_score_service0'].apply(lambda x: 1 if 0.979 >= x > 0.967 else x)
testdata['shop_score_service0'] = testdata['shop_score_service0'].apply(lambda x: 0 if  x <= 0.967 else x)

testdata['shop_review_positive_rate0'] = testdata['shop_review_positive_rate'].apply(lambda x: 2 if x == 1 else x)
testdata['shop_review_positive_rate0'] = testdata['shop_review_positive_rate0'].apply(lambda x: 1 if 1 > x > 0.98 else x)
testdata['shop_review_positive_rate0'] = testdata['shop_review_positive_rate0'].apply(lambda x: 0 if  x <= 0.98 else x)

testdata['item_price_level0'] = testdata['item_price_level'].apply(lambda x: 2 if x >= 9 else x)
testdata['item_price_level0'] = testdata['item_price_level0'].apply(lambda x: 1 if 9 > x > 5 else x)
testdata['item_price_level0'] = testdata['item_price_level0'].apply(lambda x: 0 if  x <= 5 else x)

testdata['item_sales_level0'] = testdata['item_sales_level'].apply(lambda x: 2 if x >= 14 else x)
testdata['item_sales_level0'] = testdata['item_sales_level0'].apply(lambda x: 1 if 14 > x > 9 else x)
testdata['item_sales_level0'] = testdata['item_sales_level0'].apply(lambda x: 0 if  x <= 9 else x)

testdata['item_collected_level0'] = testdata['item_collected_level'].apply(lambda x: 2 if x >= 15 else x)
testdata['item_collected_level0'] = testdata['item_collected_level0'].apply(lambda x: 1 if 15 > x > 10 else x)
testdata['item_collected_level0'] = testdata['item_collected_level0'].apply(lambda x: 0 if  x <= 10 else x)

testdata['item_collected_level0'] = testdata['item_collected_level'].apply(lambda x: 2 if x >= 15 else x)
testdata['item_collected_level0'] = testdata['item_collected_level0'].apply(lambda x: 1 if 15 > x > 10 else x)
testdata['item_collected_level0'] = testdata['item_collected_level0'].apply(lambda x: 0 if  x <= 10 else x)

testdata['item_pv_level0'] = testdata['item_pv_level'].apply(lambda x: 2 if x >= 20 else x)
testdata['item_pv_level0'] = testdata['item_pv_level0'].apply(lambda x: 1 if 20 > x > 14 else x)
testdata['item_pv_level0'] = testdata['item_pv_level0'].apply(lambda x: 0 if  x <= 14 else x)
