import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from subprocess import check_output

def read_data(name):
    frames = pd.read_csv(name)
    print("Data successfully read!")
    return frames

#memory reduction functions
def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

#use train_v2 or sample_submission_v2 depending on which file u wanna compile
#data = read_data("data/sample_submission_v2.csv")    

user_log = read_data("data/user_logs_v2.csv")
#memory reduction
change_datatype(user_log)   
change_datatype_float(user_log)
#mem = user_log.memory_usage(index=True).sum()    
#print(mem/ 1024**2,"MB")
user_log = user_log.drop(['date', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100'], axis=1)
#print(user_log.head())


comb = read_data("data/df_comb.csv")
change_datatype(comb)   
change_datatype_float(comb)
#print(comb.shape)

#remove some unneeded columns
comb = comb.drop(['transaction_date', 'membership_expire_date', 'gender', 'bd', 'registration_init_time'], axis=1)

train_data = pd.merge(comb, user_log, on='msno', how='outer')

del comb
del user_log

#normalization
train_data['amt_per_day'] = train_data['amt_per_day'].replace(to_replace=float('inf'),value = 0)
train_data['payment_plan_days'] = (train_data['payment_plan_days'] - train_data['payment_plan_days'].min())/(train_data['payment_plan_days'].max() - train_data['payment_plan_days'].min())
train_data['plan_list_price'] = (train_data['plan_list_price'] - train_data['plan_list_price'].min())/(train_data['plan_list_price'].max() - train_data['plan_list_price'].min())
train_data['actual_amount_paid'] = (train_data['actual_amount_paid'] - train_data['actual_amount_paid'].min())/(train_data['actual_amount_paid'].max() - train_data['actual_amount_paid'].min())

#aggregation
train_data = train_data.groupby('msno', as_index=False).mean()
print(train_data.shape)
train_data.to_csv("data/df_train.csv", index=False)

















