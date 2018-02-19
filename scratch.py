import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from subprocess import check_output


def normalize(df, names):
    for name in names:
        df[name] = ((df[name] - df[name].mean())/df[name].std())
    return df

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
    return df

def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
    return df

def normalize(df, names):
    for name in names:
           df[name] = ((df[name] - df[name].mean())/df[name].std())
    return df


user_log = read_data("data/user_logs_v2.csv")
user_log = user_log.drop(['date', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100'], axis=1)

#memory reduction
user_log = change_datatype(user_log)   
user_log = change_datatype_float(user_log)
#mem = user_log.memory_usage(index=True).sum()    
#print(mem/ 1024**2,"MB")
print(user_log.shape)


comb = read_data("data/df_comb.csv")
#remove some unneeded columns and save mem
comb = change_datatype(comb)   
comb = change_datatype_float(comb)
print(comb.shape)

data = pd.merge(comb, user_log, on='msno', how = 'outer')
print(data.shape)
del comb
del user_log

#train_data = read_data("data/sample_submission_v2.csv")
train_data = read_data("data/train_v2.csv")
print(train_data.shape)

train_data = pd.merge(train_data, data, on='msno', how = 'left')
train_data = train_data.replace(to_replace=float('inf'),value = 0)
train_data = train_data.fillna(value=0)
#normalization
train_data = normalize(train_data, ['amt_per_day', 'payment_plan_days','plan_list_price', 'actual_amount_paid','plan_list_price', 'payment_plan_days', 'actual_amount_paid', 'amt_per_day', 'membership_duration', 'discount'])

#aggregation
train_data = train_data.groupby('msno', as_index=False).mean()

print(train_data.shape)
#train_data.to_csv("data/df_testfinal.csv", index=False)
train_data.to_csv("data/df_trainfinal.csv", index=False)

