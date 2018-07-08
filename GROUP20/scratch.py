import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from subprocess import check_output
from multiprocessing import Pool, cpu_count


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
def reformat(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def reformat2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df
    
    
last_user_logs = []
user_log2 = pd.read_csv('data/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
i = 0
for df in user_log2:
    if i>35:
        if len(df)>0:
            print(df.shape)
            pool = Pool(cpu_count())
            df = pool.map(reformat, np.array_split(df, cpu_count()))   
            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
            df = reformat2(df)
            p.close(); p.join()
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1

user_log = read_data("data/user_logs_v2.csv")
last_user_logs.append(reformat(user_log))
last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = reformat2(last_user_logs)

#user_log = user_log.drop(['date', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100'], axis=1)

#memory reduction
user_log = change_datatype(user_log)   
user_log = change_datatype_float(user_log)
#mem = user_log.memory_usage(index=True).sum()    
#print(mem/ 1024**2,"MB")
print(user_log.shape)


comb = read_data("data/df_comb.csv")
#remove some unneeded columns and save mem
#comb = change_datatype(comb)   
#comb = change_datatype_float(comb)
data = pd.merge(comb, user_log, on='msno', how = 'outer')
del comb
del user_log

test_data = read_data("data/sample_submission_v2.csv")

train = pd.read_csv('data/train.csv')
train_data = read_data("data/train_v2.csv")
train_data = pd.concat((train, train_data), axis=0, ignore_index=True).reset_index(drop=True)
train_data = pd.merge(train_data, data, on='msno', how = 'left')
train_data = train_data.replace(to_replace=float('inf'),value = 0)
train_data = train_data.fillna(value=-1)
#normalization
train_data = normalize(train_data, ['amt_per_day', 'payment_plan_days','plan_list_price', 'actual_amount_paid','plan_list_price', 'payment_plan_days', 'actual_amount_paid', 'amt_per_day', 'membership_duration', 'discount'])

#aggregation
train_data = train_data.groupby('msno', as_index=False).mean()


test_data = pd.merge(test_data, data, on='msno', how = 'left')
test_data = test_data.replace(to_replace=float('inf'),value = 0)
test_data = test_data.fillna(value=-1)
#normalization
test_data = normalize(test_data, ['amt_per_day', 'payment_plan_days','plan_list_price', 'actual_amount_paid','plan_list_price', 'payment_plan_days', 'actual_amount_paid', 'amt_per_day', 'membership_duration', 'discount'])

#aggregation
test_data = test_data.groupby('msno', as_index=False).mean()
print(test_data.shape)

test_data.to_csv("data/df_testfinal.csv", index=False)
train_data.to_csv("data/df_trainfinal.csv", index=False)

        