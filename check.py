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
 
df_members = read_data("data/members_v3.csv")
df_trxn = read_data("data/transactions_v2.csv")
df_comb = read_data("data/df_comb.csv")
#df_train = read_data("data/df_train.csv")
print(df_members.head())
print(df_trxn.head())
print(df_comb.head())
#print(df_train.head())
print(df_members.shape)
print(df_trxn.shape)
print(df_comb.shape)


#print(df_train.shape)



















