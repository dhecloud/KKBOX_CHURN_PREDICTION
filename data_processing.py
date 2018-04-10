import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from subprocess import check_output

#methods

def read_data(name):
    df = pd.read_csv(name)
    return df

def show_train_data_stats(data):
    n_train = data.shape[0]
    n_features = data.shape[1] - 1
    n_churn = (data['is_churn'].value_counts())[1] #less people churned
    churn_rate = (float(n_churn) / (n_train)) * 100

    print("\nTrain Data:")
    print("             Total number of train data: " + str(n_train))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features])))
    print("             Churn rate: " + str(churn_rate) + "\n")


def show_transaction_data_stats(data):
    n_transactions = data.shape[0]
    n_features = data.shape[1] - 1

    print("\nTransaction Data:")
    print("             Total number of transactions: " + str(n_transactions))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features])))

def show_user_data_stats(data):
    n_logs = data.shape[0]
    n_features = data.shape[1] - 1

    print("\nUser Data:")
    print("             Total number of user logs: " + str(n_logs))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features])))

def show_member_data_stats(data):
    n_member = data.shape[0]
    n_features = data.shape[1] - 1

    print("\nMember Data:")
    print("             Total number of members: " + str(n_member))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features])))


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

'''DATA PROCESSING PART I '''

#   -- user logs data processing --
'''
user_logs_data = read_data("kaggle/data/user_logs_v2.csv")

user_logs_data.insert(9, 'total_songs_played', None)
total_songs_played = user_logs_data["num_25"] + user_logs_data["num_50"] + user_logs_data["num_75"] + user_logs_data["num_985"] + user_logs_data["num_100"]
user_logs_data["total_songs_played"] = total_songs_played

user_logs_data.insert(10, 'ave_percent_played', None)
ave_percent_played = (user_logs_data["num_25"]*25/2 + user_logs_data["num_50"]*(50+25)/2 + user_logs_data["num_75"]*(75+50)/2 + user_logs_data["num_985"]*(75+98.5)/2 + user_logs_data["num_100"]*(98.5+100)/2)/total_songs_played
user_logs_data["ave_percent_played"] = ave_percent_played

normalized_total_songs_played = (user_logs_data["total_songs_played"] - user_logs_data["total_songs_played"].min())/(user_logs_data["total_songs_played"].max() - user_logs_data["total_songs_played"].min())
normalized_ave_percent_played = (user_logs_data["ave_percent_played"] - user_logs_data["ave_percent_played"].min())/(user_logs_data["ave_percent_played"].max() - user_logs_data["ave_percent_played"].min())

user_logs_data["total_songs_played"] = normalized_total_songs_played
user_logs_data["ave_percent_played"] = normalized_ave_percent_played


normalized_num_unq = (user_logs_data["num_unq"] - user_logs_data["num_unq"].min())/(user_logs_data["num_unq"].max() - user_logs_data["num_unq"].min())
normalized_total_secs = (user_logs_data["total_secs"] - user_logs_data["total_secs"].min())/(user_logs_data["total_secs"].max() - user_logs_data["total_secs"].min())
user_logs_data["num_unq"] = normalized_num_unq
user_logs_data["total_secs"] = normalized_total_secs

user_logs_data.to_csv("kaggle/data/user_logs_v2.csv", index=False)
print(user_logs_data)
'''

#   -- transactions data processing --
'''
#transaction_data = read_data("kaggle/data/transactions_v2.csv")
#(x - min(x)) / (max(x) - min(x))   formula for normalisation

#normalized_payment_plan_days=(transaction_data["payment_plan_days"] - transaction_data["payment_plan_days"].min())/(transaction_data["payment_plan_days"].max() - transaction_data["payment_plan_days"].min())
#normalized_plan_list_price=(transaction_data["plan_list_price"] - transaction_data["plan_list_price"].min())/(transaction_data["plan_list_price"].max() - transaction_data["plan_list_price"].min())
#normalized_actual_amount_paid=(transaction_data["actual_amount_paid"] - transaction_data["actual_amount_paid"].min())/(transaction_data["actual_amount_paid"].max() - transaction_data["actual_amount_paid"].min())

#transaction_data["payment_plan_days"] = normalized_payment_plan_days
#transaction_data["plan_list_price"] = normalized_plan_list_price
#transaction_data["actual_amount_paid"] = normalized_actual_amount_paid

#transaction_data.to_csv("kaggle/data/transactions_v2.csv", index=False)
#print(transaction_data)
'''
#   -- train data processing --
'''
#train_data = read_data("kaggle/data/train.csv")

#print(train_data)
# 992931 entries of data
'''

#   -- check duplicate row with specified col --
'''
train_data = read_data("kaggle/data/train_v2.csv")
train_data[train_data.duplicated(['msno'], keep=False)]

test_data = read_data("kaggle/data/sample_submission_v2.csv")
test_data[test_data.duplicated(['msno'], keep=False)]
'''


#   -- data merging wrt 'msno' in train.csv -- incomplete
'''
train_data = read_data("kaggle/data/train_v2.csv")
transaction_data = read_data("kaggle/data/transactions_v2.csv")
user_logs_data = read_data("kaggle/data/user_logs_v2.csv")
member_data = read_data("kaggle/data/members_v3.csv")

merge_data = train_data.merge(transaction_data, on='msno', how='outer')
merge_data = pd.merge(train_data, transaction_data, on='msno', how='inner')
show_train_data_stats(merge_data)
show_train_data_stats(train_data)

merge_data_final = pd.merge(merge_data_2, member_data, on='msno', how='inner')
merge_data_final.to_csv("kaggle/data/merge_final.csv", index=False)

#merge_final = read_data("kaggle/data/merge_final.csv")
#print(merge_final)
'''

#   -- check unique col values -- 
'''
transaction_data = read_data("kaggle/data/transactions_v2.csv")
a = transaction_data.msno.unique()
print(len(a))
'''





''' ------------------------------ DATA PROCESSING PART II (not a continuation of PART I)------------------------------ '''

''' -- transactions data -- '''
df_transactions = read_data("data/transactions_v2.csv")

date_cols = ['transaction_date', 'membership_expire_date']
for col in date_cols:
    df_transactions[col] = pd.to_datetime(df_transactions[col], format='%Y%m%d')
    
#print(df_transactions.head())

#new feature membership_duration in days
df_transactions['membership_duration'] = df_transactions.membership_expire_date - df_transactions.transaction_date
df_transactions['membership_duration'] = df_transactions['membership_duration'] / np.timedelta64(1, 'D')
df_transactions['membership_duration'] = df_transactions['membership_duration'].astype(int)


df_transactions = df_transactions.drop(['transaction_date', 'membership_expire_date'],axis=1)

df_transactions = df_transactions.groupby('msno', as_index=False).mean()
#print(df_transactions.shape)


#memory reduction
change_datatype(df_transactions)   
change_datatype_float(df_transactions)

#check memory usage
'''
mem = df_transactions.memory_usage(index=True).sum()    
print(mem/ 1024**2,"MB")
'''

#check data types of cols
'''
print(df_transactions.dtypes, '\n')
print(df_members.dtypes)
'''

#check number of cols
'''
length = len(df_transactions.columns)
print(length)
'''

#new feature 'discount' to see how much discount was offered to the customer
df_transactions['discount'] = df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']
#print(df_transactions['discount'].unique())

#new featuer 'is_discount' to check whether the customer has availed any discount or not
df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
#print(df_transactions['is_discount'].head())
#print(df_transactions['is_discount'].unique())

#new feature amount_per_day
df_transactions['amt_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']
#print(df_transactions['amt_per_day'].head())

#print(df_transactions.head())
#print(len(df_transactions.columns), "columns")

change_datatype(df_transactions)
change_datatype_float(df_transactions)
#mem = df_transactions.memory_usage(index=True).sum()    
#print(mem/ 1024**2,"MB")
#print(df_transactions.head())



''' -- members data -- '''
df_members = read_data("data/members_v3.csv")  
df_members = df_members.groupby('msno', as_index=False).mean()

#memory reduction
change_datatype(df_members)
change_datatype_float(df_members)
#print(df_members.head())
#print(len(df_members.columns), "columns")

#check memory usage
'''
mem = df_members.memory_usage(index=True).sum()    
print(mem/ 1024**2,"MB")
mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2,"MB")
'''

#check data types of cols
'''
print(df_members.dtypes, '\n')
print(df_members.dtypes)
'''

#change date format

date_col = ['registration_init_time']

for col in date_col:
    df_members[col] = pd.to_datetime(df_members[col], format='%Y%m%d')

#memory reduction again
change_datatype(df_members)
change_datatype_float(df_members)
#mem = df_members.memory_usage(index=True).sum()    
#print(mem/ 1024**2,"MB")
#print(df_members.head())



''' --  dataframes merging -- '''
df_comb = pd.merge(df_transactions, df_members, on='msno', how='outer')


#deleting the dataframes to save memory
del df_transactions
del df_members
'''
print(df_comb.head())
mem = df_comb.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
'''

#new feature to see whether mebers have auto renewed and not cancelled at the same time, auto_renew = 1 and is_cancel = 0
df_comb['autorenew_&_not_cancel'] = ((df_comb.is_auto_renew == 1) == (df_comb.is_cancel == 0)).astype(np.int8)
#print(df_comb['autorenew_&_not_cancel'].unique())

#new feature to predict possible churning if auto_renew = 0 and is_cancel = 1
df_comb['notAutorenew_&_cancel'] = ((df_comb.is_auto_renew == 0) == (df_comb.is_cancel == 1)).astype(np.int8)
#print(df_comb['notAutorenew_&_cancel'].unique())
df_comb = df_comb.drop(['payment_method_id', 'is_cancel', 'bd', 'registration_init_time', 'registered_via', 'city'], axis=1)
df_comb.to_csv("data/df_comb.csv", index=False)


