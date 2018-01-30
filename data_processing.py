import pandas as pd
import numpy as np

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






