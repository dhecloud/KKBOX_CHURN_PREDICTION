import pandas as pd
def read_data(name):
    frames = pd.read_csv(name)
    print("Data successfully read!")
    return frames

#use train_v2 or sample_submission_v2 depending on which file u wanna compile
data = read_data("data/sample_submission_v2.csv")    


train_data = read_data("data/df_comb.csv")

#remove some unneeded columns
train_data.drop(['transaction_date'],1).drop(['membership_expire_date'],1).drop(['gender'],1).drop(['bd'],1).drop(['registration_init_time'],1)

train_data = pd.merge(train_data, data, on='msno')

#normalization
train_data['amt_per_day'] = train_data['amt_per_day'].replace(to_replace=float('inf'),value = 0)
train_data['payment_plan_days'] = (train_data['payment_plan_days'] - train_data['payment_plan_days'].min())/(train_data['payment_plan_days'].max() - train_data['payment_plan_days'].min())
train_data['plan_list_price'] = (train_data['plan_list_price'] - train_data['plan_list_price'].min())/(train_data['plan_list_price'].max() - train_data['plan_list_price'].min())
train_data['actual_amount_paid'] = (train_data['actual_amount_paid'] - train_data['actual_amount_paid'].min())/(train_data['actual_amount_paid'].max() - train_data['actual_amount_paid'].min())

#aggregation
train_data = train_data.groupby('msno', as_index=False).mean()

train_data.to_csv("data/testdf_comb2.csv", index=False)