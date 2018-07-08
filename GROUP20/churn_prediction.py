from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from time import time
import pickle

def read_data(name):
    frames = pd.read_csv(name)
    print("Data successfully read!")
    return frames

def show_train_data_stats(data):
    n_train = data.shape[0]
    n_features = data.shape[1] - 1
    n_churn = (data['is_churn'].value_counts()) #less people churned
    churn_rate = (float(n_churn[1]) / (n_train)) * 100

    print("\nTrain Data:")
    print("             Total number of train data: " + str(n_train))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features+1])))
    print("             Churn rate: " + str(churn_rate) + "\n")

def show_test_data_stats(data):
    n_train = data.shape[0]
    n_features = data.shape[1] - 1


    print("\nTest Data:")
    print("             Total number of test data: " + str(n_train))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features+1])))

def show_transaction_data_stats(data):
    n_transactions = data.shape[0]
    n_features = data.shape[1] - 1

    print("\nTransaction Data:")
    print("             Total number of transactions: " + str(n_transactions))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features+1])))

def show_user_data_stats(data):
    n_logs = data.shape[0]
    n_features = data.shape[1] - 1

    print("\nUser Data:")
    print("             Total number of user logs: " + str(n_logs))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features+1])))

def show_member_data_stats(data):
    n_member = data.shape[0]
    n_features = data.shape[1] - 1

    print("\nMember Data:")
    print("             Total number of members: " + str(n_member))
    print("             Number of features: " + str(n_features))
    print("             " + (str(data.dtypes.index[1:n_features+1])))


def train_classifier(clf, x_data, y_data, param=None):

    print("Training classifier..")
    start = time()
    clf.fit(x_data, y_data)
    end = time()

    print("Trained model in " + str(end-start) + " seconds")

def save_clf(clf):
    with open( "model/"+clf.__class__.__name__ + '.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    print(clf.__class__.__name__ + " model saved!")

def load_clf(clf):
    with open( "model/" + clf + '.pkl', 'rb') as fid:
        loaded_clf = pickle.load(fid)
    print(loaded_clf.__class__.__name__ + " model loaded!")
    return loaded_clf

def predict_test(clf, features):

    print("Predicting on test data..")
    start = time()
    #y_pred = clf.predict(features)
    y_pred = clf.predict_proba(features)
    end = time()
    print("Predictions made in " + str(end-start) + " seconds")
    return y_pred

def predict_outcome(clf, features, target):

    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("Predictions made in " + str(end-start) + " seconds")
    f1score = f1_score(target, y_pred, pos_label=True)
    acc = sum(target == y_pred)/float(len(y_pred))
    print("F1 score and accuracy score for training set: " + str(f1score) + ", " + str(acc))

def prepare_data(data):

    print("Preparing data")
    #data = data.groupby('msno', as_index=False).mean()
    y_data = data['is_churn']
    x_data = data.drop(['is_churn'],1).drop(['msno'],1)
    x_data1, x_test, y_data1, y_test = train_test_split(x_data, y_data,
                                                    test_size =  100000,
                                                    random_state = 2,
                                                    stratify = y_data)
    print("Prepared data!")

    return x_data1, y_data1, x_test, y_test

def merge_data(data, dfmerge):
    dfnew = data.merge(dfmerge, left_on='msno', right_on='msno', how = 'inner')
    print(np.where(pd.isnull(dfnew)))
    return dfnew

def create_compiled_data(data, transaction, log, name):
    tmp = merge_data(data, transaction)
    tmp = merge_data(tmp, log)
    tmp.to_csv("data/" + name + ".csv")

if __name__ == "__main__":      #907471 unique test points, 1103895 unique user logs ids, 1197050 trans

    if (sys.argv[1]):  
        cmd = int(sys.argv[1])
    else:
        print("Please use a valid command")
    
    try:
        if (sys.argv[2]):
            if (sys.argv[2] == "xgb"):
                model = "XGBClassifier"
            elif (sys.argv[2] == "mlp"):
                model = "MLPClassifier"
            elif (sys.argv[2] == "lgb"):
                model = "Booster"
            elif (sys.argv[2] == "cat"):
                model = "CatBoostClassifier"
            else:
                print("Please use a valid model")
    except IndexError:
        pass
    if cmd == 0:            #create compiled data

        train_data = read_data("data/train_v2.csv")
        test_data = read_data("data/sample_submission_v2.csv")   #already normalized
        transaction_data = read_data("data/transactions_v2.csv") #already normalized
        log_data = read_data("data/user_logs_v2.csv")
        #member_data = read_data("data/members_v3.csv")  #not using members data
        show_train_data_stats(train_data)
        show_test_data_stats(test_data)
        show_transaction_data_stats(transaction_data)
        show_user_data_stats(log_data)
        #show_member_data_stats(member_data)
        create_compiled_data(train_data, transaction_data, log_data, "train_compiled")
        create_compiled_data(test_data, transaction_data, log_data, "test_compiled")


    elif cmd == 1:           #train mlp

        train_data = read_data('data/df_trainfinal.csv')
        #split data into train and val sets
        x_train, y_train, x_test, y_test = prepare_data(train_data)
        print(x_train.shape)
        #make classifier
        clfa = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (15, 8, 4), verbose=True)
        #training
        train_classifier(clfa, x_train, y_train)
        #save classifer
        save_clf(clfa)
        #test/predict
        predict_outcome(clfa, x_test, y_test)

    elif cmd == 2: 
        clf = load_clf(model)
        print(clf.classes_)
        test_data = read_data("data/df_testfinal.csv")
        print(test_data.shape)
        resultdf = (test_data['msno']).to_frame()
        x_data = test_data.drop(['is_churn'],1).drop(['msno'],1)
        results = predict_test(clf, x_data)
        new = []
        for i in range(len(results)):
            new.append(results[i][1])
        print(len(new))
        resultdf['is_churn'] = pd.DataFrame({"is_churn":new})
        resultdf.to_csv("results/" + model + ".csv", index=False)
    elif cmd == 3: #xgboostd
        params = {
        'eta': 0.02, 
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 100,
        'silent': True
    }
        train_data = read_data('data/df_trainfinal.csv')
        print(train_data.columns)
        #split data into train and val sets
        x_train, y_train, x_test, y_test = prepare_data(train_data)
        #make classifier
        clfa = XGBClassifier()
        #training
        train_classifier(clfa, x_train, y_train, params)
        #save classifer
        save_clf(clfa)
        #test/predict
        predict_outcome(clfa, x_test, y_test)
        
    elif cmd == 4: #catboost
        train_data = read_data('data/df_trainfinal.csv')
        #split data into train and val sets
        x_train, y_train, x_test, y_test = prepare_data(train_data)
        #make classifier
        clfa = CatBoostClassifier()
        #training
        train_classifier(clfa, x_train, y_train)
        #save classifer
        save_clf(clfa)
        #test/predict
        predict_outcome(clfa, x_test, y_test)
    
    elif cmd == 5: #lightgbm
        train_data = read_data('data/df_trainfinal.csv')
        #split data into train and val sets
        x_train, y_train, x_test, y_test = prepare_data(train_data)
        #make classifier
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test)
        #trainin
        
        params = {
    'learning_rate': 0.05,
    'application': 'binary',
    'max_depth': 5,
    'num_leaves': 256,
    'verbosity': -1,
    'metric': 'binary_logloss'
}
        watchlist = [(lgb_eval, 'eval'), (lgb_train, 'train')]
        model = lgb.train(params, train_set=lgb_train, num_boost_round=240, verbose_eval=10) 
        #save classifer
        save_clf(model)
        #test/predict
        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
        test_data = read_data("data/df_testfinal.csv")
        test_data_dropped = test_data.drop(['is_churn'],1).drop(['msno'],1)
        lgb_pred = model.predict(test_data_dropped )
        test_data['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15)
        test_data[['msno','is_churn']].to_csv('results/LGB.csv', index=False)
        
    elif cmd == 6: #ensemble, merge csv and average
        #mlp = read_data('results/MLPClassifier.csv')
        xgb = read_data('results/XGBClassifier.csv') 
        cat =  read_data('results/CatBoostClassifier.csv') 
        lgb =  read_data('results/LGB.csv') 
        merge = cat 
        merge['tmpxgb']= xgb['is_churn']
        merge['tmplgb']= lgb['is_churn']
        merge['avg'] = merge[['is_churn','tmpxgb','tmplgb']].mean(axis=1)
        merge['is_churn'] = merge['avg']
        merge = merge.drop(['tmpxgb','tmplgb','avg'], axis =1)
        merge.to_csv('results/ensemble.csv',index=False)
    else:
        print("No valid commands")
