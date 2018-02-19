import pandas as pd
import numpy as np
import sys
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
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


def train_classifier(clf, x_data, y_data):

    print("Training classifier..")
    start = time()
    clf.fit(x_data, y_data)
    end = time()

    print("Trained model in " + str(end-start) + " seconds")

def save_clf(clf):
    with open( clf.__class__.__name__ + '.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    print(clf.__class__.__name__ + " model saved!")

def load_clf(clf):
    with open( clf + '.pkl', 'rb') as fid:
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

    cmd = int(sys.argv[1])
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


    elif cmd == 1:           #train

        train_data = read_data('data/df_trainfinal.csv')
        #split data into train and val sets
        x_train, y_train, x_test, y_test = prepare_data(train_data)
        print(x_train.shape)
        #make classifier
        clfa = MLPClassifier(solver = 'adam', alpha = 0.0001, hidden_layer_sizes= (20, 15, 10), verbose=True)
        #training
        train_classifier(clfa, x_train, y_train)
        #save classifer
        save_clf(clfa)
        #test/predict
        predict_outcome(clfa, x_test, y_test)

    elif cmd == 2:
        clf = load_clf("MLPClassifier")
        print(clf.classes_)
        #train_data = read_data("data/train_compiled.csv")
        test_data = read_data("data/df_testfinal.csv")
        test_data1 = read_data("data/sample_submission_v2.csv")
        print(test_data1.shape)
        #show_train_data_stats(train_data)
        #show_test_data_stats(test_data)
        resultdf = (test_data['msno']).to_frame()
        x_data = test_data.drop(['is_churn'],1).drop(['msno'],1)
        results = predict_test(clf, x_data)
        new = []
        for i in range(len(results)):
            new.append(results[i][1])
        print(len(new))
        resultdf['is_churn'] = pd.DataFrame({"is_churn":new})
        resultdf.to_csv("data/results.csv", index=False)
    else:
        print("No valid commands")
