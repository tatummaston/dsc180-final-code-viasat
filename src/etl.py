import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def pckt_count(a):
    a = a.split(';')
    a = [int(i) for i in a[:-1]]
    return len(a)

def agg_10sec(df):
    new_df = pd.DataFrame()
    min_time = df["Time"][0]
    latency = df["latency"][0]
    while min_time < df["Time"][len(df)-1]:
        temp_df = df[(df["Time"] >= min_time) & (df["Time"] < min_time+10)]
        row = temp_df[["1->2Bytes", "2->1Bytes", "1->2Pkts", "2->1Pkts", "total_pkts", "total_bytes"]].sum().to_frame().T
        row["packet_sizes_var"] = temp_df["total_pkts"].var()
        row["avg_time_delta"] = temp_df['packet_times'].apply(avg_time_delt).mean()
        row["1->2pkts_mean"] = temp_df['1->2Pkts'].mean()
        row["Time"] = min_time
        row["latency"] = latency
        new_df = new_df.append(row)
        min_time += 10
    return new_df.reset_index(drop=True)

def avg_time_delt(a):
    return pd.Series(a.split(';')[:-1]).astype(int).diff().mean()

# def etl_latency():
    
#     d20 = pd.read_csv("test/testdata/latency changes/20211110T224016_20-10000-iperf.csv")
#     d30 = pd.read_csv("test/testdata/latency changes/20211110T224017_30-10000-iperf.csv")
#     d10 = pd.read_csv("test/testdata/latency changes/20211110T224018_10-10000-iperf.csv")
    
#     data_list = [d20, d30, d10]
#     latency_list = [20, 30, 40] 
    
#     for i in range(len(latency_list)):
#         d = [int(latency_list[i]) for _ in range(data_list[i].shape[0])]
#         data_list[i]['latency'] = pd.Series(d)
#         data_list[i]['total_bytes'] = data_list[i]['1->2Bytes'] + data_list[i]['2->1Bytes']
#         data_list[i]['total_pkts'] = data_list[i]['1->2Pkts'] + data_list[i]['2->1Pkts']
        
    
#     return agg_10(d20), agg_10(d30), agg_10(d10)



# def etl_packet_loss():
    
#     run_200_5000 = pd.read_csv("test/testdata/packet loss changes/20211015T175337_200-5000-iperf.csv")
#     run_200_500 = pd.read_csv("test/testdata/packet loss changes/20211023T012228_200-500-iperf.csv")
#     run_200_750 = pd.read_csv("test/testdata/packet loss changes/20211023T012231_200-750-iperf.csv")
    
#     ratio_list = [0.04, 0.4, 0.266]
#     data_list = [run_200_5000, run_200_500, run_200_750]
    
#     for i in range(len(ratio_list)):
#         d = [int(ratio_list[i]) for _ in range(data_list[i].shape[0])]
#         data_list[i]['ratio'] = pd.Series(d)
#         data_list[i]['total_bytes'] = data_list[i]['1->2Bytes'] + data_list[i]['2->1Bytes']
#         data_list[i]['total_pkts'] = data_list[i]['1->2Pkts'] + data_list[i]['2->1Pkts']
        
        
def latency_data_prep(file18, file19, file20)
    d20 = pd.read_csv(file18)
    d30 = pd.read_csv(file19)
    d10 = pd.read_csv(file20)
    
    data_list = [d20, d30, d10]
    latency_list = [20, 30, 10]
    
    for i in range(len(latency_list)):
        d = [int(latency_list[i]) for _ in range(data_list[i].shape[0])]
        data_list[i]['latency'] = pd.Series(d)
        data_list[i]['total_bytes'] = data_list[i]['1->2Bytes'] + data_list[i]['2->1Bytes']
        data_list[i]['total_pkts'] = data_list[i]['1->2Pkts'] + data_list[i]['2->1Pkts']
        data_list[i]['pck_ct'] = data_list[i]['packet_times'].apply(pckt_count)
        
    
    agg_20 = agg_10sec(d20)
    agg_30 = agg_10sec(d30)
    agg_10 = agg_10sec(d10)
    
    full_df = pd.concat([agg_10, agg_50, agg_100, agg_150, agg_200])
    
    return full_df



def loss_data_prep(file1, file2, file3):
    
    run_200_5000 = pd.read_csv(file1)
    run_200_500 = pd.read_csv(file2)
    run_200_750 = pd.read_csv(file3)
    
    ratio_list = [0.04, 0.4, 0.266]
    data_list = [run_200_5000, run_200_500, run_200_750]

    for i in range(len(ratio_list)):
        d = [int(ratio_list[i]) for _ in range(data_list[i].shape[0])]
        data_list[i]['ratio'] = pd.Series(d)
        data_list[i]['total_bytes'] = data_list[i]['1->2Bytes'] + data_list[i]['2->1Bytes']
        data_list[i]['total_pkts'] = data_list[i]['1->2Pkts'] + data_list[i]['2->1Pkts']
        data_list[i]['pck_ct'] = data_list[i]['packet_times'].apply(pckt_count)
        
    agg_200_5000 = agg_10sec(run_200_5000)
    agg_200_500 = agg_10sec(run_200_500)
    agg_200_750 = agg_10sec(run_200_750)
    
    full_df = pd.concat([agg_200_5000, agg_200_500, agg_200_750])
    
    return loss_data_prep


def packet_linear_reg(df):
    features = ['packet_size_total', 'pck_ct', 'packet_sizes_var','avg_time_delta', 'Time']
    df_X = full_df[features]
    df_y = full_df['ratio']

  # Use only one feature
  #diabetes_X = diabetes_X[:, np.newaxis, 2]

  # Split the data into training/testing sets
    X_train, X_rem, y_train, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

  # new data 

    full_df = pd.concat([agg_200_400, agg_200_800, agg_200_1400, agg_200_1800])

  # Load the diabetes dataset
    features = ['packet_size_total', 'pck_ct', 'packet_sizes_var','avg_time_delta', 'Time']
    df_X = full_df[features]
    df_y = full_df['ratio']

    filler, X_rem, filler, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  # Create linear regression object
    regr = linear_model.LinearRegression()

  # Train the model using the training sets
    regr.fit(X_train, y_train)

  # Make predictions using the testing set
    y_pred = regr.predict(X_test)

  # The coefficients
  #print("Coefficients: \n", regr.coef_)
  # The mean squared error
  #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
  # The coefficient of determination: 1 is perfect prediction
  #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    
    f = open("outputs/model_output.txt", "a")
    f.write(f"linear regression packet loss r^2 score: {r2_score(y_test, y_pred)})
    f.close()
    
  
    return r2_score(y_test, y_pred) #r2_score(y_test.reset_index(drop=True), y_pred), y_test.reset_index(drop=True), y_pred



def latency_linear_reg(df):
  # Load the dataset 'packet_sizes_var', "Time"
    features = ['packet_size_total', 'pck_ct', "total_bytes", 'packet_sizes_var']
    df_X = df[features]
    df_y = df['latency']

  # Split the data into training/testing sets
    X_train, X_rem, y_train, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  # Create linear regression object
    regr = linear_model.LinearRegression()

  # Train the model using the training sets
    regr.fit(X_train, y_train)

  # Make predictions using the testing set
    y_pred = regr.predict(X_test)

  # The coefficients
  #print("Coefficients: \n", regr.coef_)

  # The mean squared error
  #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

  # The coefficient of determination: 1 is perfect prediction
  #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    
    f = open("outputs/model_output.txt", "a")
    f.write(f"linear regression latency r^2 score: {r2_score(y_test.reset_index(drop=True), y_pred)})
    f.close()
    
    
    return r2_score(y_test.reset_index(drop=True), y_pred)#, y_test.reset_index(drop=True), y_pred



def decision_tree(df):
  # Load the dataset 'packet_sizes_var', "Time"
      features = ['packet_size_total', 'pck_ct', "total_bytes", 'packet_sizes_var']
    df_X = df[features]
    df_y = df['latency']

  # Split the data into training/testing sets
    X_train, X_rem, y_train, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  # Create linear regression object
    clf = tree.DecisionTreeClassifier()

  # Train the model using the training sets
    clf = clf.fit(X_train, y_train)

  # Make predictions using the testing set
    y_pred = clf.predict(X_test)

  # The coefficients
  #print("Coefficients: \n", regr.coef_)

  # The mean squared error
  #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

  # The coefficient of determination: 1 is perfect prediction
  #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
            
    f = open("outputs/model_output.txt", "a")
    f.write(f"decision tree latency r^2 score: {clf.score(X_test, y_test.reset_index(drop=True))})
    f.close()

    return clf.score(X_test, y_test.reset_index(drop=True))#, y_test.reset_index(drop=True), y_pred
  
def svm(df):
  # Load the dataset 'packet_sizes_var', "Time"
    features = ['packet_size_total', 'pck_ct', "total_bytes", 'packet_sizes_var', 'Time']
    df_X = df[features]
    df_y = df['latency']

  # Split the data into training/testing sets
    X_train, X_rem, y_train, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  # Create linear regression object
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

  # Train the model using the training sets
    clf = clf.fit(X_train, y_train)

  # Make predictions using the testing set
    y_pred = clf.predict(X_test)

  # The coefficients
  #print("Coefficients: \n", regr.coef_)

  # The mean squared error
  #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

  # The coefficient of determination: 1 is perfect prediction
  #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
                                                          
    f = open("outputs/model_output.txt", "a")
    f.write(f"svm latency r^2 score: {clf.score(X_test, y_test.reset_index(drop=True))})
    f.close()                                                     
                                                     

    return clf.score(X_test, y_test.reset_index(drop=True))#, y_test.reset_index(drop=True), y_pred