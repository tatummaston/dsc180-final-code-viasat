import pandas as pd 
import numpy as np


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

def etl_latency():
    
    d20 = pd.read_csv("test/testdata/latency changes/20211110T224016_20-10000-iperf.csv")
    d30 = pd.read_csv("test/testdata/latency changes/20211110T224017_30-10000-iperf.csv")
    d10 = pd.read_csv("test/testdata/latency changes/20211110T224018_10-10000-iperf.csv")
    
    data_list = [d20, d30, d10]
    latency_list = [20, 30, 40] 
    
    for i in range(len(latency_list)):
        d = [int(latency_list[i]) for _ in range(data_list[i].shape[0])]
        data_list[i]['latency'] = pd.Series(d)
        data_list[i]['total_bytes'] = data_list[i]['1->2Bytes'] + data_list[i]['2->1Bytes']
        data_list[i]['total_pkts'] = data_list[i]['1->2Pkts'] + data_list[i]['2->1Pkts']
        
    
    return agg_10(d20), agg_10(d30), agg_10(d10)



def etl_packet_loss():
    
    run_200_5000 = pd.read_csv("test/testdata/packet loss changes/20211015T175337_200-5000-iperf.csv")
    run_200_500 = pd.read_csv("test/testdata/packet loss changes/20211023T012228_200-500-iperf.csv")
    run_200_750 = pd.read_csv("test/testdata/packet loss changes/20211023T012231_200-750-iperf.csv")
    
    ratio_list = [0.04, 0.4, 0.266]
    data_list = [run_200_5000, run_200_500, run_200_750]
    
    for i in range(len(ratio_list)):
        d = [int(ratio_list[i]) for _ in range(data_list[i].shape[0])]
        data_list[i]['ratio'] = pd.Series(d)
        data_list[i]['total_bytes'] = data_list[i]['1->2Bytes'] + data_list[i]['2->1Bytes']
        data_list[i]['total_pkts'] = data_list[i]['1->2Pkts'] + data_list[i]['2->1Pkts']
        
        
    