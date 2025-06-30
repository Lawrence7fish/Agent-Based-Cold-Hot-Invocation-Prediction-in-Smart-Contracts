import requests
import csv
import os
import pandas as pd
import numpy as np
from dl_process import window
api_url = "https://api.etherscan.io/v2/api"

proxies = {
    'http': 'http://127.0.0.1:7897',
    'https': 'http://127.0.0.1:7897',
}


def build_request_params(address, startblock, endblock):
    return {
        'chainid': 1,
        'module': 'account',
        'action': 'txlistinternal',
        'startblock': startblock,
        'endblock': endblock,
        'sort': 'asc',
        'address': address,
        'apikey': '8EXT2V7URCFRW3WVD5AWH7FSQS8J8GHZUB'
    }

def raw_data(address, current, page = 1, offset = 0):
    params = build_request_params(address, current - window + 1, current)
    params['page'] = page
    params['offset'] = offset
    try:
        response = requests.get(api_url, params=params, proxies=proxies, timeout=30, verify=True)
        response.raise_for_status()  # 检查HTTP响应是否成功

        data = response.json()

        if data['status'] == '1':
            d_data = pd.DataFrame(data['result'])
            d_data = d_data[(d_data['to'] == address.lower()) & (d_data['isError'] == '0')].copy()
            d_data['type'], _ = pd.factorize(d_data['type'])
            mask = d_data['type'] == 1
            d_data['to'] = np.where(mask, d_data['contractAddress'], d_data['to'])
            d_data = d_data.drop(columns=['type', 'contractAddress', 'to'])
            d_data = d_data.groupby(['blockNumber', 'timeStamp']).agg(
                gas_mean=('gas', 'mean'),
                gasUsed_mean=('gasUsed', 'mean'),
                counts=('gas', 'size')
            ).reset_index()

            d_data['timeStamp'] = pd.to_datetime(d_data['timeStamp'], unit='s')
            d_data['hour'] = d_data['timeStamp'].dt.hour
            d_data['minute'] = d_data['timeStamp'].dt.minute
            d_data['second'] = d_data['timeStamp'].dt.second
            d_data['blockNumber'] = current - d_data['blockNumber'].values.astype(int)
            d_data = d_data.drop(columns=['timeStamp'])
            return d_data
        else:
            print(f"API Error: {data['message']}")
            return []

    except requests.exceptions.Timeout:
        print("HTTP Request timed out. ")
        return 'timeout'

    except requests.exceptions.RequestException:
        print(f"HTTP Request failed")
        return "request failed"
def get_internal_cluster_pred(address, current, page = 1, offset = 0):
    params = build_request_params(address, current - window + 1, current)
    params['page'] = page
    params['offset'] = offset
    try:
        response = requests.get(api_url, params=params, proxies=proxies, timeout=30, verify=True)
        response.raise_for_status()  # 检查HTTP响应是否成功

        data = response.json()

        if data['status'] == '1':
            c_data = pd.DataFrame(data['result'])
            c_data = c_data[(c_data['to'] == address.lower()) & (c_data['isError'] == '0')].copy()
            c_data = c_data[['blockNumber', 'to', 'contractAddress', 'type']]
            c_data['type'], _ = pd.factorize(c_data['type'])
            mask = c_data['type'] == 1
            c_data['to'] = np.where(mask, c_data['contractAddress'], c_data['to'])
            c_data = c_data.drop(columns=['type', 'contractAddress'])
            # 按照blockNumber进行分组，即每个区块被调用次数，以区块为最小粒度
            c_data = c_data.groupby(['blockNumber']).size().reset_index(name='counts')

            return c_data
        else:
            print(f"API Error: {data['message']}")
            return []
    except requests.exceptions.Timeout:
        print("HTTP Request timed out.")
        return 'timeout'

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed")
        return "request failed"


def get_internal_lstm_pred(address, current, page = 1, offset = 0):
    params = build_request_params(address, current - window + 1, current)
    params['page'] = page
    params['offset'] = offset
    try:
        response = requests.get(api_url, params=params, proxies=proxies, timeout=30, verify=True)
        response.raise_for_status()  # 检查HTTP响应是否成功

        data = response.json()

        if data['status'] == '1':
            d_data = pd.DataFrame(data['result'])
            d_data = d_data[(d_data['to'] == address.lower()) & (d_data['isError'] == '0')].copy()
            d_data['type'], _ = pd.factorize(d_data['type'])
            mask = d_data['type'] == 1
            d_data['to'] = np.where(mask, d_data['contractAddress'], d_data['to'])
            d_data = d_data.drop(columns=['type', 'contractAddress', 'to'])
            d_data = d_data.groupby(['blockNumber', 'timeStamp']).agg(
                gas_mean=('gas', 'mean'),
                gasUsed_mean=('gasUsed', 'mean'),
                counts=('gas', 'size')
            ).reset_index()

            d_data['timeStamp'] = pd.to_datetime(d_data['timeStamp'], unit='s')
            d_data['hour'] = d_data['timeStamp'].dt.hour
            d_data['minute'] = d_data['timeStamp'].dt.minute
            d_data['second'] = d_data['timeStamp'].dt.second
            d_data = d_data.drop(columns=['blockNumber', 'timeStamp'])

            return d_data
        else:
            print(f"API Error: {data['message']}")
            return []

    except requests.exceptions.Timeout:
            print("HTTP Request timed out. ")
            return 'timeout'

    except requests.exceptions.RequestException:
            print(f"HTTP Request failed")
            return "request failed"

# 请预测账号0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5，在区块22122821后5个区块的调用频率
if __name__ == '__main__':
     result_d = raw_data("0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5", 22122821)
     print(np.array(result_d).tolist())