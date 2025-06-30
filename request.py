import requests
import csv
import os

api_url = "https://api.etherscan.io/v2/api"

proxies = {
    'http': 'http://127.0.0.1:7897',
    'https': 'http://127.0.0.1:7897',
}


# # 请求参数字典
# params = {
#     'chainid': 1,
#     'module': 'account',
#     'action': 'txlistinternal',
#     'startblock': 21840285,
#     'endblock': 21841418,
#     'sort': 'asc',
#     'apikey': '8EXT2V7URCFRW3WVD5AWH7FSQS8J8GHZUB'
# }
def build_request_params(startblock, endblock):
    return {
        'chainid': 1,
        'module': 'account',
        'action': 'txlistinternal',
        'startblock': startblock,
        'endblock': endblock,
        'sort': 'asc',
        'apikey': '8EXT2V7URCFRW3WVD5AWH7FSQS8J8GHZUB'
    }


def fetch_internal_transactions(page, offset, startblock, endblock):
    params = build_request_params(startblock, endblock)
    params['page'] = page
    params['offset'] = offset
    try:
        response = requests.get(api_url, params=params, proxies=proxies, timeout=30, verify=True)
        response.raise_for_status()  # 检查HTTP响应是否成功

        data = response.json()

        if data['status'] == '1':
            print("成功获取数据")
            return data['result']
        else:
            print(f"API Error: {data['message']}")
            return []

    except requests.exceptions.Timeout:
        print("HTTP Request timed out. Writing data to CSV before exiting.")
        return 'timeout'

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return []


def write_to_csv(transactions, filename='dataset.csv'):
    # 定义CSV文件头（列名）
    fieldnames = ['blockNumber', 'timeStamp', 'hash', 'from', 'to', 'value', 'contractAddress', 'input', 'type', 'gas',
                  'gasUsed', 'traceId', 'isError', 'errCode']
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for tx in transactions:
            writer.writerow(tx)


def all_transactions():
    start_block = 21836418
    current_block = start_block
    end_block = 21846418
    num_blocks = 70
    while current_block < end_block:
        all_transactions = []
        page = 1
        offset = 1000
        has_more_data = True

        if current_block + num_blocks < end_block:
            next_block = current_block + num_blocks
        else:
            next_block = end_block

        while has_more_data:
            transactions = fetch_internal_transactions(page, offset, current_block, next_block)
            if transactions == 'timeout':
                break
            elif not transactions:
                has_more_data = False
            else:
                all_transactions.extend(transactions)
                print(f"Block {current_block} Page {page}: Retrieved {len(transactions)} transactions")
                page += 1

            # 如果达到最大偏移量或没有更多数据，则停止循环
            if len(transactions) < offset or not transactions:
                has_more_data = False

        if all_transactions:
            write_to_csv(all_transactions)
            print("所有数据已成功写入CSV文件")
            current_block += num_blocks


if __name__ == "__main__":
    all_transactions()