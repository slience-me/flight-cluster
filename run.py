# Author: slience_me
# Date: 2024/9/29 19:50

import pandas as pd
import flight_clustering

if __name__ == "__main__":
    # 加载数据集
    print("Loading dataset...")
    # 仅仅修改 valid_data.csv -> test_data.csv
    df = pd.read_csv('data/valid_data.csv')
    print("Loading dataset finished. ")
    flight_clustering.cluster(df)
