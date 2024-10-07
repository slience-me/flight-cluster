# Author: slience_me
# Date: 2024/9/29 20:10
import os.path

import numpy as np
import pandas as pd
import time
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score


# 计算聚类度量
def calculate_metrics(true_labels, predicted_labels, dataset):
    print("Calculating clustering metrics (approx. 16 mins)...")

    # 内部评估指标
    internal_silhouette = silhouette_score(dataset, predicted_labels)
    internal_db_index = davies_bouldin_score(dataset, predicted_labels)
    # 外部评估指标
    external_ari = adjusted_rand_score(true_labels, predicted_labels)
    external_nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    # 打印各类评估结果
    print(f'Silhouette Coefficient: {internal_silhouette:.4f}')
    print(f'Davies-Bouldin Index: {internal_db_index:.4f}')
    print(f'Adjusted Rand Index: {external_ari:.4f}')
    print(f'Normalized Mutual Information: {external_nmi:.4f}')

    # 计算加权分数
    weights = {'silhouette': 0.25, 'davies_bouldin': 0.25, 'adjusted_rand': 0.25, 'normalized_mutual_info': 0.25}
    final_score = (weights['silhouette'] * internal_silhouette +
                   weights['davies_bouldin'] * (1 - internal_db_index) +
                   weights['adjusted_rand'] * external_ari +
                   weights['normalized_mutual_info'] * external_nmi)

    print(f'Final Weighted Score: {final_score:.4f}')


# 加载数据集
def load_dataset(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df


# 数据预处理
def preprocess_data_min(df, feature_cols, flight_id_col):
    grouped = df.groupby(flight_id_col)
    min_len = grouped.size().min()  # 获取最小分组长度

    # 对齐和截断分组数据
    aligned_groups = {flight_id: group[feature_cols].values[:min_len][::-1]
    if group[feature_cols].values[-1, 2] < group[feature_cols].values[0, 2]
    else group[feature_cols].values[:min_len]
                      for flight_id, group in grouped}
    return aligned_groups


def preprocess_data(df, feature_cols, flight_id_col):
    grouped = df.groupby(flight_id_col)
    max_len = grouped.size().max()  # 获取最大分组长度

    # 对齐和零填充分组数据
    aligned_groups = {}
    for flight_id, group in grouped:
        group_values = group[feature_cols].values
        # 检查是否需要逆序
        if group_values[-1, 2] < group_values[0, 2]:
            group_values = group_values[::-1]

        # 如果分组长度不足，进行0填充
        padded_group = np.zeros((max_len, group_values.shape[1]))
        padded_group[:group_values.shape[0], :] = group_values  # 填充原始数据

        aligned_groups[flight_id] = padded_group

    return aligned_groups, grouped


# 计算距离矩阵
def calculate_distance_matrix(aligned_groups):
    if os.path.exists('data/distance_df.csv'):
        distance_matrix = pd.read_csv('data/distance_df.csv').iloc[:, 1:].to_numpy()
        return distance_matrix

    flight_ids = list(aligned_groups.keys())
    num_flights = len(flight_ids)
    distance_matrix = np.zeros((num_flights, num_flights))

    for i, flight_id1 in enumerate(flight_ids):
        print(f"Calculating distances for flight {flight_id1} ({i + 1}/{num_flights})")
        group1 = aligned_groups[flight_id1]
        for j, flight_id2 in enumerate(flight_ids):
            if i != j:
                group2 = aligned_groups[flight_id2]
                distance_matrix[i, j] = np.mean(np.linalg.norm(group1 - group2, axis=1))

    print("Distance matrix calculation finished.")
    result = pd.DataFrame(distance_matrix, index=flight_ids, columns=flight_ids)
    result.to_csv('data/distance_df.csv', index=True)
    return result


# 提取特征和标签
def extract_features_and_labels(df):
    features = ['x', 'y', 'z']
    return df[features].values, df['label'].values, df['flight_id'].values


# 执行DBSCAN聚类
def perform_dbscan_clustering(distance_matrix, eps=0.025, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    print("DBSCAN clustering in progress...")
    return dbscan.fit_predict(distance_matrix)


# 聚类并计算度量
def calculate_clustering_metrics(labels, predicted_labels, dataset):
    start = time.time()
    calculate_metrics(labels, predicted_labels, dataset)
    print(f"Metrics calculation completed in {time.time() - start:.2f} seconds.")


# 主函数
def cluster(df, distance_matrix_path='data/distance_df.csv'):
    print('Begin clustering process.')

    # 特征与标签提取
    #dataset, labels, flight_ids = extract_features_and_labels(df)
    print('Feature and label extraction complete.')

    # 读取距离矩阵
    print('Reading precomputed distance matrix...')
    # distance_matrix = pd.read_csv(distance_matrix_path).iloc[:, 1:].to_numpy()
    aligned_groups, grouped = preprocess_data(df, ['x', 'y', 'z'], 'flight_id')
    distance_matrix = calculate_distance_matrix(aligned_groups)
    print('Distance matrix loaded.')

    # 执行聚类
    predicted_labels = perform_dbscan_clustering(distance_matrix)
    print('DBSCAN clustering finished.')

    # 扩展聚类标签到每个航班
    #predicted_labels_expanded = np.repeat(predicted_labels, df.groupby('flight_id').size().values)
    #print('Labels unified across flight groups.')

    # 计算并打印聚类的评估指标
    print('Calculating clustering metrics...')
    # calculate_clustering_metrics(labels, predicted_labels_expanded, dataset)
    flight_ids = list(aligned_groups.keys())
    all_grouped_label = [group['label'].values[0] for _, group in grouped]
    all_grouped_values = [aligned_groups[id] for id in flight_ids]


    # 方案1
    # data = np.array([data.mean(axis=0) for data in all_grouped_values])
    # 方案2
    data = np.array([value.flatten() for value in all_grouped_values])
    calculate_clustering_metrics(all_grouped_label, predicted_labels, data)
    print('Clustering process completed.')

    # 输出聚类结果
    print("Cluster labels for each sample:", predicted_labels)
