"""
@author:chenyankai
@file:KGR_train.py
@time:2020/10/09
"""

from utility.data_loader import *
from utility.log_helper import *
import random
from src.model import *
# from utility1.metrics import *
# from utility1.helper import *
import numpy as np
import tensorflow as tf
from time import time


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def get_ngh_sample_feed_dict(model, adj_u2i, adj_i2u, adj_e2e, adj_relation):
    feed_dict = {model.adj_u2i: adj_u2i,
                 model.adj_i2u: adj_i2u,
                 model.adj_e2e: adj_e2e,
                 model.adj_relation: adj_relation}
    return feed_dict


def ctr_evaluate(sess, model, data, batch_size, ngh_sample_feed_dict):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        feed_dict = get_feed_dict(model, data, start, start + batch_size)
        feed_dict.update(ngh_sample_feed_dict)
        auc, f1 = model.eval(sess, feed_dict)
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_evaluate(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size, ngh_sample_feed_dict):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            feed_dict = {model.user_indices: [user] * batch_size,
                         model.item_indices: test_item_list[start:start + batch_size]}
            feed_dict.update(ngh_sample_feed_dict)
            items, scores = model.get_scores(sess, feed_dict=feed_dict)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            feed_dict = {model.user_indices: [user] * batch_size,
                         model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)}
            feed_dict.update(ngh_sample_feed_dict)

            items, scores = model.get_scores(sess, feed_dict)
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        # item_sorted = item_sorted[:100]  # at most top@100

        hits = np.zeros(len(item_sorted))
        index = [i for i, x in enumerate(item_sorted) if x in test_record[user]]
        hits[index] = 1

        for k in k_list:
            hit_k = hits[:k]
            hit_num = np.sum(hit_k)
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))
            dcg = np.sum((2 ** hit_k - 1) / np.log2(np.arange(2, k + 2)))
            sorted_hits_k = np.flip(np.sort(hits))[:k]
            idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)))
            # idcg[idcg == 0] = np.inf
            ndcg_list[k].append(dcg / idcg)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg


def get_total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def topk_settings(train_data, test_data, n_item):
    user_num = 100
    k_list = [1, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, item_set, k_list


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
