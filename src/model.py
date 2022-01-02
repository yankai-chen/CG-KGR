"""
@author:chenyankai
@file:KGR_model.py
@time:2020/10/09
"""
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
from src.aggregators import EntityAggregator, UserAggregator
import numpy as np


class CGKGR(object):
    def __init__(self, args, n_user, n_entity, n_relation):
        self._parse_args(args)
        self._build_inputs()
        self._build_model(args, n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args):
        # [entity_num, neighbor_sample_size]
        self.n_layer = args.n_layer
        self.n_head = args.n_head
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size


        self.dropout = args.dropout
        self.node_dim = args.node_dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr

        # currently we have n_layers for entity aggregator
        self.entity_aggregator = EntityAggregator
        # current we only have 1 layer for user aggregator
        self.user_aggregator = UserAggregator

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        self.adj_u2i = tf.placeholder(dtype=np.int32, shape=[None, None], name='adj_u2i')
        self.adj_i2u = tf.placeholder(dtype=np.int32, shape=[None, None], name='adj_i2u')
        self.adj_e2e = tf.placeholder(dtype=np.int32, shape=[None, None], name='adj_e2e')
        self.adj_relation = tf.placeholder(dtype=np.int32, shape=[None, None], name='adj_relation')

    def _build_model(self, args, n_user, n_entity, n_relation):
        self.n_user = n_user
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.node_dim], initializer=CGKGR.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.node_dim], initializer=CGKGR.get_initializer(), name='entity_emb_matrix')

        self.W_R = tf.get_variable(shape=[n_relation + 1, self.node_dim, self.node_dim],
                                   initializer=tf.contrib.layers.xavier_initializer(), name='transform_weight')

        # [batch_size, dim]
        new_user_embeddings, self.user_aggregator = self.aggregate_for_users(args, self.user_indices)
        new_item_embeddings, self.entity_aggregator_list = self.aggregate_for_items(args, self.item_indices,
                                                                                    new_user_embeddings)

        # [batch_size]
        self.scores = tf.reduce_sum(new_user_embeddings * new_item_embeddings, axis=-1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def _get_item_ngh(self, user_seeds):
        """
        :param user_seeds:      [batch_size]
        :return:                [batch_size, sample_size]
        """
        return tf.reshape(tf.gather(self.adj_u2i, user_seeds), [self.batch_size, -1])

    def _get_user_ngh(self, item_seeds):
        """
        :param item_seeds:      [batch_size]
        :return:                [batch_size, sample_size]
        """
        return tf.reshape(tf.gather(self.adj_i2u, item_seeds), [self.batch_size, -1])

    def _get_entity_ngh_multihop(self, item_seeds):
        """
        :param item_seeds:  [batch_size]
        :return:   entity: {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_layer]}
                   relation: {[batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_layer]}
        """
        #   [batch_size, 1]
        item_seeds = tf.expand_dims(item_seeds, axis=1)
        entities = [item_seeds]
        relations = []
        for i in range(self.n_layer):
            neighbor_entities = tf.reshape(tf.gather(self.adj_e2e, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate_for_users(self, args, user_index):
        """
        :param args:
        :param user_index:              [batch_size]
        :return:                        [batch_size, dim]
        """
        user_aggregator = self.user_aggregator(args, act_f=tf.nn.tanh, name=None)
        # [batch_size, sample_size]
        item_ngh_index = self._get_item_ngh(user_index)
        # [batch_size, d]
        user_embedding = tf.nn.embedding_lookup(self.user_emb_matrix, user_index)
        # [batch_size, sample_size, d]
        item_ngh_embedding = tf.nn.embedding_lookup(self.entity_emb_matrix, item_ngh_index)
        W_ui = tf.nn.embedding_lookup(self.W_R, self.n_relation)

        # W_ui:   [dim, dim]
        output = user_aggregator(user_embedding, item_ngh_embedding, W_ui)

        return output, user_aggregator

    def aggregate_for_items(self, args, item_index, new_user_embeddings):
        """
        :param args:
        :param item_index:              [batch_size]
        :param new_user_embeddings:     [batch_size, dim]
        :param W_R                      [n_relation, dim, dim]
        :return:
        """
        # [batch_size, sample_size]
        ngh_user_index = self._get_user_ngh(item_index)
        # [batch_size, dim]
        item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, item_index)

        # [batch_size, sample_size, dim]
        ngh_user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, ngh_user_index)

        # part 2: aggregate from entity side in KG
        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # entity: {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_layer]}
        # relation: {[batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_layer]}
        entities, relations = self._get_entity_ngh_multihop(item_index)

        entity_aggregators_list = []  # store all entity_aggregators
        entity_embeddings = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        is_item_layer = False
        for i in range(self.n_layer):
            if i == self.n_layer - 1:
                entity_aggregator = self.entity_aggregator(args, act_f=tf.nn.tanh, name=None)
                is_item_layer = True
            else:
                entity_aggregator = self.entity_aggregator(args, act_f=tf.nn.relu, name=None)
            entity_aggregators_list.append(entity_aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_layer - i):
                # [batch_size, -1, sampled_size, dim]
                shape = [self.batch_size, -1, self.sample_size, self.node_dim]
                # [batch_size, -1, sampled_size, dim, dim]
                tmp_index = tf.reshape(relations[hop], [self.batch_size, -1, self.sample_size])

                W_r = tf.nn.embedding_lookup(self.W_R, tmp_index)
                W_ui = tf.nn.embedding_lookup(self.W_R, self.n_relation)
                para = [W_r, W_ui]

                embeddings = entity_aggregator(self_embeddings=entity_embeddings[hop],
                                               ngh_user_embeddings=ngh_user_embeddings,
                                               ngh_entity_embeddings=tf.reshape(entity_embeddings[hop + 1], shape),
                                               user_embeddings=new_user_embeddings,
                                               item_embeddings=item_embeddings,
                                               parameters=para,
                                               is_item_layer=is_item_layer)

                entity_vectors_next_iter.append(embeddings)
            entity_embeddings = entity_vectors_next_iter

        res = tf.reshape(entity_embeddings[0], [self.batch_size, self.node_dim])

        return res, entity_aggregators_list

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix)
        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.W_R)

        user_aggregator = self.user_aggregator
        W = user_aggregator.get_weight()
        self.l2_loss = self.l2_loss + tf.nn.l2_loss(W)
        b = user_aggregator.get_bias()
        self.l2_loss = self.l2_loss + tf.nn.l2_loss(b)

        for aggregator in self.entity_aggregator_list:
            W1, W2 = aggregator.get_weight()
            b1, b2 = aggregator.get_bias()
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(W1)
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(b1)
            if W2 is not None:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(W2)
            if b2 is not None:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(b2)

        self.loss = self.base_loss + self.l2_weight * self.l2_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def eval_save(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        raw_scores = tf.identity(scores)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1, raw_scores.eval()


    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)



