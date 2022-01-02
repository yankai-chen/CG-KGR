"""
@author:chenyankai
@file:aggregators.py
@time:2020/10/09
"""
import tensorflow as tf

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class UserAggregator(object):
    def __init__(self, args, act_f, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = args.dropout
        self.act_f = act_f
        self.batch_size = args.batch_size
        self.sample_size = args.sample_size
        self.dim = args.node_dim
        self.agg_type = args.agg_type
        self.n_head = args.n_head
        self.n_head = args.n_head


        if self.agg_type in ['sum', 'ngh']:
            with tf.variable_scope(self.name):
                self.weights = tf.get_variable(
                    shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
                self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')
        elif self.agg_type == 'concat':
            with tf.variable_scope(self.name):
                self.weights = tf.get_variable(
                    shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
                self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')
        else:
            raise NotImplementedError

    def _compute_item_ego_embedding(self, self_embeddings, item_ngh_embeddings, W):
        """
        :param self_embeddings:                 [batch_size, dim]
        :param item_ngh_embeddings:             [batch_size, sample_size, dim]
        :param parameters:                      [dim, dim]
        :return:                                [batch_size, dim]
        """
        # [batch_size, 1, dim]
        self_embeddings = tf.reshape(self_embeddings, [self.batch_size, 1, self.dim])
        # [1, dim, dim]
        W = tf.reshape(W, [1, self.dim, self.dim])
        # [batch_size, 1, dim]
        self_embeddings = tf.reshape(self_embeddings, [self.batch_size, 1, self.dim])
        # [batch_size, 1, dim]
        Wu = tf.reshape(tf.reduce_sum(W * self_embeddings, axis=-1), [self.batch_size, 1, self.dim])

        # [n_head*batch_size, 1, dim/n_head]
        Wu = tf.concat(tf.split(Wu, self.n_head, axis=-1), axis=0)
        # [n_head*batch_size, sample_size, dim/n_head]
        item_ngh_embeddings = tf.concat(tf.split(item_ngh_embeddings, self.n_head, axis=-1), axis=0)
        # [n_head*batch_size, sample_size]
        att = tf.reduce_sum(Wu * item_ngh_embeddings, axis=-1) / tf.sqrt(float(self.dim) / float(self.n_head))

        # [n_head*batch_size, sample_size]
        att_norm = tf.nn.softmax(att, dim=1)
        # [n_head*batch_size, sample_size, 1]
        att_norm = tf.expand_dims(att_norm, axis=-1)
        # [n_head*batch_size, dim/n_head]
        ego_embeddings = tf.reduce_sum(att_norm * item_ngh_embeddings, axis=1)
        # [batch_size, dim]
        ego_embeddings = tf.concat(tf.split(ego_embeddings, self.n_head, axis=0), axis=-1)

        return ego_embeddings

    def __call__(self, self_embeddings, item_ngh_embeddings, W):
        """
        :param self_embeddings:                 [batch_size, dim]
        :param item_ngh_embeddings:             [batch_size, sample_size, dim]
        :param W:                               [dim, dim]
        :return:                                [batch_size, dim]
        """
        # [batch_size, dim]
        ego_embeddings = self._compute_item_ego_embedding(self_embeddings, item_ngh_embeddings, W)

        if self.agg_type == 'sum':
            # [batch_size, dim]
            output = self_embeddings + ego_embeddings
        elif self.agg_type == 'concat':
            # [batch_size, 2 * dim]
            output = tf.concat([self_embeddings, ego_embeddings], axis=-1)
        elif self.agg_type == 'ngh':
            # [batch_size, dim]
            output = ego_embeddings
        else:
            raise NotImplementedError

        output = tf.matmul(output, self.weights) + self.bias

        output = tf.nn.dropout(output, keep_prob=1 - self.dropout)
        # [batch_size, dim]
        return self.act_f(output)

    def get_weight(self):
        if self.agg_type in ['sum', 'ngh', 'concat']:
            return self.weights
        else:
            raise NotImplementedError

    def get_bias(self):
        if self.agg_type in ['sum', 'ngh', 'concat']:
            return self.bias
        else:
            raise NotImplementedError


class EntityAggregator(object):
    def __init__(self, args, act_f, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = args.dropout
        self.act_f = act_f
        self.batch_size = args.batch_size
        self.dim = args.node_dim
        self.sample_size = args.sample_size
        self.agg_type = args.agg_type
        self.repr_type = args.repr_type
        self.n_head = args.n_head
        self.a = args.a

        with tf.variable_scope(self.name):
            pass

        if self.agg_type in ['sum', 'ngh']:
            with tf.variable_scope(self.name):
                # [dim, dim]
                self.weights = tf.get_variable(
                    shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
                self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

                self.weights_UI = tf.get_variable(
                    shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights_UI')
                self.bias_UI = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias_UI')

        elif self.agg_type == 'concat':
            with tf.variable_scope(self.name):
                # [2 * dim, dim] for normal entity layers
                self.weights_1 = tf.get_variable(
                    shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(),
                    name='weights_1')
                self.bias_1 = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias_1')
                # [3 * dim, dim] for item layer
                self.weights_2 = tf.get_variable(shape=[self.dim * 3, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer(), name='weights_2')
                self.bias_2 = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias_2')

                self.weights_UI = tf.get_variable(
                    shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights_UI')
                self.bias_UI = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias_UI')

        else:
            raise NotImplementedError

    def _get_representative(self, user_embeddings, item_embeddings):
        """
        :param user_embeddings:         [batch_size, dim]
        :param item_embeddings:         [batch_size, dim]
        :return:                        [batch_size, dim]
        """

        if self.repr_type == 'sum':
            ui_embeddings = (user_embeddings + item_embeddings)
        elif self.repr_type == 'mean':
            ui_embeddings = (user_embeddings + item_embeddings) / 2
        elif self.repr_type == 'max':
            ui_embeddings = tf.where(user_embeddings > item_embeddings, user_embeddings, item_embeddings)
        elif self.repr_type == 'combine':
            ui_embeddings = self.a * user_embeddings + (1.0 - self.a) * item_embeddings
        else:
            raise NotImplementedError

        return ui_embeddings

    def _compute_user_ego_embedding(self, item_embeddings, ngh_user_embeddings, W):
        """
        :param item_embeddings:                 [batch_size, dim]
        :param ngh_user_embeddings:             [batch_size, sample_size, dim]
        :param W:                               [dim, dim]
        :return:                                [batch_size, 1, dim]
        """
        # [batch_size, 1, dim]
        item_embeddings = tf.reshape(item_embeddings, [self.batch_size, 1, self.dim])
        # [1, dim, dim]
        W = tf.reshape(W, [1, self.dim, self.dim])
        # [batch_size, 1, dim]
        item_embeddings = tf.reshape(item_embeddings, [self.batch_size, 1, self.dim])
        # [batch_size, 1, dim]
        Wi = tf.reshape(tf.reduce_sum(W * item_embeddings, axis=-1), [self.batch_size, 1, self.dim])

        # [n_head*batch_size, 1, dim/n_head]
        Wi = tf.concat(tf.split(Wi, self.n_head, axis=-1), axis=0)
        # [n_head*batch_size, sample_size, dim/n_head]
        ngh_user_embeddings = tf.concat(tf.split(ngh_user_embeddings, self.n_head, axis=-1), axis=0)

        # [n_head*batch_size, sample_size]
        att = tf.reduce_sum(Wi * ngh_user_embeddings, axis=-1) / tf.sqrt(float(self.dim) / float(self.n_head))

        # [n_head*batch_size, sample_size]
        att_norm = tf.nn.softmax(att, dim=1)
        # [n_head*batch_size, sample_size, 1]
        att_norm = tf.expand_dims(att_norm, axis=-1)
        # [n_head*batch_size, dim/n_head]
        user_side_ego_embeddings = tf.reduce_sum(att_norm * ngh_user_embeddings, axis=1)
        # [batch_size, dim]
        user_side_ego_embeddings = tf.concat(tf.split(user_side_ego_embeddings, self.n_head, axis=0), axis=-1)
        # [batch_size, 1, dim]
        user_side_ego_embeddings = tf.expand_dims(user_side_ego_embeddings, axis=1)

        #  for ease of collaborative encoding, we compute the UI side aggregation for items

        if self.agg_type == 'sum':
            # [-1, dim]
            output = tf.reshape(item_embeddings + user_side_ego_embeddings, [-1, self.dim])
            output = tf.matmul(output, self.weights_UI) + self.bias_UI
            output = tf.nn.dropout(output, keep_prob=1 - self.dropout)

        elif self.agg_type == 'concat':
            output = tf.concat([item_embeddings, user_side_ego_embeddings,], axis=-1)
            # [-1, dim * 2]
            output = tf.reshape(output, [-1, self.dim * 2])
            output = tf.matmul(output, self.weights_UI) + self.bias_UI
            output = tf.nn.dropout(output, keep_prob=1 - self.dropout)

        elif self.agg_type == 'ngh':
            # [-1, dim]
            output = tf.reshape(user_side_ego_embeddings, [-1, self.dim])
            output = tf.matmul(output, self.weights_UI) + self.bias_UI
            output = tf.nn.dropout(output, keep_prob=1 - self.dropout)
        else:
            raise NotImplementedError
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        # [batch_size, -1, dim]
        output = self.act_f(output)
        # [batch_size, dim]
        output = tf.squeeze(output)
        return user_side_ego_embeddings, output

    def _compute_entity_ego_embedding(self, user_embeddings, item_embeddings, self_embeddings,
                                      ngh_entity_embeddings, W_r):
        """
        we compute the attention for the quadruplet <(u,i), h, r, t>
        :param user_embeddings:                 [batch_size, dim]
        :param item_embeddings:                 [batch_size, dim]
        :param self_embeddings:                 [batch_size, -1, dim]
        :param ngh_entity_embeddings:           [batch_size, -1, sample_size, dim]
        :param W_r:                             [batch_size, -1, sample_size, dim, dim]
        :return:                                [batch_size, -1, dim]
        """

        # [batch_size, dim]
        signal = self._get_representative(user_embeddings, item_embeddings)
        if self.repr_type == 'concat':
            # [batch_size, 1, 1, 1, dim]
            signal = tf.reshape(signal, [self.batch_size, 1, 1, 1, 2 * self.dim])
            # [batch_size, -1, sample_size, dim, 2 * dim]
            W_r = tf.concat([W_r, W_r], axis=-1)
            # [batch_size, -1, sample_size, dim, 2 * dim]
            W_rui = W_r * signal
            # [batch_size, -1, 1, 1, 2 * dim]
            self_embeddings = tf.concat([self_embeddings, self_embeddings], axis=-1)
            self_embeddings = tf.reshape(self_embeddings, [self.batch_size, -1, 1, 1, 2 * self.dim])
            # [batch_size, -1, sample_size, dim]
            # W_rui_mut_Vh = tf.reduce_sum(W_rui * self_embeddings, axis=-1)
            W_rui_mut_Vh = tf.reduce_mean(W_rui * self_embeddings, axis=-1)

        else:
            # [batch_size, 1, 1, 1, dim]
            signal = tf.reshape(signal, [self.batch_size, 1, 1, 1, self.dim])
            # [batch_size, -1, sample_size, dim, dim]
            W_rui = W_r * signal
            # [batch_size, -1, 1, 1, dim]
            self_embeddings = tf.reshape(self_embeddings, [self.batch_size, -1, 1, 1, self.dim])
            # [batch_size, -1, sample_size, dim]
            W_rui_mut_Vh = tf.reduce_sum(W_rui * self_embeddings, axis=-1)

        # [n_head*batch_size, -1, sample_size, dim/n_head]
        W_rui_mut_Vh = tf.concat(tf.split(W_rui_mut_Vh, self.n_head, axis=-1), axis=0)
        ngh_entity_embeddings = tf.concat(tf.split(ngh_entity_embeddings, self.n_head, axis=-1), axis=0)
        # [n_head*batch_size, -1, sample_size]
        att = tf.reduce_sum(W_rui_mut_Vh * ngh_entity_embeddings, axis=-1) / tf.sqrt(
            float(self.dim) / float(self.n_head))

        # [n_head*batch_size, -1, sample_size]
        att_norm = tf.nn.softmax(att, dim=-1)
        # [n_head*batch_size, -1, sample_size, 1]
        att_norm = tf.expand_dims(att_norm, axis=-1)
        # [n_head*batch_size, -1, dim/n_head]
        ego_embedding = tf.reduce_sum(att_norm * ngh_entity_embeddings, axis=2)
        # [batch_size, -1, dim]
        ego_embedding = tf.concat(tf.split(ego_embedding, self.n_head, axis=0), axis=-1)

        return ego_embedding

    def __call__(self, self_embeddings, ngh_user_embeddings, ngh_entity_embeddings,
                 user_embeddings, item_embeddings, parameters, is_item_layer):
        """
        :param self_embeddings:                 [batch_size, -1, dim]
        :param ngh_user_embeddings:             [batch_size, sample_size, dim]
        :param ngh_entity_embeddings:           [batch_size, -1, sample_size, dim]
        :param user_embeddings:                 [batch_size, dim]
        :param item_embeddings:                 [batch_size, dim]
        :param parameters:                      W_r, [batch_size, -1, sample_size, dim, dim]

        :return:                                [batch_size, -1, dim]
        """
        _, num, _ = self_embeddings.shape
        W_r, W_ui = parameters
        # parameter1 = [W_r, None]
        # parameter2 = [W_ui, None]

        # For user side
        # [batch_size, 1, dim]
        user_side_ego_embeddings, item_UI_embedding = self._compute_user_ego_embedding(item_embeddings, ngh_user_embeddings, W_ui)

        if not is_item_layer:

            '''
            If it's not the layer for items, we only aggregate the entity neighbors
            '''
            # [batch_size, -1, dim]
            entity_ego_embeddings = self._compute_entity_ego_embedding(user_embeddings, item_UI_embedding,
                                                                       self_embeddings, ngh_entity_embeddings, W_r)
            # aggregate them up
            if self.agg_type == 'sum':
                # [-1, dim]
                output = tf.reshape(self_embeddings + entity_ego_embeddings, [-1, self.dim])
                output = tf.matmul(output, self.weights) + self.bias

                output = tf.nn.dropout(output, keep_prob=1 - self.dropout)

            elif self.agg_type == 'concat':
                output = tf.concat([self_embeddings, entity_ego_embeddings], axis=-1)
                # [-1, dim * 2]
                output = tf.reshape(output, [-1, self.dim * 2])
                output = tf.matmul(output, self.weights_1) + self.bias_1

                output = tf.nn.dropout(output, keep_prob=1 - self.dropout)

            elif self.agg_type == 'ngh':
                # [-1, dim]
                output = tf.reshape(entity_ego_embeddings, [-1, self.dim])
                output = tf.matmul(output, self.weights) + self.bias

                output = tf.nn.dropout(output, keep_prob=1 - self.dropout)

            else:
                raise NotImplementedError

            output = tf.reshape(output, [self.batch_size, -1, self.dim])

            # [batch_size, -1, dim]
            return self.act_f(output)

        else:
            '''
            If it is the layer for items, we aggregate the user part and entity part together
            '''

            # For entity side
            # [batch_size, -1, dim]
            entity_ego_embeddings = self._compute_entity_ego_embedding(user_embeddings, item_UI_embedding,
                                                                       self_embeddings, ngh_entity_embeddings, W_r)
            # aggregate them up
            if self.agg_type == 'sum':
                # [-1, dim]

                output = tf.reshape(self_embeddings + user_side_ego_embeddings + entity_ego_embeddings, [-1, self.dim])
                output = tf.matmul(output, self.weights) + self.bias
                output = tf.nn.dropout(output, keep_prob=1 - self.dropout)

            elif self.agg_type == 'concat':
                output = tf.concat([self_embeddings, user_side_ego_embeddings, entity_ego_embeddings], axis=-1)
                # [-1, dim * 2]
                output = tf.reshape(output, [-1, self.dim * 3])
                output = tf.matmul(output, self.weights_2) + self.bias_2
                output = tf.nn.dropout(output, keep_prob=1 - self.dropout)

            elif self.agg_type == 'ngh':
                # [-1, dim]
                output = tf.reshape(entity_ego_embeddings + user_side_ego_embeddings, [-1, self.dim])
                output = tf.matmul(output, self.weights) + self.bias
                output = tf.nn.dropout(output, keep_prob=1 - self.dropout)
            else:
                raise NotImplementedError
            output = tf.reshape(output, [self.batch_size, -1, self.dim])

            # [batch_size, -1, dim]
            return self.act_f(output)

    def get_weight(self):
        if self.agg_type in ['sum', 'ngh']:
            return self.weights, None
        elif self.agg_type == 'concat':
            return self.weights_1, self.weights_2
        else:
            raise NotImplementedError

    def get_bias(self):
        if self.agg_type in ['sum', 'ngh']:
            return self.bias, None
        elif self.agg_type == 'concat':
            return self.bias_1, self.bias_2
        else:
            raise NotImplementedError

