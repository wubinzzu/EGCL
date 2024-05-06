import numpy as np
from util import timer
import tensorflow as tf
from model.AbstractRecommender import SocialAbstractRecommender
from util.tool import inner_product, l2_loss
import scipy.sparse as sp
from scipy.special import factorial


class EGCL(SocialAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(EGCL, self).__init__(dataset, conf)
        self.embedding_size = int(conf["embedding_size"])
        self.num_epochs = int(conf["epochs"])
        self.beta = float(conf["beta"])
        self.embedding_reg = float(conf["embedding_reg"])
        self.weight_reg = float(conf["weight_reg"])
        self.temperature = float(conf["temperature"])
        self.intra_reg = float(conf["intra_reg"])
        self.inter_reg = float(conf["inter_reg"])
        self.lr = float(conf["learning_rate"])
        self.layer_size = int(conf["layer_size"])
        self.verbose = int(conf["verbose"])

        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.sess = sess

    def _create_recsys_adj_mat(self):

        user_item_idx = [[u, i] for (u, i), r in self.dataset.train_matrix.todok().items()]
        user_list, item_list = list(zip(*user_item_idx))

        self.user_idx = tf.constant(user_list, dtype=tf.int32, shape=None, name="user_idx")
        self.item_idx = tf.constant(item_list, dtype=tf.int32, shape=None, name="item_idx")

        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))

        rating_matrix = sp.csr_matrix((ratings, (user_np, item_np)), shape=(self.num_users, self.num_items))
        n_inter = np.asarray(rating_matrix.sum(axis=0))
        rec_n_inter = np.power(n_inter, -1).reshape([1, -1])
        rec_n_inter[np.isinf(rec_n_inter)] = 0.
        norm_rating_matrix = rating_matrix.multiply(rec_n_inter)

        return self._normalize_spmat(adj_mat), rowsum, norm_rating_matrix

    def _create_social_adj_mat(self):

        uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
        u1_idx, u2_idx = list(zip(*uu_idx))

        self.u1_idx = tf.constant(u1_idx, dtype=tf.int32, shape=None, name="u1_idx")
        self.u2_idx = tf.constant(u2_idx, dtype=tf.int32, shape=None, name="u2_idx")

        u1_idx = np.array(u1_idx, dtype=np.int32)
        u2_idx = np.array(u2_idx, dtype=np.int32)
        ratings = np.ones_like(u1_idx, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (u1_idx, u2_idx)), shape=(self.num_users, self.num_users))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))

        social_n = np.asarray(tmp_adj.sum(axis=0))
        social_n_inter = np.power(social_n, -1).reshape([1, -1])
        social_n_inter[np.isinf(social_n_inter)] = 0.
        norm_social_matrix = tmp_adj.multiply(social_n_inter)

        return self._normalize_spmat(adj_mat), rowsum, norm_social_matrix

    def _normalize_spmat(self, adj_mat):
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()

        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.is_train_phase = tf.placeholder_with_default(0., shape=None)

    def _create_variables(self):
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = tf.Variable(initializer(shape=[self.num_users, self.embedding_size]),
                                           name='user_embeddings')
        self.item_embeddings = tf.Variable(initializer(shape=[self.num_items, self.embedding_size]),
                                           name='item_embeddings')

        self.W1 = tf.Variable(tf.truncated_normal(shape=[3 * self.embedding_size, self.embedding_size], mean=0.0,
                                                  stddev=0.01), dtype=tf.float32)
        self.B1 = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0,
                                                  stddev=0.01), dtype=tf.float32)

        self.W2 = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, 1], mean=0.0,
                                                  stddev=0.01), dtype=tf.float32)
        self.B2 = tf.Variable(tf.truncated_normal(shape=[1], mean=0.0,
                                                  stddev=0.01), dtype=tf.float32)

        self.W3 = tf.Variable(tf.truncated_normal(shape=[3 * self.embedding_size, self.embedding_size], mean=0.0,
                                                  stddev=0.01), dtype=tf.float32)
        self.B3 = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0,
                                                  stddev=0.01), dtype=tf.float32)

        self.W4 = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, 1], mean=0.0,
                                                  stddev=0.01), dtype=tf.float32)
        self.B4 = tf.Variable(tf.truncated_normal(shape=[1], mean=0.0,
                                                  stddev=0.01), dtype=tf.float32)

        self.W_u_att = tf.Variable(tf.truncated_normal(shape=[2 * self.embedding_size, self.embedding_size], mean=0.0,
                                                       stddev=0.01), dtype=tf.float32)
        self.b_u_att = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0, stddev=0.01),
                                   dtype=tf.float32)
        self.h_u_att = tf.Variable(tf.ones([self.embedding_size, 1]), dtype=tf.float32)

        self.W_s_att = tf.Variable(tf.truncated_normal(shape=[2 * self.embedding_size, self.embedding_size], mean=0.0,
                                                       stddev=0.01), dtype=tf.float32)
        self.b_s_att = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0, stddev=0.01),
                                   dtype=tf.float32)
        self.h_s_att = tf.Variable(tf.ones([self.embedding_size, 1]), dtype=tf.float32)

        self.W_i_att = tf.Variable(tf.truncated_normal(shape=[2 * self.embedding_size, self.embedding_size], mean=0.0,
                                                       stddev=0.01), dtype=tf.float32)
        self.b_i_att = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0, stddev=0.01),
                                   dtype=tf.float32)
        self.h_i_att = tf.Variable(tf.ones([self.embedding_size, 1]), dtype=tf.float32)

    def _social_gcn(self):
        e = 1e-7
        norm_adj, deg, norm_social_matrix = self._create_social_adj_mat()
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)

        degree = tf.sigmoid(tf.log(tf.convert_to_tensor(deg) + e))
        t_u = degree
        user_t = t_u
        user_t = tf.stop_gradient(user_t)

        all_embed = self.user_embeddings
        agg_embed = all_embed
        embs = []
        for k in range(self.layer_size):
            interact_mat = norm_adj
            side_embeddings = tf.sparse_tensor_dense_matmul(interact_mat, agg_embed, name="sparse_dense")
            user_embedds = side_embeddings

            user_embedds = user_embedds * (
                    tf.exp(-user_t) * tf.pow(user_t, tf.constant([k], dtype=tf.float32)) / tf.constant(
                [factorial(k)], dtype=tf.float32))

            side_embeddings_cur = user_embedds
            agg_embed = side_embeddings
            embs.append(side_embeddings_cur)
        all_embeddings = tf.stack(embs, 1)

        initial_social_embeddings = tf.tile(tf.expand_dims(self.user_embeddings, axis=1), tf.stack([1, self.layer_size, 1]))
        mlp_output = tf.matmul(tf.reshape(tf.concat([all_embeddings, initial_social_embeddings], axis=2),
                                          [-1, 2 * self.embedding_size]),self.W_s_att) + self.b_s_att
        mlp_output = tf.nn.relu(mlp_output)
        A_ = tf.reshape(tf.matmul(mlp_output, self.h_s_att), [self.num_users, self.layer_size])
        exp_A_ = tf.exp(A_)
        exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)
        A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)

        return tf.reduce_sum(A * all_embeddings, axis=1, keepdims=False), norm_social_matrix

    def _recsys_gcn(self):
        e = 1e-7
        norm_adj, deg, norm_rating_matrix = self._create_recsys_adj_mat()
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)

        degree = tf.sigmoid(tf.log(tf.convert_to_tensor(deg) + e))
        t_u, t_i = tf.split(degree, [self.num_users, self.num_items], axis=0)
        user_t = t_u
        item_t = t_i
        user_t = tf.stop_gradient(user_t)
        item_t = tf.stop_gradient(item_t)

        all_embed = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        agg_embed = all_embed

        embs = []
        for k in range(self.layer_size):
            interact_mat = norm_adj
            side_embeddings = tf.sparse_tensor_dense_matmul(interact_mat, agg_embed, name="sparse_dense")
            user_embedds, item_embedds = tf.split(side_embeddings, [self.num_users, self.num_items], axis=0)

            user_embedds = user_embedds * (
                    tf.exp(-user_t) * tf.pow(user_t, tf.constant([k], dtype=tf.float32)) / tf.constant(
                [factorial(k)], dtype=tf.float32))
            item_embedds = item_embedds * (
                    tf.exp(-item_t) * tf.pow(item_t, tf.constant([k], dtype=tf.float32)) / tf.constant(
                [factorial(k)], dtype=tf.float32))

            side_embeddings_cur = tf.concat([user_embedds, item_embedds], axis=0)
            agg_embed = side_embeddings
            embs.append(side_embeddings_cur)

        all_embeddings = tf.stack(embs, 1)
        user_embeddings, item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)

        # user-side attention
        initial_user_embeddings = tf.tile(tf.expand_dims(self.user_embeddings, axis=1), tf.stack([1, self.layer_size, 1]))
        mlp_output_user = tf.matmul(tf.reshape(tf.concat([user_embeddings, initial_user_embeddings], axis=2),
                                               [-1, 2 * self.embedding_size]), self.W_u_att) + self.b_u_att
        mlp_output_user = tf.nn.relu(mlp_output_user)
        A_u = tf.reshape(tf.matmul(mlp_output_user, self.h_u_att), [self.num_users, self.layer_size])
        exp_A_u = tf.exp(A_u)
        exp_sum_u = tf.reduce_sum(exp_A_u, 1, keepdims=True)
        Att_u = tf.expand_dims(tf.div(exp_A_u, exp_sum_u), 2)

        # item-side attention
        initial_item_embeddings = tf.tile(tf.expand_dims(self.item_embeddings, axis=1), tf.stack([1, self.layer_size, 1]))
        mlp_output_item = tf.matmul(tf.reshape(tf.concat([item_embeddings, initial_item_embeddings], axis=2),
                                               [-1, 2 * self.embedding_size]), self.W_i_att) + self.b_i_att
        mlp_output_item = tf.nn.relu(mlp_output_item)
        A_i = tf.reshape(tf.matmul(mlp_output_item, self.h_i_att), [self.num_items, self.layer_size])
        exp_A_i = tf.exp(A_i)
        exp_sum_i = tf.reduce_sum(exp_A_i, 1, keepdims=True)
        Att_i = tf.expand_dims(tf.div(exp_A_i, exp_sum_i), 2)
        return tf.reduce_sum(Att_u * user_embeddings, axis=1, keepdims=False), tf.reduce_sum(Att_i * item_embeddings, axis=1, keepdims=False), norm_rating_matrix

    def _fast_loss(self, embeddings_a, embeddings_b, index_a, index_b, alpha):
        term1 = tf.matmul(embeddings_a, embeddings_a, transpose_a=True)
        term2 = tf.matmul(embeddings_b, embeddings_b, transpose_a=True)
        loss1 = tf.reduce_sum(tf.multiply(term1, term2))

        embed_a = tf.nn.embedding_lookup(embeddings_a, index_a)
        embed_b = tf.nn.embedding_lookup(embeddings_b, index_b)
        pos_ratings = inner_product(embed_a, embed_b)

        loss2 = tf.reduce_sum((alpha - 1) * tf.square(pos_ratings)) - 2.0 * tf.reduce_sum(alpha * pos_ratings)
        loss3 = tf.reduce_sum(alpha)

        return loss1 + loss2, loss3

    def InfoNCE(self, view1, view2, temperature):
        normalize_view1 = tf.nn.l2_normalize(view1, 1)
        normalize_view2 = tf.nn.l2_normalize(view2, 1)
        pos_score = tf.reduce_sum(tf.multiply(normalize_view1, normalize_view2), axis=1)
        ttl_score = tf.matmul(normalize_view1, normalize_view2, transpose_a=False, transpose_b=True)
        pos_score = tf.exp(pos_score / temperature)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / temperature), axis=1)
        cl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))

        return cl_loss

    def build_graph(self):
        self._create_placeholder()
        self._create_variables()

        user_embeddings, item_embeddings, norm_rating_matrix = self._recsys_gcn()
        u_embeds1 = tf.nn.embedding_lookup(user_embeddings, self.user_idx)
        i_embeds1 = tf.nn.embedding_lookup(item_embeddings, self.item_idx)
        cat_ui = tf.concat([u_embeds1, i_embeds1, tf.multiply(u_embeds1, i_embeds1)], axis=1)
        embed_ui = tf.nn.relu(tf.matmul(cat_ui, self.W1) + self.B1)
        s_ui = tf.nn.softplus(tf.matmul(embed_ui, self.W2) + self.B2) + 8
        s_ui = tf.squeeze(s_ui)

        ui_loss12, ui_loss3 = self._fast_loss(user_embeddings, item_embeddings, self.user_idx, self.item_idx, s_ui)

        self.rec_weight_loss = ui_loss12 + ui_loss3 + self.weight_reg * l2_loss(self.W1, self.W2, self.B1, self.B2)
        self.rec_embedding_loss = ui_loss12 + self.embedding_reg * l2_loss(self.user_embeddings,
                                  self.item_embeddings) + self.weight_reg * l2_loss(self.W_u_att, self.h_u_att, self.b_u_att,
                                  self.W_i_att, self.h_i_att, self.b_i_att)

        social_embeddings, norm_social_matrix = self._social_gcn()
        u_embeds2 = tf.nn.embedding_lookup(social_embeddings, self.u1_idx)
        u_embeds3 = tf.nn.embedding_lookup(social_embeddings, self.u2_idx)
        cat_uu = tf.concat([u_embeds2, u_embeds3, tf.multiply(u_embeds2, u_embeds3)], axis=1)
        embed_uu = tf.nn.relu(tf.matmul(cat_uu, self.W3) + self.B3)
        s_uu = tf.nn.softplus(tf.matmul(embed_uu, self.W4) + self.B4) + 10
        s_uu = tf.squeeze(s_uu)

        uu_loss12, uu_loss3 = self._fast_loss(social_embeddings, social_embeddings, self.u1_idx, self.u2_idx, s_uu)

        self.soc_weight_loss = uu_loss12 + uu_loss3 + self.weight_reg * l2_loss(self.W3, self.W4, self.B3, self.B4)
        self.soc_embedding_loss = uu_loss12 + self.weight_reg * l2_loss(self.W_s_att, self.h_s_att, self.b_s_att)

        self.weight_loss = self.rec_weight_loss + self.beta * self.soc_weight_loss
        self.embedding_loss = self.rec_embedding_loss + self.beta * self.soc_embedding_loss

        ###  intra-view SSL
        norm_rating_matrix = self._convert_sp_mat_to_sp_tensor(norm_rating_matrix)
        item_neiborhood_embeddings = tf.sparse_tensor_dense_matmul(norm_rating_matrix, item_embeddings)
        user_neiborhood_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(norm_rating_matrix, perm=[1, 0]), user_embeddings)
        norm_social_matrix = self._convert_sp_mat_to_sp_tensor(norm_social_matrix)
        social_neiborhood_embeddings = tf.sparse_tensor_dense_matmul(norm_social_matrix, social_embeddings)

        self.SL_SSL_loss = self.embedding_loss + self.intra_reg * (
                    self.InfoNCE(user_embeddings, item_neiborhood_embeddings, self.temperature) +
                    self.InfoNCE(social_embeddings, social_neiborhood_embeddings, self.temperature) +
                    self.InfoNCE(item_embeddings, user_neiborhood_embeddings, self.temperature)) + \
                    self.inter_reg * self.InfoNCE(user_embeddings, social_embeddings, self.temperature)

        ada1 = tf.train.AdamOptimizer(learning_rate=self.lr)
        ada2 = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.embedding_optimizer = ada1.minimize(self.SL_SSL_loss, var_list=[self.user_embeddings, self.item_embeddings,
                                                                             self.W_u_att, self.h_u_att, self.b_u_att,
                                                                             self.W_i_att, self.h_i_att, self.b_i_att,
                                                                             self.W_s_att, self.h_s_att, self.b_s_att])

        self.weight_optimizer = ada2.minimize(self.weight_loss, var_list=[self.W1, self.W2,
                                                                          self.B1, self.B2, self.W3, self.W4, self.B3,
                                                                          self.B4])

        self.item_embeddings_final = tf.Variable(tf.zeros([self.num_items, self.embedding_size]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.num_users, self.embedding_size]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, user_embeddings),
                           tf.assign(self.item_embeddings_final, item_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.user_ph)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True)

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.num_epochs):
            _, _ = self.sess.run([self.weight_optimizer, self.weight_loss], feed_dict={self.is_train_phase: 1})
            _, _ = self.sess.run([self.embedding_optimizer, self.SL_SSL_loss])
            if epoch >= 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            ratings = self.sess.run(self.batch_ratings, feed_dict={self.user_ph: user_ids, self.is_train_phase: 0})
        else:
            ratings = self.sess.run(self.batch_ratings, feed_dict={self.user_ph: user_ids, self.is_train_phase: 0})
            ratings = [ratings[idx][test_items] for idx, test_items in enumerate(candidate_items)]

        return ratings
