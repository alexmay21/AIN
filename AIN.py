import tensorflow as tf
from time import time
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)

class AIN:
    def __init__(self, sess, dims, num_context, global_mean, epoch, k, k_c, k_e, keep_prob, learning_rate, batch_size, mlp1_layer, mlp2_layer, mlp1_hidden_size, mlp2_hidden_size, dataset):
        self.sess = sess
        self.dims = dims
        self.global_mean = global_mean
        self.num_users, self.num_items = self.dims[0], self.dims[1]
        self.num_context_dims = len(self.dims) - 2
        self.num_context = num_context
        self.epoch = epoch
        self.k = k
        self.k_c = k_c
        self.k_e = k_e
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mlp1_layer = mlp1_layer
        self.mlp2_layer = mlp2_layer
        self.mlp1_hidden_size = mlp1_hidden_size
        self.mlp2_hidden_size = mlp2_hidden_size

        self.dataset_name = dataset

        self.train_loss_record = []
        self.valid_loss_record = []
        self.test_loss_record = []

        self.train_rmse_record, self.train_mae_record = [], []
        self.valid_rmse_record, self.valid_mae_record = [], []
        self.test_rmse_record, self.test_mae_record = [], []

        self.build_graph()

    def build_graph(self):
        self.u_idx = tf.placeholder(tf.int32, [None], "user_id")
        self.v_idx = tf.placeholder(tf.int32, [None], "item_id")
        self.c1_idx = tf.placeholder(tf.int32, [None], "context1_id")
        self.c2_idx = tf.placeholder(tf.int32, [None], "context2_id")
        self.r = tf.placeholder(tf.float32, [None], "real_rating")
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")

        # initialize weights
        self.weights = self._initialize_weights()

        self.globalmean = tf.constant(self.global_mean, dtype=tf.float32, name="global_mean")

        self.U = weight_variable([self.num_users, self.k], "U")
        self.V = weight_variable([self.num_items, self.k], "V")
        self.U_bias = weight_variable([self.num_users], "U_bias")
        self.V_bias = weight_variable([self.num_items], "V_bias")
        self.C = weight_variable([self.num_context, self.k_c], "C")
        self.C_bias = weight_variable([self.num_context], "C_bias")

        with tf.name_scope("get_latent_vector"):
            self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx, name="U_embed")
            self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx, name="V_embed")
            self.C1_embed = tf.nn.embedding_lookup(self.C, self.c1_idx, name="C1_embed")
            self.C2_embed = tf.nn.embedding_lookup(self.C, self.c2_idx, name="C2_embed")

            self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
            self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)

            self.C1_bias_embed = tf.nn.embedding_lookup(self.C_bias, self.c1_idx)
            self.C2_bias_embed = tf.nn.embedding_lookup(self.C_bias, self.c2_idx)

            self.U_embed = tf.nn.dropout(self.U_embed, self.dropout_keep)
            self.V_embed = tf.nn.dropout(self.V_embed, self.dropout_keep)
            self.C1_embed = tf.nn.dropout(self.C1_embed, self.dropout_keep)
            self.C2_embed = tf.nn.dropout(self.C2_embed, self.dropout_keep)

        with tf.name_scope("concat_user_context"):
            self.user_c1_concat = tf.concat((self.U_embed, self.C1_embed), axis=1)
            self.user_c2_concat = tf.concat((self.U_embed, self.C2_embed), axis=1)

        with tf.name_scope("concat_item_context"):
            self.item_c1_concat = tf.concat((self.V_embed, self.C1_embed), axis=1)
            self.item_c2_concat = tf.concat((self.V_embed, self.C2_embed), axis=1)

        with tf.name_scope("get_total_context_bias"):
            self.total_context_bias = self.C1_bias_embed + self.C2_bias_embed

        with tf.name_scope("user_MLP1"):
            self.h_user_c1_umlp1 = self.user_c1_concat
            self.h_user_c2_umlp1 = self.user_c2_concat

            if self.mlp1_layer > 0:
                for i in range(self.mlp1_layer):
                    self.h_user_c1_umlp1 = tf.add(tf.matmul(self.h_user_c1_umlp1, self.weights['W_userMLP1_layer%d' % i]), self.weights['bias_userMLP1_layer%d' % i])
                    self.h_user_c1_umlp1 = tf.nn.relu(self.h_user_c1_umlp1)
                    self.h_user_c1_umlp1 = tf.nn.dropout(self.h_user_c1_umlp1, self.dropout_keep)

                    self.h_user_c2_umlp1 = tf.add(tf.matmul(self.h_user_c2_umlp1, self.weights['W_userMLP1_layer%d' % i]), self.weights['bias_userMLP1_layer%d' % i])
                    self.h_user_c2_umlp1 = tf.nn.relu(self.h_user_c2_umlp1)
                    self.h_user_c2_umlp1 = tf.nn.dropout(self.h_user_c2_umlp1, self.dropout_keep)

            self.c1_user_effect = tf.matmul(self.h_user_c1_umlp1, self.weights['W_userMLP1_output']) + self.weights['bias_userMLP1_output']
            self.c2_user_effect = tf.matmul(self.h_user_c2_umlp1, self.weights['W_userMLP1_output']) + self.weights['bias_userMLP1_output']

            self.c1_user_effect = tf.nn.dropout(self.c1_user_effect, self.dropout_keep)
            self.c2_user_effect = tf.nn.dropout(self.c2_user_effect, self.dropout_keep)

        with tf.name_scope("item_MLP1"):
            self.h_item_c1_umlp1 = self.item_c1_concat
            self.h_item_c2_umlp1 = self.item_c2_concat

            if self.mlp1_layer > 0:
                for i in range(self.mlp1_layer):
                    self.h_item_c1_umlp1 = tf.add(tf.matmul(self.h_item_c1_umlp1, self.weights['W_itemMLP1_layer%d' % i]), self.weights['bias_itemMLP1_layer%d' % i])
                    self.h_item_c1_umlp1 = tf.nn.relu(self.h_item_c1_umlp1)
                    self.h_item_c1_umlp1 = tf.nn.dropout(self.h_item_c1_umlp1, self.dropout_keep)

                    self.h_item_c2_umlp1 = tf.add(tf.matmul(self.h_item_c2_umlp1, self.weights['W_itemMLP1_layer%d' % i]), self.weights['bias_itemMLP1_layer%d' % i])
                    self.h_item_c2_umlp1 = tf.nn.relu(self.h_item_c2_umlp1)
                    self.h_item_c2_umlp1 = tf.nn.dropout(self.h_item_c2_umlp1, self.dropout_keep)

            self.c1_item_effect = tf.matmul(self.h_item_c1_umlp1, self.weights['W_itemMLP1_output']) + self.weights['bias_itemMLP1_output']
            self.c2_item_effect = tf.matmul(self.h_item_c2_umlp1, self.weights['W_itemMLP1_output']) + self.weights['bias_itemMLP1_output']

            self.c1_item_effect = tf.nn.dropout(self.c1_item_effect, self.dropout_keep)
            self.c2_item_effect = tf.nn.dropout(self.c2_item_effect, self.dropout_keep)

        with tf.name_scope("user_attention"):
            self.attention_c1_user = tf.nn.relu(
                tf.matmul(self.c1_user_effect, self.weights['W1_user_attention'])
                + tf.matmul(self.U_embed, self.weights['W2_user_attention'])
                + self.weights['bias_user_attention']
            )
            self.attention_c2_user = tf.nn.relu(
                tf.matmul(self.c2_user_effect, self.weights['W1_user_attention'])
                + tf.matmul(self.U_embed, self.weights['W2_user_attention'])
                + self.weights['bias_user_attention']
            )

            self.attention_c1_user_exp = tf.exp(self.attention_c1_user)
            self.attention_c2_user_exp = tf.exp(self.attention_c2_user)


            self.attention_user_exp_sum = self.attention_c1_user_exp + self.attention_c2_user_exp

            self.attention_c1_user_out = tf.div(self.attention_c1_user_exp, self.attention_user_exp_sum, name="attention_c1_user_out")
            self.attention_c2_user_out = tf.div(self.attention_c2_user_exp, self.attention_user_exp_sum, name="attention_c2_user_out")

        with tf.name_scope("total_effect_on_user"):
            self.context_effect_on_user = tf.add(tf.multiply(self.attention_c1_user_out, self.c1_user_effect), tf.multiply(self.attention_c2_user_out, self.c2_user_effect))

        with tf.name_scope("concat_user_effect"):
            self.user_effect_concat = tf.concat((self.U_embed, self.context_effect_on_user), axis=1)

        with tf.name_scope("item_attention"):
            self.attention_c1_item = tf.nn.relu(
                tf.matmul(self.c1_item_effect, self.weights['W1_item_attention'])
                + tf.matmul(self.V_embed, self.weights['W2_item_attention'])
                + self.weights['bias_item_attention']
            )
            self.attention_c2_item = tf.nn.relu(
                tf.matmul(self.c2_item_effect, self.weights['W1_item_attention'])
                + tf.matmul(self.V_embed, self.weights['W2_item_attention'])
                + self.weights['bias_item_attention']
            )

            self.attention_c1_item_exp = tf.exp(self.attention_c1_item)
            self.attention_c2_item_exp = tf.exp(self.attention_c2_item)

            self.attention_item_exp_sum = tf.add(self.attention_c1_item_exp, self.attention_c2_item_exp)

            self.attention_c1_item_out = tf.div(self.attention_c1_item_exp, self.attention_item_exp_sum, name="attention_c1_item_out")
            self.attention_c2_item_out = tf.div(self.attention_c2_item_exp, self.attention_item_exp_sum, name="attention_c2_item_out")

        with tf.name_scope("total_effect_on_item"):
            self.context_effect_on_item = tf.add(tf.multiply(self.attention_c1_item_out, self.c1_item_effect),
                                                 tf.multiply(self.attention_c2_item_out, self.c2_item_effect))

        with tf.name_scope("concat_item_effect"):
            self.item_effect_concat = tf.concat((self.V_embed, self.context_effect_on_item), axis=1)

        with tf.name_scope("user_MLP2"):
            self.h_umlp2 = self.user_effect_concat

            if self.mlp2_layer > 0:
                for i in range(self.mlp2_layer):
                    self.h_umlp2 = tf.add(tf.matmul(self.h_umlp2, self.weights['W_userMLP2_layer%d' % i]), self.weights['bias_userMLP2_layer%d' % i])
                    self.h_umlp2 = tf.nn.relu(self.h_umlp2)
                    self.h_umlp2 = tf.nn.dropout(self.h_umlp2, self.dropout_keep)

            self.U_under_context_embed = tf.add(tf.matmul(self.h_umlp2, self.weights['W_userMLP2_output']), self.weights['bias_userMLP2_output'], name="U_under_context_embed")

            self.U_under_context_embed = tf.nn.dropout(self.U_under_context_embed, self.dropout_keep)

        with tf.name_scope("item_MLP2"):
            self.h_vmlp2 = self.item_effect_concat

            if self.mlp2_layer > 0:
                for i in range(self.mlp2_layer):
                    self.h_vmlp2 = tf.add(tf.matmul(self.h_vmlp2, self.weights['W_itemMLP2_layer%d' % i]), self.weights['bias_itemMLP2_layer%d' % i])
                    self.h_vmlp2 = tf.nn.relu(self.h_vmlp2)
                    self.h_vmlp2 = tf.nn.dropout(self.h_vmlp2, self.dropout_keep)

            self.V_under_context_embed = tf.add(tf.matmul(self.h_vmlp2, self.weights['W_itemMLP2_output']), self.weights['bias_itemMLP2_output'], name="V_under_context_embed")

            self.V_under_context_embed = tf.nn.dropout(self.V_under_context_embed, self.dropout_keep)

        with tf.name_scope("predict"):

            self.r_ = tf.reduce_sum(tf.multiply(self.U_under_context_embed, self.V_under_context_embed), reduction_indices=1)
            # add global mean, user bias and item bias
            self.r_ = tf.add(self.r_, self.globalmean)
            self.r_ = tf.add(self.r_, self.U_bias_embed)
            self.r_ = tf.add(self.r_, self.V_bias_embed)
            self.r_ = tf.add(self.r_, self.total_context_bias, name="predicted_score")

        with tf.name_scope("loss"):
            self.loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_))

            # if self.reg_lambda > 0:
            #     self.reg_u = tf.contrib.layers.l2_regularizer(self.reg_lambda)(self.U)
            #     self.reg_v = tf.contrib.layers.l2_regularizer(self.reg_lambda)(self.V)
            #     self.reg_bias = tf.contrib.layers.l2_regularizer(self.reg_lambda)(
            #         self.U_bias) + tf.contrib.layers.l2_regularizer(self.reg_lambda)(
            #         self.V_bias) + tf.contrib.layers.l2_regularizer(self.reg_lambda)(self.C_bias)
            #     self.reg_c = tf.contrib.layers.l2_regularizer(self.reg_lambda)(self.C)
            #     self.squared_loss = self.loss
            #     self.squared_loss += self.reg_u + self.reg_v + self.reg_bias + self.reg_c
            #
            self.squared_loss = self.loss

        with tf.name_scope("optimizer"):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.squared_loss)

        self.merged_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def _initialize_weights(self):
        all_weights = dict()

        # user MLP1 variables
        num_userMLP1_layer = self.mlp1_layer
        if num_userMLP1_layer > 0:
            for i in range(num_userMLP1_layer):
                all_weights['W_userMLP1_layer%d' % i] = tf.get_variable(
                    "W_user_mlp1_layer%d" % i, shape=[(self.mlp1_hidden_size), (self.mlp1_hidden_size)], initializer=tf.contrib.layers.xavier_initializer()
                )
                all_weights['bias_userMLP1_layer%d' % i] = tf.Variable(
                    tf.zeros([(self.mlp1_hidden_size)]), name="bias_user_mlp1_layer%d" % i, dtype=tf.float32
                )
        # user MLP1 output layer
        all_weights['W_userMLP1_output'] = tf.get_variable(
            "W_user_mlp1_output", shape=[(self.mlp1_hidden_size), (self.k_e)],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        all_weights['bias_userMLP1_output'] = tf.Variable(
            tf.zeros([self.k_e]), name="bias_user_mlp1_output", dtype=tf.float32
        )

        # user MLP2 variables
        num_userMLP2_layer = self.mlp2_layer
        if num_userMLP2_layer > 0:
            for i in range(num_userMLP2_layer):
                all_weights['W_userMLP2_layer%d' % i] = tf.get_variable(
                    "W_user_mlp2_layer%d" % i, shape=[(self.mlp2_hidden_size), (self.mlp2_hidden_size)], initializer=tf.contrib.layers.xavier_initializer()
                )
                all_weights['bias_userMLP2_layer%d' % i] = tf.Variable(
                    tf.zeros([(self.mlp2_hidden_size)]), name="bias_user_mlp2_layer%d" % i, dtype=tf.float32
                )
        # user MLP2 output layer
        all_weights['W_userMLP2_output'] = tf.get_variable(
            "W_user_mlp2_output", shape=[(self.mlp2_hidden_size), (self.k)],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        all_weights['bias_userMLP2_output'] = tf.Variable(
            tf.zeros([self.k]), name="bias_user_mlp2_output", dtype=tf.float32
        )

        # item MLP1 variables
        num_itemMLP1_layer = self.mlp1_layer
        if num_itemMLP1_layer > 0:
            for i in range(num_itemMLP1_layer):
                all_weights['W_itemMLP1_layer%d' % i] = tf.get_variable(
                    "W_item_mlp1_layer%d" % i, shape=[(self.mlp1_hidden_size), (self.mlp1_hidden_size)], initializer=tf.contrib.layers.xavier_initializer()
                )
                all_weights['bias_itemMLP1_layer%d' % i] = tf.Variable(
                    tf.zeros([(self.mlp1_hidden_size)]), name="bias_item_mlp1_layer%d" % i, dtype=tf.float32
                )
        # item MLP1 output layer
        all_weights['W_itemMLP1_output'] = tf.get_variable(
            "W_item_mlp1_output", shape=[(self.mlp1_hidden_size), (self.k_e)],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        all_weights['bias_itemMLP1_output'] = tf.Variable(
            tf.zeros([self.k_e]), name="bias_item_mlp1_output", dtype=tf.float32
        )

        # item MLP2 variables
        num_itemMLP2_layer = self.mlp2_layer
        if num_itemMLP2_layer > 0:
            for i in range(num_itemMLP2_layer):
                all_weights['W_itemMLP2_layer%d' % i] = tf.get_variable(
                    "W_item_mlp2_layer%d" % i, shape=[(self.mlp2_hidden_size), (self.mlp2_hidden_size)],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                all_weights['bias_itemMLP2_layer%d' % i] = tf.Variable(
                    tf.zeros([(self.mlp2_hidden_size)]), name="bias_item_mlp2_layer%d" % i, dtype=tf.float32
                )
        # item MLP2 output layer
        all_weights['W_itemMLP2_output'] = tf.get_variable(
            "W_item_mlp2_output", shape=[(self.mlp2_hidden_size), (self.k)],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        all_weights['bias_itemMLP2_output'] = tf.Variable(
            tf.zeros([self.k]), name="bias_item_mlp2_output", dtype=tf.float32
        )

        # user attention layer variables
        all_weights['W1_user_attention'] = tf.get_variable(
            "W1_user_attention", shape=[self.k_e, 1], initializer=tf.contrib.layers.xavier_initializer()
        )

        all_weights['W2_user_attention'] = tf.get_variable(
            "W2_user_attention", shape=[self.k, 1], initializer=tf.contrib.layers.xavier_initializer()
        )

        all_weights['bias_user_attention'] = tf.Variable(
            tf.zeros([1]),
            name="bias_user_attention",
            dtype=tf.float32
        )

        # item attention layer variables
        all_weights['W1_item_attention'] = tf.get_variable(
            "W1_item_attention", shape=[self.k_e, 1], initializer=tf.contrib.layers.xavier_initializer()
        )

        all_weights['W2_item_attention'] = tf.get_variable(
            "W2_item_attention", shape=[self.k, 1], initializer=tf.contrib.layers.xavier_initializer()
        )

        all_weights['bias_item_attention'] = tf.Variable(
            tf.zeros([1]),
            name="bias_item_attention",
            dtype=tf.float32
        )
        #
        # # bi-linear of userMLP1
        # all_weights['W_user_userMLP1_bi'] = tf.get_variable(
        #     "W_user_userMLP1_bi", shape=[self.k, self.hidden_size_mlp1_bi], initializer=tf.contrib.layers.xavier_initializer()
        # )
        # all_weights['W_c_userMLP1_bi'] = tf.get_variable(
        #     "W_c_userMLP1_bi", shape=[self.k_c, self.hidden_size_mlp1_bi], initializer=tf.contrib.layers.xavier_initializer()
        # )
        # all_weights['bias_userMLP1_bi'] = tf.Variable(
        #     tf.zeros([self.hidden_size_mlp1_bi]), name="bias_userMLP1_bi", dtype=tf.float32
        # )
        #
        # # bi-linear of itemMLP1
        # all_weights['W_item_itemMLP1_bi'] = tf.get_variable(
        #     "W_item_itemMLP1_bi", shape=[self.k, self.hidden_size_mlp1_bi], initializer=tf.contrib.layers.xavier_initializer()
        # )
        # all_weights['W_c_itemMLP1_bi'] = tf.get_variable(
        #     "W_c_itemMLP1_bi", shape=[self.k_c, self.hidden_size_mlp1_bi], initializer=tf.contrib.layers.xavier_initializer()
        # )
        # all_weights['bias_itemMLP1_bi'] = tf.Variable(
        #     tf.zeros([self.hidden_size_mlp1_bi]), name="bias_itemMLP1_bi", dtype=tf.float32
        # )
        #
        # # bi-linear of userMLP2
        # all_weights['W_user_userMLP2_bi'] = tf.get_variable(
        #     "W_user_userMLP2_bi", shape=[self.k, self.hidden_size_mlp2_bi], initializer=tf.contrib.layers.xavier_initializer()
        # )
        # all_weights['W_e_userMLP2_bi'] = tf.get_variable(
        #     "W_e_userMLP2_bi", shape=[self.k_e, self.hidden_size_mlp2_bi], initializer=tf.contrib.layers.xavier_initializer()
        # )
        # all_weights['bias_userMLP2_bi'] = tf.Variable(
        #     tf.zeros([self.hidden_size_mlp2_bi]), name="bias_userMLP2_bi", dtype=tf.float32
        # )
        #
        # # bi-linear of itemMLP2
        # all_weights['W_item_itemMLP2_bi'] = tf.get_variable(
        #     "W_item_itemMLP2_bi", shape=[self.k, self.hidden_size_mlp2_bi], initializer=tf.contrib.layers.xavier_initializer()
        # )
        # all_weights['W_e_itemMLP2_bi'] = tf.get_variable(
        #     "W_e_itemMLP2_bi", shape=[self.k_e, self.hidden_size_mlp2_bi], initializer=tf.contrib.layers.xavier_initializer()
        # )
        # all_weights['bias_itemMLP2_bi'] = tf.Variable(
        #     tf.zeros([self.hidden_size_mlp2_bi]), name="bias_itemMLP2_bi", dtype=tf.float32
        # )

        return all_weights

    # shuffle
    def shuffle_in_unison_scary(self,a,b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def construct_feeddict(self, batch_data, batch_label, phase):
        u_idx = batch_data.T[0]
        v_idx = batch_data.T[1]
        c1_idx = batch_data.T[2]
        c2_idx = batch_data.T[3]

        r = batch_label

        if phase == "train":
            return {self.u_idx: u_idx, self.v_idx: v_idx, self.r: r, self.c1_idx: c1_idx, self.c2_idx: c2_idx, self.dropout_keep: self.keep_prob}
        else:
            return {self.u_idx: u_idx, self.v_idx: v_idx, self.r: r, self.c1_idx: c1_idx, self.c2_idx: c2_idx, self.dropout_keep: 1}


    def train(self, Train_data, Validation_data, Test_data, result_path='save/'):
        self.writer = tf.summary.FileWriter("./logs")
        self.writer.add_graph(self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # output the initial evaluation information before the first epoch
        t = time()
        init_valid_rmse, init_valid_mae = self.evaluate(Validation_data)
        init_test_rmse, init_test_mae = self.evaluate(Test_data)
        print("Init: \t validation_rmse=%.4f, validation_mae=%.4f [%.1f s]" % (init_valid_rmse, init_valid_mae, time() - t))
        print("Init: \t test_rmse=%.4f, test_mae=%.4f [%.1f s]" % (init_test_rmse, init_test_mae, time() - t))

        print("epoch: ", self.epoch)
        print("training data size: ", len(Train_data[1]))
        print("batch_size: ", self.batch_size)
        print("batch num：", int(len(Train_data[1]) / self.batch_size))
        for epoch in range(self.epoch):
            # record the start time of each epoch
            epoch_start_time = time()

            # shuffle
            self.shuffle_in_unison_scary(Train_data[0], Train_data[1])

            total_batch = int(len(Train_data[1]) / self.batch_size)

            tr_loss = 0
            for i in range(total_batch+1):
                if i == total_batch:
                    batch_data = Train_data[0][i * self.batch_size:]
                    batch_label = Train_data[1][i * self.batch_size:]
                    print("last batch's batch size", str(len(batch_data)))

                else:
                    batch_data = Train_data[0][i * self.batch_size:(i + 1) * self.batch_size]
                    batch_label = Train_data[1][i * self.batch_size:(i + 1) * self.batch_size]

                # 1.feed
                feed_dict = self.construct_feeddict(batch_data, batch_label, "train")

                loss, _ = self.sess.run([self.squared_loss, self.train_step], feed_dict=feed_dict)
                tr_loss = tr_loss + loss
            print("-----------------------------------------------------------------")
            tr_loss = tr_loss/(total_batch+1)
            print("training loss of epoch {}: {}".format(epoch+1, tr_loss))
            self.train_loss_record.append(tr_loss)
            epoch_end_time = time()

            # evaluate the validation and testing set
            valid_result_rmse, valid_result_mae = self.evaluate(Validation_data)
            test_result_rmse, test_result_mae = self.evaluate(Test_data)

            # add the rmse&mae result of each epoch to the record list
            self.train_rmse_record.append(0)
            self.train_mae_record.append(0)
            self.valid_rmse_record.append(valid_result_rmse)
            self.valid_mae_record.append(valid_result_mae)
            self.test_rmse_record.append(test_result_rmse)
            self.test_mae_record.append(test_result_mae)

            # output the result of each epoch/or specified epoch
            print("Epoch %d [%.1f s]\tvalidation_rmse=%.4f, validation_mae=%.4f [%.1f s]"
                  % (epoch + 1, epoch_end_time - epoch_start_time, valid_result_rmse, valid_result_mae,
                     time() - epoch_end_time))
            print("Epoch %d [%.1f s]\ttest_rmse=%.4f, test_mae=%.4f [%.1f s]"
                  % (epoch + 1, epoch_end_time - epoch_start_time, test_result_rmse, test_result_mae,
                     time() - epoch_end_time))

            if self.eva_termination():
                break

        self.saver.save(self.sess, result_path + "/model.ckpt")
        return self.valid_rmse_record, self.valid_mae_record, self.test_rmse_record, self.test_mae_record

    def eva_termination(self):
        if len(self.valid_rmse_record) > 5 :
            if self.valid_rmse_record[-1] > self.valid_rmse_record[-2] and self.valid_rmse_record[-2] > self.valid_rmse_record[-3] and self.valid_rmse_record[-3] > self.valid_rmse_record[-4] and self.valid_rmse_record[-4] > self.valid_rmse_record[-5]:
                return True
        return False

    def evaluate(self, dataset):
        num_example = len(dataset[1])
        batch_data = dataset[0]
        batch_label = dataset[1]

        # 1.feed
        feed_dict = self.construct_feeddict(batch_data, batch_label, "evaluate")

        predictions = self.sess.run(self.r_, feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(batch_label, (num_example,))


        prediction_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))
        prediction_bounded = np.minimum(prediction_bounded, np.ones(num_example) * max(y_true))

        final_rmse = math.sqrt(mean_squared_error(y_true, prediction_bounded))
        final_mae = mean_absolute_error(y_true, prediction_bounded)

        return final_rmse, final_mae