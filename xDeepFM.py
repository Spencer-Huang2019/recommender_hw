# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf

from base_model import BaseModel


__all__ = ["XDeepFMModel"]


class XDeepFMModel(BaseModel):
    """xDeepFM model

    :Citation:

        J. Lian, X. Zhou, F. Zhang, Z. Chen, X. Xie, G. Sun, "xDeepFM: Combining Explicit
        and Implicit Feature Interactions for Recommender Systems", in Proceedings of the
        24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining,
        KDD 2018, London, 2018.
    """

    def _build_graph(self):
        """xddepfm模型的主要函数.

        Returns:
            object: 该模型预测的评分.
        """
        hparams = self.hparams

        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        with tf.variable_scope("XDeepFM") as scope:
            with tf.variable_scope("embedding", initializer=self.initializer) as escope:
                self.embedding = tf.get_variable(
                    name="embedding_layer",
                    shape=[hparams.FEATURE_COUNT, hparams.dim],
                    dtype=tf.float32,
                )
                self.embed_params.append(self.embedding)

                # 获得嵌入层的输出和规模
                embed_out, embed_layer_size = self._build_embedding()

            logit = 0

            # 论文有linear
            if hparams.use_Linear_part:
                print("Add linear part.")
                logit = logit + self._build_linear()

            if hparams.use_FM_part:
                print("Add FM part.")
                logit = logit + self._build_fm()

            # 论文有CIN
            if hparams.use_CIN_part:
                print("Add CIN part.")
                if hparams.fast_CIN_d <= 0:
                    logit = logit + self._build_CIN(
                        embed_out, res=True, direct=False, bias=False, is_masked=True
                    )
                else:
                    logit = logit + self._build_fast_CIN(
                        embed_out, res=True, direct=False, bias=False
                    )

            # 论文有DNN
            if hparams.use_DNN_part:
                print("Add DNN part.")
                logit = logit + self._build_dnn(embed_out, embed_layer_size)

            return logit

    def _build_embedding(self):
        """嵌入层。MLP要求有固定长度的向量作为输入。
        这个函数计算了每个field的特征嵌入池化。

        Returns:
            embedding:  嵌入层输出结果 #_fields * #_dim.
            embedding_size: 嵌入层规模 #_fields * #_dim
        """
        hparams = self.hparams
        fm_sparse_index = tf.SparseTensor(
            self.iterator.dnn_feat_indices, # 非零值元素的索引
            self.iterator.dnn_feat_values, # 提供了indices中的每个元素的值
            self.iterator.dnn_feat_shape, # dense_shape=[3,6]指定二维3x6张量
        )
        fm_sparse_weight = tf.SparseTensor(
            self.iterator.dnn_feat_indices,
            self.iterator.dnn_feat_weights,
            self.iterator.dnn_feat_shape,
        )
        w_fm_nn_input_orgin = tf.nn.embedding_lookup_sparse( # 直接根据特征的值，从embedding中查表得到embedding的结果
            self.embedding, fm_sparse_index, fm_sparse_weight, combiner="sum"
        )
        embedding = tf.reshape( # 将输出结果转换为一个行向量，即 1 x 200
            w_fm_nn_input_orgin, [-1, hparams.dim * hparams.FIELD_COUNT]
        )
        embedding_size = hparams.FIELD_COUNT * hparams.dim
        return embedding, embedding_size

    def _build_linear(self):
        """构建线性模型.
        线性回归.

        Returns:
            object: 线性回归模型预估的分数.
        """
        with tf.variable_scope("linear_part", initializer=self.initializer) as scope:
            # 获取权重向量，size是 10000x1
            w = tf.get_variable(
                name="w", shape=[self.hparams.FEATURE_COUNT, 1], dtype=tf.float32
            )
            # 获取偏置
            b = tf.get_variable(
                name="b",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
            )
            x = tf.SparseTensor(
                self.iterator.fm_feat_indices,
                self.iterator.fm_feat_values,
                self.iterator.fm_feat_shape,
            )
            # 进行线性模型的计算 y=wx+b
            linear_output = tf.add(tf.sparse_tensor_dense_matmul(x, w), b)
            self.layer_params.append(w)
            self.layer_params.append(b)
            tf.summary.histogram("linear_part/w", w) # 用来显示训练过程中变量的分布情况
            tf.summary.histogram("linear_part/b", b)
            return linear_output

    # 本次论文复现不使用fm，故跳过
    def _build_fm(self):
        """Construct the factorization machine part for the model.
        This is a traditional 2-order FM module.

        Returns:
            object: Prediction score made by factorization machine.
        """
        with tf.variable_scope("fm_part") as scope:
            x = tf.SparseTensor(
                self.iterator.fm_feat_indices,
                self.iterator.fm_feat_values,
                self.iterator.fm_feat_shape,
            )
            xx = tf.SparseTensor(
                self.iterator.fm_feat_indices,
                tf.pow(self.iterator.fm_feat_values, 2),
                self.iterator.fm_feat_shape,
            )
            fm_output = 0.5 * tf.reduce_sum(
                tf.pow(tf.sparse_tensor_dense_matmul(x, self.embedding), 2)
                - tf.sparse_tensor_dense_matmul(xx, tf.pow(self.embedding, 2)),
                1,
                keep_dims=True,
            )
            return fm_output

    def _build_CIN(
        self, nn_input, res=False, direct=False, bias=False, is_masked=False
    ):
        """构建CIN网络-compressed interaction network.
        这个模块提供了显性且向量级的高阶特征交互。

        Args:
            nn_input (object): 嵌入层的输出结果作为CIN的输入.
            res (bool): CIN中每一层的结果是否使用残差的结构.
            direct (bool): 如果是true, 则所有隐层的节点都连接到下一层和输出层中;
                    否则, 隐层中一半的节点连接到下一层，另一半节点直接连接到输出层.
            bias (bool): 计算特征图的时候是否添加偏置.
            is_masked (bool): CIN的第一层是否要移除自交互.

        Returns:
            object: CIN模型预测的结果.
        """
        hparams = self.hparams
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        # 将embedding的输出转换成矩阵形式，即X0, size: 20 x 10
        nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), hparams.dim])
        # 存储CIN每一层数据field的数量，即20（一般是不变的）
        field_nums.append(int(field_num))
        # 存储CIN每一层的输出Xk，和第一层的输入X0
        hidden_nn_layers.append(nn_input)
        final_result = []
        # 将一个张量划分成多个张量。将X0分解为10列
        split_tensor0 = tf.split(hidden_nn_layers[0], hparams.dim * [1], 2)
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                # 在第k层处理的时，获取k-1层的输出X(k-1)，然后将其分解为10列
                split_tensor = tf.split(hidden_nn_layers[-1], hparams.dim * [1], 2)
                # 计算 X(k-1)*X0,matmul是矩阵相乘，不应该是对应元素相乘么？
                dot_result_m = tf.matmul(
                    split_tensor0, split_tensor, transpose_b=True
                )  # shape :  (Dim, Batch, FieldNum, HiddenNum), a.k.a (D,B,F,H)
                dot_result_o = tf.reshape(
                    dot_result_m,
                    shape=[hparams.dim, -1, field_nums[0] * field_nums[-1]],
                )  # shape: (D,B,FH)
                dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])  # (B,D,FH)

                # 获取W参数（卷积核）
                filters = tf.get_variable(
                    name="f_" + str(idx),
                    shape=[1, field_nums[-1] * field_nums[0], layer_size],
                    dtype=tf.float32,
                )

                # is_masked=true：去除第一层的自交互（self-interaction）
                # 如果只有1层，按公式计算会出现 X0*X0的情况
                if is_masked and idx == 0:
                    # 值全为1的m*m矩阵，m表示field的数量。因为只有1层，所以取第0个元素
                    ones = tf.ones([field_nums[0], field_nums[0]], dtype=tf.float32)
                    # matrix_band_part 复制张量ones，保留上三角元素值，下三角元素值都置0
                    # mask_matrix 最后是保留了上三角元素值，其他都为0，为什么这么做？
                    mask_matrix = tf.matrix_band_part(ones, 0, -1) - tf.diag(
                        tf.ones(field_nums[0])
                    )
                    mask_matrix = tf.reshape(
                        mask_matrix, shape=[1, field_nums[0] * field_nums[0]]
                    )

                    # 矩阵对应元素相乘。
                    dot_result = tf.multiply(dot_result, mask_matrix) * 2
                    self.dot_result = dot_result

                # CIN每一层的输出经过CNN的卷积过程
                curr_out = tf.nn.conv1d(
                    dot_result, filters=filters, stride=1, padding="VALID"
                )  # shape : (B,D,H`)

                # 如果设置了偏置，则添加偏置信息
                if bias:
                    b = tf.get_variable(
                        name="f_b" + str(idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    curr_out = tf.nn.bias_add(curr_out, b)
                    self.cross_params.append(b)

                # 是否在隐藏层中使用batch normalization
                if hparams.enable_BN is True:
                    curr_out = tf.layers.batch_normalization(
                        curr_out,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                # 输出经过激活函数处理
                curr_out = self._activate(curr_out, hparams.cross_activation)

                curr_out = tf.transpose(curr_out, perm=[0, 2, 1])  # shape : (B,H,D)

                # direct=true：隐藏层的所有节点连接到下一层和输出层
                if direct:
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    # 隐藏层一半节点连接到下一层，一半节点连接到输出层
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        # 当不是CIN最后一层的情况，一半节点到下一层，一半节点到输出层
                        next_hidden, direct_connect = tf.split(
                            curr_out, 2 * [int(layer_size / 2)], 1
                        )
                        final_len += int(layer_size / 2)
                    else:
                        # CIN网络的最后一层，所有节点都连接到输出层
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))

                final_result.append(direct_connect)
                # 添加CIN的下一层信息
                hidden_nn_layers.append(next_hidden)

                self.cross_params.append(filters)

            # 将矩阵进行拼接，行数不变
            result = tf.concat(final_result, axis=1)
            # 在倒数第1个维度上进行求和，进行sum pooling
            result = tf.reduce_sum(result, -1)  # shape : (B,H)

            # res=true：使用残差结构结合到CIN每一层的结果中
            if res:
                base_score = tf.reduce_sum(result, 1, keepdims=True)  # (B,1)
            else:
                base_score = 0

            # 获得训练后的W？
            w_nn_output = tf.get_variable(
                name="w_nn_output", shape=[final_len, 1], dtype=tf.float32
            )
            # 获得训练后的偏置
            b_nn_output = tf.get_variable(
                name="b_nn_output",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)

            # CIN模型预测的输出
            # tf.nn.xw_plus_b用来计算 matmul(x, weights)+biases
            exFM_out = base_score + tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)
            return exFM_out

    # 用了两层卷积层
    def _build_fast_CIN(self, nn_input, res=False, direct=False, bias=False):
        """Construct the compressed interaction network with reduced parameters.
        This component provides explicit and vector-wise higher-order feature interactions.
        Parameters from the filters are reduced via a matrix decomposition method.
        Fast CIN is more space and time efficient than CIN.

        Args:
            nn_input (object): The output of field-embedding layer. This is the input for CIN.
            res (bool): Whether use residual structure to fuse the results from each layer of CIN.
            direct (bool): If true, then all hidden units are connected to both next layer and output layer;
                    otherwise, half of hidden units are connected to next layer and the other half will be connected to output layer.
            bias (bool): Whether to add bias term when calculating the feature maps.

        Returns:
            object: Prediction score made by fast CIN.
        """
        hparams = self.hparams
        hidden_nn_layers = []
        field_nums = []
        final_len = 0
        field_num = hparams.FIELD_COUNT
        fast_CIN_d = hparams.fast_CIN_d
        nn_input = tf.reshape(
            nn_input, shape=[-1, int(field_num), hparams.dim]
        )  # (B,F,D)
        nn_input = tf.transpose(nn_input, perm=[0, 2, 1])  # (B,D,F)
        field_nums.append(int(field_num))
        hidden_nn_layers.append(nn_input)
        final_result = []
        with tf.variable_scope("exfm_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.cross_layer_sizes):
                if idx == 0:
                    fast_w = tf.get_variable(
                        "fast_CIN_w_" + str(idx),
                        shape=[1, field_nums[0], fast_CIN_d * layer_size],
                        dtype=tf.float32,
                    )

                    self.cross_params.append(fast_w)
                    dot_result_1 = tf.nn.conv1d(
                        nn_input, filters=fast_w, stride=1, padding="VALID"
                    )  # shape: (B,D,d*H)
                    dot_result_2 = tf.nn.conv1d(
                        tf.pow(nn_input, 2),
                        filters=tf.pow(fast_w, 2),
                        stride=1,
                        padding="VALID",
                    )  # shape: ((B,D,d*H)
                    dot_result = tf.reshape(
                        0.5 * (dot_result_1 - dot_result_2),
                        shape=[-1, hparams.dim, layer_size, fast_CIN_d],
                    )
                    curr_out = tf.reduce_sum(
                        dot_result, 3, keepdims=False
                    )  # shape: ((B,D,H)
                else:
                    fast_w = tf.get_variable(
                        "fast_CIN_w_" + str(idx),
                        shape=[1, field_nums[0], fast_CIN_d * layer_size],
                        dtype=tf.float32,
                    )
                    fast_v = tf.get_variable(
                        "fast_CIN_v_" + str(idx),
                        shape=[1, field_nums[-1], fast_CIN_d * layer_size],
                        dtype=tf.float32,
                    )

                    self.cross_params.append(fast_w)
                    self.cross_params.append(fast_v)

                    dot_result_1 = tf.nn.conv1d(
                        nn_input, filters=fast_w, stride=1, padding="VALID"
                    )  # shape: ((B,D,d*H)
                    dot_result_2 = tf.nn.conv1d(
                        hidden_nn_layers[-1], filters=fast_v, stride=1, padding="VALID"
                    )  # shape: ((B,D,d*H)
                    dot_result = tf.reshape(
                        tf.multiply(dot_result_1, dot_result_2),
                        shape=[-1, hparams.dim, layer_size, fast_CIN_d],
                    )
                    curr_out = tf.reduce_sum(
                        dot_result, 3, keepdims=False
                    )  # shape: ((B,D,H)

                if bias:
                    b = tf.get_variable(
                        name="f_b" + str(idx),
                        shape=[1, 1, layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    curr_out = tf.nn.bias_add(curr_out, b)
                    self.cross_params.append(b)

                if hparams.enable_BN is True:
                    curr_out = tf.layers.batch_normalization(
                        curr_out,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                curr_out = self._activate(curr_out, hparams.cross_activation)

                if direct:
                    direct_connect = curr_out
                    next_hidden = curr_out
                    final_len += layer_size
                    field_nums.append(int(layer_size))

                else:
                    if idx != len(hparams.cross_layer_sizes) - 1:
                        next_hidden, direct_connect = tf.split(
                            curr_out, 2 * [int(layer_size / 2)], 2
                        )
                        final_len += int(layer_size / 2)
                        field_nums.append(int(layer_size / 2))
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                        field_nums.append(int(layer_size))

                final_result.append(direct_connect)
                hidden_nn_layers.append(next_hidden)

            result = tf.concat(final_result, axis=2)
            result = tf.reduce_sum(result, 1, keepdims=False)  # (B,H)

            if res:
                base_score = tf.reduce_sum(result, 1, keepdims=True)  # (B,1)
            else:
                base_score = 0

            w_nn_output = tf.get_variable(
                name="w_nn_output", shape=[final_len, 1], dtype=tf.float32
            )
            b_nn_output = tf.get_variable(
                name="b_nn_output",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output) + base_score

        return exFM_out

    def _build_dnn(self, embed_out, embed_layer_size):
        """构建MLP模型
        这个模块提供了隐性高阶特征交互.

        Args:
            embed_out (object): 嵌入层的输出，作为DNN的输入.
            embed_layer_size (object): 嵌入层输出的规模 field_num * embedding_dim

        Returns:
            object: DNN模型预测的输出结果.
        """
        hparams = self.hparams
        w_fm_nn_input = embed_out
        last_layer_size = embed_layer_size
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(w_fm_nn_input)
        with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
            for idx, layer_size in enumerate(hparams.layer_sizes):
                # 获取当前网络层的权重参数
                curr_w_nn_layer = tf.get_variable(
                    name="w_nn_layer" + str(layer_idx),
                    shape=[last_layer_size, layer_size],
                    dtype=tf.float32,
                )
                # 获取当前网络层的偏置参数
                curr_b_nn_layer = tf.get_variable(
                    name="b_nn_layer" + str(layer_idx),
                    shape=[layer_size],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )
                tf.summary.histogram(
                    "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                )
                tf.summary.histogram(
                    "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                )
                # 当前隐藏层输出 wx+b
                curr_hidden_nn_layer = tf.nn.xw_plus_b(
                    hidden_nn_layers[layer_idx], curr_w_nn_layer, curr_b_nn_layer
                )
                scope = "nn_part" + str(idx)
                # 获取当前隐藏层的激活函数类型
                activation = hparams.activation[idx]

                if hparams.enable_BN is True:
                    curr_hidden_nn_layer = tf.layers.batch_normalization(
                        curr_hidden_nn_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                # 隐藏层输出经过激活函数处理
                curr_hidden_nn_layer = self._active_layer(
                    logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                )
                # 添加当前隐藏层信息
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            # 获取输出层的权重信息
            w_nn_output = tf.get_variable(
                name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
            )
            # 获取输出层的偏置信息
            b_nn_output = tf.get_variable(
                name="b_nn_output",
                shape=[1],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
            )
            tf.summary.histogram(
                "nn_part/" + "w_nn_output" + str(layer_idx), w_nn_output
            )
            tf.summary.histogram(
                "nn_part/" + "b_nn_output" + str(layer_idx), b_nn_output
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)

            # DNN输出层的输出结果
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output
