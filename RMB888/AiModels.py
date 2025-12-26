import json
import os
import autokeras as ak
import keras_tuner
import tensorflow as tf
from kerastuner import Objective
from kerastuner.tuners import RandomSearch

import ComClass.BaseClass as bs
import ComClass.ComFun as comf


# 旋转位置编码 (RoPE) 层
class RotaryPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RotaryPositionEmbedding, self).__init__(**kwargs)
    
    def call(self, x, seq_len=None):
        d_model = x.shape[-1]
        dtype = x.dtype  # 使用输入张量的类型以支持混合精度
        
        # 获取批次维度
        batch_size = tf.shape(x)[0]
        
        # 计算位置索引
        pos_indices = tf.range(0, seq_len, dtype=tf.int32)[:, None]  # [seq_len, 1]
        dim_indices = tf.range(0, d_model // 2, dtype=tf.int32)[None, :]  # [1, d_model//2]
        
        # 转换为浮点类型进行数值计算
        pos = tf.cast(pos_indices, dtype)  # [seq_len, 1]
        i = tf.cast(dim_indices, dtype)  # [1, d_model//2]
        
        # 使用安全的方式计算角度
        inv_freq = 1.0 / (10000 ** (2 * i / d_model))  # [1, d_model//2]
        angle = tf.matmul(pos, inv_freq)  # [seq_len, d_model//2]
        
        # 计算正弦和余弦
        sin_part = tf.sin(angle)  # [seq_len, d_model//2]
        cos_part = tf.cos(angle)  # [seq_len, d_model//2]
        
        # 扩展维度以便广播：[seq_len, d_model//2] -> [batch_size, seq_len, d_model//2]
        sin_part = tf.tile(sin_part[tf.newaxis, :, :], [batch_size, 1, 1])
        cos_part = tf.tile(cos_part[tf.newaxis, :, :], [batch_size, 1, 1])
        
        # 确保三角函数结果与输入类型一致
        sin_part = tf.cast(sin_part, dtype)
        cos_part = tf.cast(cos_part, dtype)
        
        # 对输入进行切片
        x1 = x[..., ::2]  # [batch_size, seq_len, d_model//2]
        x2 = x[..., 1::2]  # [batch_size, seq_len, d_model//2]
        
        # 应用RoPE
        rotated_x1 = x1 * cos_part - x2 * sin_part
        rotated_x2 = x1 * sin_part + x2 * cos_part
        
        # 拼接结果
        rotated = tf.concat([rotated_x1, rotated_x2], axis=-1)  # [batch_size, seq_len, d_model]
        
        return rotated
    
    def get_config(self):
        config = super(RotaryPositionEmbedding, self).get_config()
        return config

# 相对位置偏差层
class RelativePositionBias(tf.keras.layers.Layer):
    def __init__(self, num_heads, max_distance=128, **kwargs):
        super(RelativePositionBias, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.max_distance = max_distance
    
    def build(self, input_shape):
        self.relative_attention_bias_table = self.add_weight(
            shape=(2 * self.max_distance - 1, self.num_heads),
            initializer='zeros',
            trainable=True,
            name='relative_attention_bias_table'
        )
        super(RelativePositionBias, self).build(input_shape)
    
    def call(self, x):
        q_len = tf.shape(x)[1]
        k_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        
        # 计算相对位置
        relative_position = tf.range(q_len)[:, None] - tf.range(k_len)[None, :]
        relative_position_clipped = tf.clip_by_value(relative_position, -self.max_distance + 1, self.max_distance - 1)
        
        # 获取相对位置偏差 [q_len, k_len, num_heads]
        relative_position_bias = tf.gather(self.relative_attention_bias_table, relative_position_clipped + self.max_distance - 1)
        
        return relative_position_bias
    
    def get_config(self):
        config = super(RelativePositionBias, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'max_distance': self.max_distance
        })
        return config

# 学习率调度器 - 结合预热期和余弦退火
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, total_steps):
        super(WarmupCosineSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_steps - warmup_steps,
            alpha=0.1  # 设置最低学习率为初始值的10%
        )

    def __call__(self, step):
        # 预热期：线性增加学习率
        warmup_lr = self.initial_learning_rate * step / self.warmup_steps
        # 余弦退火期：逐渐降低学习率
        cosine_lr = self.cosine_decay(step - self.warmup_steps)
        # 选择学习率：预热期使用线性增加的学习率，之后使用余弦退火的学习率
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }
        return config

# Stochastic Depth层
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_rate=0.1, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_rate = drop_rate
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        dtype = inputs.dtype  # 获取输入张量类型以支持混合精度
        keep_prob = tf.cast(1 - self.drop_rate, dtype)
        batch_size = tf.shape(inputs)[0]
        shape = [batch_size] + [1] * (len(inputs.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=dtype)
        binary_tensor = tf.floor(random_tensor)
        return inputs / keep_prob * binary_tensor
    
    def get_config(self):
        config = super(StochasticDepth, self).get_config()
        config.update({
            'drop_rate': self.drop_rate
        })
        return config

class TFModels(object):
    
    def __init__(self, dset: bs._Dset = bs._Dset(), DDD: bs._Data = bs._Data()):
        self.DDD = DDD
        self.inShape = []
        self.OutShape = []
        self.dset = dset
        self.loss_name = dset.loss_name
        self.metrics_name = dset.metrics_name  # 保留字符串形式用于文件名格式化
        self.metrics_func = dset.metrics_name  # 创建新属性存储函数对象
        
        # Transformer模型参数
        self.dim = 128  # 嵌入维度，增加到128以适应70个输入特征
        self.num_heads = 4  # 注意力头数量，保持4个以确保128/4=32的头维度
        self.num_layers = 3  # Transformer层数量，增加到3层以增强特征提取能力
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)  # 优化器
        if dset.metrics_name == "cailoss":
            self.metrics_func = comf.TfComFun.cailoss
        if dset.metrics_name == "cailoss1":
            self.metrics_func = comf.TfComFun.cailoss1
        if dset.metrics_name == "cailoss2":
            self.metrics_func = comf.TfComFun.cailoss2
        if dset.metrics_name == "cai3dloss":
            self.metrics_func = comf.TfComFun.cai3dloss
        if dset.metrics_name == "cai3dloss2":
            self.metrics_func = comf.TfComFun.cai3dloss2
        if dset.metrics_name == "cai3dloss3":
            self.metrics_func = comf.TfComFun.cai3dloss3
        if dset.metrics_name == "rmbloss":
            self.metrics_func = comf.TfComFun.rmbloss
        if dset.metrics_name == "cuploss":
            self.metrics_func = comf.TfComFun.cuploss
        if dset.metrics_name == "cuploss_z":
            self.metrics_func = comf.TfComFun.cuploss_z
        if dset.metrics_name == "myacc":
            self.metrics_func = comf.TfComFun.myacc
        if dset.metrics_name == "guaccuracy":
            self.metrics_func = comf.TfComFun.guaccuracy

        # tf.keras.mixed_precision.set_global_policy("mixed_float16")  # 注释掉混合精度以避免类型不匹配错误

    def LSTM(self):
        model_try = tf.keras.Sequential()
        model_try.add(tf.keras.layers.Input(shape=self.inShape))
        model_try.add(tf.keras.layers.LSTM(units=256, return_sequences=True, activation='tanh'))
        model_try.add(tf.keras.layers.Dropout(0.1))
        model_try.add(tf.keras.layers.LSTM(units=128, return_sequences=True, activation='tanh'))
        model_try.add(tf.keras.layers.Dropout(0.1))
        model_try.add(tf.keras.layers.LSTM(units=64, return_sequences=True, activation='tanh'))
        # model_try.add(tf.keras.layers.Dropout(0.1))
        model_try.add(tf.keras.layers.LSTM(units=64, activation='tanh'))
        # model_try.add(tf.keras.layers.Dropout(0.1))
        model_try.add(
            tf.keras.layers.Dense(units=32,
                                  activation=tf.nn.swish))  # kernel_regularizer=tf.keras.regularizers.l2(0.001)
        # model_try.add(
        #     tf.keras.layers.Dense(units=32, activation=tf.nn.swish, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model_try.add(tf.keras.layers.Dense(units=self.OutShape))
        model_try.compile(tf.keras.optimizers.Adam(0.001)  # 优化方法Nadam,adam,sgd随机,=tf.keras.optimizers.Adam(0.001),
                          , loss=self.loss_name  # mse,mae,kld,Huber,mean_squared_error,myloss
                          , metrics=self.metrics_func  # valloss#计算正确率，RootMeanSquaredError，MAPE，MeanSquaredError
                          )
        model_try.summary()
        return model_try

    pass

    def LSTM_GPT(self):
        """
        GPT优化后的LSTM模型
        @return:
        """
        model_try = tf.keras.Sequential()
        model_try.add(tf.keras.layers.Input(shape=self.inShape))

        model_try.add(tf.keras.layers.LSTM(units=256, return_sequences=True, activation='tanh',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model_try.add(tf.keras.layers.GaussianNoise(0.5))
        model_try.add(tf.keras.layers.Dropout(0.1))

        model_try.add(tf.keras.layers.LSTM(units=128, return_sequences=True, activation='tanh',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model_try.add(tf.keras.layers.GaussianNoise(0.5))
        model_try.add(tf.keras.layers.Dropout(0.1))

        model_try.add(tf.keras.layers.LSTM(units=64, return_sequences=True, activation='tanh',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model_try.add(tf.keras.layers.GaussianNoise(0.5))
        model_try.add(tf.keras.layers.Dropout(0.1))

        model_try.add(
            tf.keras.layers.LSTM(units=64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model_try.add(tf.keras.layers.GaussianNoise(0.5))

        model_try.add(tf.keras.layers.Dense(units=32, activation=tf.nn.swish))
        model_try.add(tf.keras.layers.Dense(units=self.OutShape))

        # 学习率调整
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100, decay_rate=0.9, staircase=True)

        model_try.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self.loss_name,
            metrics=self.metrics_func
        )

        model_try.summary()
        return model_try

    def LSTM_F(self, isClass=True, activeName=tf.nn.swish):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.inShape))
        model.add(tf.keras.layers.LSTM(units=256, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.LSTM(units=128, return_sequences=True, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units=64, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(32, activation=activeName))  # activation='linear', activation=tf.nn.swish
        # model.add(tf.keras.layers.Dense(16, activation=tf.nn.swish))
        if isClass:
            if self.dset.UseOneHot:
                outsp = self.OutShape
            else:
                outsp = self.dset.OneHotCount
            # if isClass:
            #     outsp = self.dset.OneHotCount
            model.add(tf.keras.layers.Dense(outsp, activation=tf.nn.softmax))
        else:
            model.add(tf.keras.layers.Dense(self.OutShape))
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt,
                      loss=self.loss_name,  # 'sparse_categorical_crossentropy',
                      metrics=self.metrics_func)  # ['accuracy']
        model.summary()
        return model

    def HierarchicalCaiTransformerOld(self, isClass=True, activeName=tf.nn.gelu):
        """HierarchicalCaiTransformer - 很牛的3D彩票模型+特征了
        
        特征已在SQL中预计算完成，包括：奇偶统计、大小统计、和值、跨度、位置关系、号码类型等
        模型仅保留核心Transformer结构，移除所有Python端特征工程代码
        """
        # 输入层
        inputs = tf.keras.layers.Input(shape=self.inShape)
        x = inputs
        
        # 彩票号码归一化
        x = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)(x)
        
        # 嵌入层：将输入特征转换为嵌入向量
        embedded = tf.keras.layers.Dense(self.dim, activation=activeName)(x)
        embedded = tf.keras.layers.LayerNormalization(epsilon=1e-6)(embedded)
        
        # 应用旋转位置编码 (RoPE)
        seq_len = self.inShape[0]  # 使用静态序列长度
        x = RotaryPositionEmbedding()(embedded, seq_len=seq_len)

        # 堆叠多个注意力层
        head_dim = self.dim // self.num_heads
        for _ in range(self.num_layers):
            # 多头自注意力
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=head_dim,
                dropout=0.1
            )(x, x, x, attention_mask=None)
            
            # 确保注意力输出与输入类型一致
            attn_output = tf.cast(attn_output, x.dtype)
            
            # 残差连接和层归一化
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # 前馈网络
            ff_output = tf.keras.layers.Dense(self.dim * 4, activation=activeName)(x)
            ff_output = tf.keras.layers.Dense(x.shape[-1])(ff_output)
            
            # 确保前馈网络输出与输入类型一致
            ff_output = tf.cast(ff_output, x.dtype)
            
            # Stochastic Depth
            ff_output = StochasticDepth(drop_rate=0.1)(ff_output)
            
            # 残差连接和层归一化
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        
        # 池化层
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # 分类/回归头
        if isClass:
            if hasattr(self.dset, 'UseOneHot') and self.dset.UseOneHot:
                outsp = self.OutShape
            else:
                outsp = getattr(self.dset, 'OneHotCount', self.OutShape)
            outputs = tf.keras.layers.Dense(outsp, activation='softmax', name='output')(x)
        else:
            outputs = tf.keras.layers.Dense(self.OutShape, activation='linear', name='output')(x)
        
        # 构建模型
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(optimizer=self.opt, loss=self.loss_name, metrics=self.metrics_func)
        
        model.summary()
        
        return model
    def HierarchicalCaiTransformer(self, isClass=True, activeName=tf.nn.gelu):
        """
        彩票预测专用Transformer模型 - 改进版本
        结合多尺度特征提取、增强的位置编码和分层注意力机制
        
        参数:
            isClass: 是否为分类任务
            activeName: 激活函数，默认使用GELU
        
        返回:
            编译好的Keras模型
        """
        # 输入层
        inputs = tf.keras.layers.Input(shape=self.inShape)
        x = inputs
        
        # 彩票号码归一化
        x = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)(x)
        
        # ====== 1. 多尺度特征提取 ======
        # 使用不同大小的卷积核提取不同时间尺度的特征
        scales = []
        
        # 尺度1: 短期特征 (3天窗口)
        scale1 = tf.keras.layers.Conv1D(filters=self.dim, kernel_size=3, padding='same', activation=activeName)(x)
        scale1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(scale1)
        scales.append(scale1)
        
        # 尺度2: 中期特征 (5天窗口)
        scale2 = tf.keras.layers.Conv1D(filters=self.dim, kernel_size=5, padding='same', activation=activeName)(x)
        scale2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(scale2)
        scales.append(scale2)
        
        # 尺度3: 长期特征 (7天窗口)
        scale3 = tf.keras.layers.Conv1D(filters=self.dim, kernel_size=7, padding='same', activation=activeName)(x)
        scale3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(scale3)
        scales.append(scale3)
        
        # 融合多尺度特征
        fused = tf.keras.layers.Concatenate(axis=-1)(scales)
        fused = tf.keras.layers.Dense(self.dim, activation=activeName)(fused)
        fused = tf.keras.layers.LayerNormalization(epsilon=1e-6)(fused)
        
        # ====== 2. 增强的位置编码 ======
        seq_len = self.inShape[0]
        embedded = fused
        
        # 应用旋转位置编码 (RoPE)
        x = RotaryPositionEmbedding()(embedded, seq_len=seq_len)
        
        # ====== 3. 分层注意力机制 ======
        head_dim = self.dim // self.num_heads
        
        # 存储不同层的特征用于后续融合
        layer_features = []
        
        for i in range(self.num_layers):
            # 多头自注意力
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=head_dim,
                dropout=0.1
            )(x, x, x, attention_mask=None)
            
            # 确保注意力输出与输入类型一致
            attn_output = tf.cast(attn_output, x.dtype)
            
            # 残差连接和层归一化
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # 改进的前馈网络
            ff_output = tf.keras.layers.Dense(self.dim * 4, activation=activeName)(x)
            ff_output = tf.keras.layers.Dropout(0.1)(ff_output)
            ff_output = tf.keras.layers.Dense(self.dim * 2, activation=activeName)(ff_output)
            ff_output = tf.keras.layers.Dense(x.shape[-1])(ff_output)
            
            # 确保前馈网络输出与输入类型一致
            ff_output = tf.cast(ff_output, x.dtype)
            
            # Stochastic Depth
            ff_output = StochasticDepth(drop_rate=0.1)(ff_output)
            
            # 残差连接和层归一化
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
            
            # 保存当前层的特征
            layer_features.append(x)
        
        # ====== 4. 跨层特征融合 ======
        # 合并不同层次的特征
        if len(layer_features) > 1:
            # 从每个层取最后一个时间步的特征
            layer_outputs = [tf.keras.layers.GlobalAveragePooling1D()(feat) for feat in layer_features]
            fused_layers = tf.keras.layers.Concatenate()(layer_outputs)
            # 特征转换
            fused_layers = tf.keras.layers.Dense(self.dim, activation=activeName)(fused_layers)
            fused_layers = tf.keras.layers.LayerNormalization(epsilon=1e-6)(fused_layers)
        else:
            fused_layers = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # ====== 5. 最终特征融合 ======
        # 全局池化获取序列级特征
        global_avg = tf.keras.layers.GlobalAveragePooling1D()(x)
        global_max = tf.keras.layers.GlobalMaxPooling1D()(x)
        
        # 融合所有特征
        final_features = tf.keras.layers.Concatenate()([global_avg, global_max, fused_layers])
        
        # 特征转换
        final_features = tf.keras.layers.Dense(self.dim, activation=activeName)(final_features)
        final_features = tf.keras.layers.Dropout(0.2)(final_features)
        final_features = tf.keras.layers.Dense(self.dim // 2, activation=activeName)(final_features)
        final_features = tf.keras.layers.LayerNormalization(epsilon=1e-6)(final_features)
        
        # ====== 6. 输出层 ======
        if isClass:
            if hasattr(self.dset, 'UseOneHot') and self.dset.UseOneHot:
                outsp = self.OutShape
            else:
                outsp = getattr(self.dset, 'OneHotCount', self.OutShape)
            outputs = tf.keras.layers.Dense(outsp, activation='softmax', name='output')(final_features)
        else:
            outputs = tf.keras.layers.Dense(self.OutShape, activation='linear', name='output')(final_features)
        
        # 构建模型
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # ====== 7. 优化器和编译 ======
        # 使用学习率调度器
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        
        # 优化的Adam优化器
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            clipnorm=1.0
        )
        
        # 编译模型
        model.compile(optimizer=opt, loss=self.loss_name, metrics=self.metrics_func)
        
        model.summary()
        
        return model

    def HierarchicalCaiTransformerL2(self, isClass=True, activeName=tf.nn.gelu):
        """
        轻量级分层Transformer彩票预测模型L2版本
        基于HierarchicalCaiTransformerOld，增加过拟合控制机制
        结合彩票预测成熟经验：号码分布平衡、周期性模式、位置相关性等
        
        主要改进:
        1. 增强正则化控制过拟合
        2. 引入彩票特有的号码平衡机制
        3. 优化注意力权重分布
        4. 改进残差连接和特征融合
        5. 保持与Old版本相似的参数复杂度
        
        参数:
            isClass: 是否为分类任务，True 表示分类，False 表示回归
            activeName: 激活函数，默认使用 GELU

        返回:
            编译好的 Keras 模型
        """
        # 输入层
        inputs = tf.keras.layers.Input(shape=self.inShape)
        x = inputs
        
        # ===== 1. 彩票数据预处理增强 =====
        # 基础归一化
        x = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6)(x)
        
        # 彩票特有的号码平衡处理 - 基于号码分布均匀性
        # 通过轻微的噪声注入来模拟号码平衡特性
        noise = tf.keras.layers.GaussianNoise(stddev=0.01)(x)
        x = tf.keras.layers.Add()([x, noise])
        
        # ===== 2. 增强嵌入层 =====
        # 主要嵌入 - 保持与Old版本相似的维度
        embedded = tf.keras.layers.Dense(self.dim, activation=activeName, 
                                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        embedded = tf.keras.layers.LayerNormalization(epsilon=1e-6)(embedded)
        
        # 位置信息增强嵌入 - 专门处理彩票时序特性
        pos_enhanced = tf.keras.layers.Dense(self.dim // 4, activation=activeName)(x)
        pos_enhanced = tf.keras.layers.LayerNormalization(epsilon=1e-6)(pos_enhanced)
        
        # 融合主要嵌入和位置增强嵌入
        embedded = tf.keras.layers.Concatenate()([embedded, pos_enhanced])
        embedded = tf.keras.layers.Dense(self.dim, activation=activeName)(embedded)
        embedded = tf.keras.layers.LayerNormalization(epsilon=1e-6)(embedded)
        
        # ===== 3. 旋转位置编码 =====
        seq_len = self.inShape[0]
        x = RotaryPositionEmbedding()(embedded, seq_len=seq_len)
        
        # ===== 4. 核心Transformer层 - 保持与Old版本相似的复杂度 =====
        head_dim = self.dim // self.num_heads
        layer_features = []
        
        for layer_idx in range(self.num_layers):
            # ===== 4.1 增强多头自注意力 =====
            # 添加Layer-wise注意力权重衰减
            attention_dropout = 0.1 + (layer_idx * 0.05)  # 随层数增加dropout
            
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=head_dim,
                dropout=attention_dropout,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )(x, x, x, attention_mask=None)
            
            # 确保注意力输出与输入类型一致
            attn_output = tf.cast(attn_output, x.dtype)
            
            # ===== 4.2 改进的残差连接和层归一化 =====
            # 使用Pre-LayerNorm模式，提高训练稳定性
            norm_x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            x = x + attn_output
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + norm_x)
            
            # ===== 4.3 改进的前馈网络 =====
            # 两层前馈网络，增加正则化
            ff_output = tf.keras.layers.Dense(self.dim * 2, activation=activeName,
                                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
            ff_output = tf.keras.layers.Dropout(attention_dropout)(ff_output)
            ff_output = tf.keras.layers.Dense(self.dim, activation=activeName,
                                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(ff_output)
            ff_output = tf.keras.layers.Dropout(attention_dropout)(ff_output)
            
            # 维度适配层
            ff_output = tf.keras.layers.Dense(x.shape[-1])(ff_output)
            
            # 确保前馈网络输出与输入类型一致
            ff_output = tf.cast(ff_output, x.dtype)
            
            # ===== 4.4 Stochastic Depth - 增强版本 =====
            # 彩票数据适合更保守的深度随机化
            drop_rate = 0.1 + (layer_idx * 0.02)  # 随层数轻微增加
            ff_output = StochasticDepth(drop_rate=drop_rate)(ff_output)
            
            # ===== 4.5 最终残差连接 =====
            x = x + ff_output
            
            # 保存层特征用于后续融合
            layer_features.append(x)
            
            # ===== 4.6 层间特征融合增强 =====
            if layer_idx > 0 and layer_idx % 2 == 0:  # 每隔两层进行特征融合
                # 使用注意力机制进行层间特征融合
                fused_layer = tf.keras.layers.GlobalAveragePooling1D()(x)
                fused_layer = tf.keras.layers.Reshape((1, -1))(fused_layer)
                attention_weights = tf.keras.layers.Dense(1, activation='sigmoid')(fused_layer)
                
                # 加权融合当前层和前一层
                prev_feature = layer_features[layer_idx - 1]
                prev_feature_pooled = tf.keras.layers.GlobalAveragePooling1D()(prev_feature)
                prev_feature_pooled = tf.keras.layers.Reshape((1, -1))(prev_feature_pooled)
                
                weighted_prev = prev_feature_pooled * attention_weights
                weighted_current = fused_layer * (1 - attention_weights)
                
                # 扩展维度以匹配原始序列长度
                seq_weight = tf.keras.layers.Dense(self.dim, activation='sigmoid')(weighted_prev + weighted_current)
                seq_weight = tf.keras.layers.Reshape((1, self.dim))(seq_weight)
                
                # 应用权重到当前层
                x = x * seq_weight
        
        # ===== 5. 彩票特性特征提取 =====
        # 提取不同时间粒度的特征
        final_features_list = []
        
        for feat in layer_features[-2:]:  # 使用最后两层特征
            # 全局平均池化 - 捕捉整体趋势
            avg_pool = tf.keras.layers.GlobalAveragePooling1D()(feat)
            final_features_list.append(avg_pool)
            
            # 最大池化 - 捕捉峰值特征
            max_pool = tf.keras.layers.GlobalMaxPooling1D()(feat)
            final_features_list.append(max_pool)
            
            # 注意力池化 - 学习性权重分配
            attention_pool = tf.keras.layers.GlobalAveragePooling1D()(feat)
            attention_pool = tf.keras.layers.Reshape((1, -1))(attention_pool)
            attention_weights = tf.keras.layers.Dense(1, activation='softmax')(attention_pool)
            attention_pool = attention_pool * attention_weights
            attention_pool = tf.keras.layers.Flatten()(attention_pool)
            final_features_list.append(attention_pool)
        
        # 融合所有特征
        final_features = tf.keras.layers.Concatenate()(final_features_list)
        
        # ===== 6. 特征转换和正则化 =====
        # 多层特征变换，增加正则化
        final_features = tf.keras.layers.Dense(self.dim * 2, activation=activeName,
                                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(final_features)
        final_features = tf.keras.layers.Dropout(0.3)(final_features)
        
        final_features = tf.keras.layers.Dense(self.dim, activation=activeName,
                                             kernel_regularizer=tf.keras.regularizers.l2(0.001))(final_features)
        final_features = tf.keras.layers.Dropout(0.2)(final_features)
        
        final_features = tf.keras.layers.Dense(self.dim // 2, activation=activeName)(final_features)
        final_features = tf.keras.layers.LayerNormalization(epsilon=1e-6)(final_features)
        
        # ===== 7. 输出层 =====
        if isClass:
            if hasattr(self.dset, 'UseOneHot') and self.dset.UseOneHot:
                outsp = self.OutShape
            else:
                outsp = getattr(self.dset, 'OneHotCount', self.OutShape)
            outputs = tf.keras.layers.Dense(outsp, activation='softmax', name='output')(final_features)
        else:
            outputs = tf.keras.layers.Dense(self.OutShape, activation='linear', name='output')(final_features)
        
        # ===== 8. 构建模型 =====
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # ===== 9. 优化器和编译 - 增强版本 =====
        # 使用学习率调度器 - 适合彩票数据的训练策略
        initial_lr = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        
        # 优化的Adam优化器 - 针对彩票预测任务调优
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            clipnorm=1.0  # 梯度裁剪，防止梯度爆炸
        )
        
        # 编译模型
        model.compile(optimizer=opt, loss=self.loss_name, metrics=self.metrics_func)
        
        model.summary()
        
        return model
       

    """
    LSTM自动模型搜索
    """
    def LSTM_FA(self, isClass=True):
        # clf = ak.ImageClassifier(
        #     num_classes=self.dset.OneHotCount,
        #     directory=self.dset.SearchModelfile,
        #     overwrite=False,
        #     max_trials=self.dset.SearchModelMaxTry,  # 训练10个模型
        #     loss=self.loss_name,  # 'binary_crossentropy',
        #     metrics=self.metrics_name,
        #     objective="val_" + self.dset.metrics_name
        # )  # [tf.keras.metrics.BinaryAccuracy()]
        # 定义自动机器学习模型
        valloss = keras_tuner.Objective('val_' + self.dset.metrics_name, direction=self.dset.valLossSaveMode)
        input_node = ak.Input()
        lstm_node = ak.RNNBlock(layer_type='lstm', return_sequences=True)(input_node)
        output_node = ak.DenseBlock()(lstm_node)
        if isClass:
            output_node = ak.ClassificationHead()(output_node)
        else:
            output_node = ak.RegressionHead()(output_node)
        clf = ak.AutoModel(inputs=input_node, outputs=output_node,

                           # num_classes=self.dset.OneHotCount,
                           directory=self.dset.SearchModelfile,
                           overwrite=False,
                           max_trials=self.dset.SearchModelMaxTry,  # 训练10个模型
                           loss=self.loss_name,  # 'binary_crossentropy',
                           metrics=self.metrics_func,
                           objective=valloss
                           # keras_tuner.Objective(self.metrics_name, direction=self.dset.valLossSaveMode)
                           # "val_" + self.dset.metrics_name
                           )
        return clf

    pass

    def RNN(self):
        model_try = tf.keras.Sequential()
        model_try.add(tf.keras.layers.Input(shape=self.inShape))
        model_try.add(tf.keras.layers.Dense(units=128, activation=tf.nn.swish))
        # model_try.add(tf.keras.layers.Dropout(0.1))
        model_try.add(tf.keras.layers.Dense(units=64, ))
        model_try.add(
            tf.keras.layers.Dense(units=64, activation=tf.nn.swish, kernel_regularizer=tf.keras.regularizers.l2(0.1)))
        model_try.add(
            tf.keras.layers.Dense(units=32, activation=tf.nn.swish, kernel_regularizer=tf.keras.regularizers.l2(0.1)))
        model_try.add(tf.keras.layers.Dense(units=self.OutShape))
        model_try.compile(tf.keras.optimizers.RMSprop(0.001)  # 优化方法Nadam,adam,sgd随机,=tf.keras.optimizers.Adam(0.001),
                          , loss=self.loss_name  # mse,mae,kld,Huber,mean_squared_error,myloss
                          , metrics=self.metrics_func  # valloss#计算正确率，RootMeanSquaredError，MAPE，MeanSquaredError
                          )
        model_try.summary()

        return model_try

    pass

    def build_model(self, hp):
        model_try = tf.keras.Sequential()
        model_try.add(tf.keras.layers.Input(shape=self.inShape))
        # dulstmif = hp.Choice('使用双向LSTM', values=[0, 1])
        for lst in range(0, hp.Int('LSTM循环', min_value=1, max_value=4, step=1)):
            lstm = tf.keras.layers.LSTM(  # LSTM,GRU
                units=hp.Int('LSTM层_{}'.format(lst),
                             min_value=32,
                             max_value=256,
                             step=32), return_sequences=True)
            # if dulstmif == 1:
            #     lstm = tf.keras.layers.Bidirectional(lstm)
            model_try.add(lstm)
            if hp.Choice('是否丢掉_{}'.format(lst), values=[0, 1]) == 1:
                model_try.add(
                    tf.keras.layers.Dropout(hp.Float('丢掉层_{}'.format(lst), min_value=0.1, max_value=0.3, step=0.1)))
                pass

        for des in range(0, hp.Int('全链循环',
                                   min_value=0,
                                   max_value=3,
                                   step=1)):
            if hp.Choice('使用正则_{}'.format(des), values=[0, 1]) == 1:
                model_try.add(tf.keras.layers.Dense(units=hp.Int('全链层_正则_{}'.format(des),
                                                                 min_value=32,
                                                                 max_value=128,
                                                                 step=32),
                                                    activation=tf.nn.swish,
                                                    kernel_regularizer=tf.keras.regularizers.l2(
                                                        hp.Choice('正则_{}'.format(des),
                                                                  values=[0.004, 0.01, 0.1]))))
            else:
                model_try.add(tf.keras.layers.Dense(units=hp.Int('全链层_{}'.format(des),
                                                                 min_value=32,
                                                                 max_value=128,
                                                                 step=32),
                                                    activation=tf.nn.swish))

        model_try.add(tf.keras.layers.Dense(self.OutShape, activation=tf.nn.swish))  # tf.nn.softmax用于区分是否的概率激活
        model_try.compile(optimizer=tf.keras.optimizers.Adam(0.001)  # 优化方法Nadam,adam,sgd随机
                          , loss=self.loss_name
                          # mse,mae,kld,Huber,mean_squared_error,myloss,多分类sparse_categorical_crossentropy，categorical_crossentropy(onehot),相似度cosine_similarity
                          , metrics=self.metrics_func
                          # valloss#计算正确率，RootMeanSquaredError，MAPE，MeanSquaredError,['acc'],['accuracy'],CategoricalAccuracy(onehot)
                          )
        return model_try

    def build_model_rnn(self, hp):
        model_try = tf.keras.Sequential()
        model_try.add(tf.keras.layers.Input(shape=self.inShape))
        # dulstmif = hp.Choice('使用双向LSTM', values=[0, 1])
        for lst in range(0, hp.Int('Rnn循环', min_value=1, max_value=4, step=1)):
            lstm = tf.keras.layers.Dense(  # LSTM,GRU
                units=hp.Int('Rnn层_{}'.format(lst),
                             min_value=32,
                             max_value=256,
                             step=32), activation=tf.nn.swish)
            # if dulstmif == 1:
            #     lstm = tf.keras.layers.Bidirectional(lstm)
            model_try.add(lstm)
            if hp.Choice('是否丢掉_{}'.format(lst), values=[0, 1]) == 1:
                model_try.add(
                    tf.keras.layers.Dropout(hp.Float('丢掉层_{}'.format(lst), min_value=0.1, max_value=0.3, step=0.1)))
                pass

        for des in range(0, hp.Int('全链循环',
                                   min_value=0,
                                   max_value=3,
                                   step=1)):
            if hp.Choice('使用正则_{}'.format(des), values=[0, 1]) == 1:
                model_try.add(tf.keras.layers.Dense(units=hp.Int('全链层_正则_{}'.format(des),
                                                                 min_value=32,
                                                                 max_value=128,
                                                                 step=32),
                                                    activation=tf.nn.swish,
                                                    kernel_regularizer=tf.keras.regularizers.l2(
                                                        hp.Choice('正则_{}'.format(des),
                                                                  values=[0.001, 0.005, 0.01, 0.1]))))
            else:
                model_try.add(tf.keras.layers.Dense(units=hp.Int('全链层_{}'.format(des),
                                                                 min_value=32,
                                                                 max_value=128,
                                                                 step=32),
                                                    activation=tf.nn.swish))

        model_try.add(tf.keras.layers.Dense(self.OutShape, activation=tf.nn.swish))  # tf.nn.softmax用于区分是否的概率激活
        model_try.compile(optimizer=tf.keras.optimizers.Adam(0.001)  # 优化方法Nadam,adam,sgd随机
                          , loss=self.loss_name
                          # mse,mae,kld,Huber,mean_squared_error,myloss,多分类sparse_categorical_crossentropy，categorical_crossentropy(onehot),相似度cosine_similarity
                          , metrics=self.metrics_func
                          # valloss#计算正确率，RootMeanSquaredError，MAPE，MeanSquaredError,['acc'],['accuracy'],CategoricalAccuracy(onehot)
                          )
        return model_try

    def SearchModel(self):
        monitor = 'val_' + self.dset.metrics_name
        mode = self.dset.valLossSaveMode
        if (mode == '' or mode is None) and (
                self.dset.metrics_name == "mape" or self.dset.metrics_name == "cailoss" or self.dset.metrics_name == "myloss"):
            mode = 'min'
        savefile = self.dset.SearchModelGuid
        dose = self.dset.SearchModeDo
        if savefile == "":
            dose = True
            savefile = self.dset.guid
        savefile = self.dset.SearchModelfile + savefile

        modelType = "AutoLSTM"
        if self.dset.tfModelType == "rnn_a":
            modelType = "AutoRnn"

        bmodel = self.build_model
        if modelType == "AutoRnn":
            bmodel = self.build_model_rnn

        tuner = RandomSearch(
            bmodel,
            objective=Objective(monitor, direction=mode),
            # min,max优化目标为精度val_loss，val_valloss，评估值参_val
            max_trials=self.dset.SearchModelMaxTry,  # 总共试验10次
            executions_per_trial=self.dset.SearchModelPerTry,  # 每次试验训练模型三次
            directory=savefile,
            project_name=modelType)

        if dose:
            tuner.search_space_summary()
            tuner.search(self.DDD.DsFix_In, self.DDD.DsFix_R, validation_split=self.dset.SearchModelFitSplit,
                         epochs=self.dset.SearchModelEpochs)
            tuner.results_summary()
            pass
        bestpara = tuner.get_best_hyperparameters(1)[0]
        print('最好参数----\r\n{}'.format(bestpara.values))
        model = tuner.hypermodel.build(bestpara)
        model.summary()
        return model
        pass
    def GetModel(self, modelName) -> tf.keras.Sequential:
        model = []
        if modelName == "lstm":
            model = self.LSTM()
        if modelName == "lstm_gpt":
            model = self.LSTM_GPT()
        if modelName in ("lstm_a", "rnn_a"):
            model = self.SearchModel()
        if modelName == "rnn":
            model = self.RNN()
        if modelName == "lstm_f":  # 分类
            model = self.LSTM_F()
        if modelName == "lstm_l":  # 回归
            model = self.LSTM_F(False)
        if modelName == "lstm_ll":  # 线性回归
            model = self.LSTM_F(False, 'linear')
        if modelName == 'lstm_f_auto':  # 分类自动
            model = self.LSTM_FA()
        if modelName == 'lstm_l_auto':  # 回归自动
            model = self.LSTM_FA(False)
        if modelName == 'f_cai_hierarchical_old':  # 旧版分层Transformer彩票分类预测
            model = self.HierarchicalCaiTransformerOld()
        if modelName == 'l_cai_hierarchical_old':  # 旧版分层Transformer彩票回归预测
            model = self.HierarchicalCaiTransformerOld(False)
        if modelName == 'f_cai_hierarchical':  # 新版分层Transformer彩票分类预测
            model = self.HierarchicalCaiTransformer()
        if modelName == 'l_cai_hierarchical':  # 新版分层Transformer彩票回归预测
            model = self.HierarchicalCaiTransformer(False)
        if modelName == 'f_cai_hierarchical_l2':  # 新版分层Transformer彩票分类预测
            model = self.HierarchicalCaiTransformerL2()
        if modelName == 'l_cai_hierarchical_l2':  # 新版分层Transformer彩票回归预测
            model = self.HierarchicalCaiTransformerL2(False)
        return model
        pass
