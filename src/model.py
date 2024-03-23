import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import gelu, softmax
from tensorflow.keras.models import Sequential

class MHA(Layer):
    '''
    Multi-Head Attention Layer
    '''
    
    def __init__(self, num_head, dropout = 0):
        super(MHA, self).__init__()
        
        # Constants
        self.num_head = num_head
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        query_shape = input_shape
        d_model = query_shape[-1]
        units = d_model // self.num_head
        
        # Loop for Generate each Attention
        self.layer_q = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build(query_shape)
            self.layer_q.append(layer)
            
        self.layer_k = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build(query_shape)
            self.layer_k.append(layer)
            
        self.layer_v = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build(query_shape)
            self.layer_v.append(layer)
            
        self.out = Dense(d_model, activation = None, use_bias = False)
        self.out.build(query_shape)
        self.dropout = Dropout(self.dropout_rate)
        self.dropout.build(query_shape)
        
    def call(self, x):
        d_model = x.shape[-1]
        scale = d_model ** -0.5
        
        attention_values = []
        for i in range(self.num_head):
            attention_score = softmax(tf.matmul(self.layer_q[i](x), self.layer_k[i](x), transpose_b=True) * scale)
            attention_final = tf.matmul(attention_score, self.layer_v[i](x))
            attention_values.append(attention_final)
            
        attention_concat = tf.concat(attention_values, axis = -1)
        out = self.out(self.dropout(attention_concat))
        
        return out

class IMHA(Layer):
    '''
    Intersample Multi Head Attention
    Attend on row(samples) not column(features)
    '''
    
    def __init__(self, num_head, dropout = 0):
        super(IMHA, self).__init__()
        
        # Constants
        self.num_head = num_head
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        b, n, d = input_shape
        query_shape = input_shape
        units = (d * n) // self.num_head
        # Loop for Generate each Attention
        self.layer_q = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build([1, b, int(n * d)])
            self.layer_q.append(layer)
            
        self.layer_k = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build([1, b, int(n * d)])
            self.layer_k.append(layer)
            
        self.layer_v = []
        for _ in range(self.num_head):
            layer = Dense(units, activation = None, use_bias = False)
            layer.build([1, b, int(n * d)])
            self.layer_v.append(layer)
            
        self.out = Dense(d, activation = None, use_bias = False)
        self.out.build(query_shape)
        self.dropout = Dropout(self.dropout_rate)
        self.dropout.build(query_shape)
        
    def call(self, x):
        b, n, d = x.shape
        scale = d ** -0.5
        x = tf.reshape(x, (1, b, int(n * d)))
        attention_values = []
        
        for i in range(self.num_head):
            attention_score = softmax(tf.matmul(self.layer_q[i](x), self.layer_k[i](x), transpose_b=True) * scale)
            attention_final = tf.matmul(attention_score, self.layer_v[i](x))
            attention_final = tf.reshape(attention_final, (b, n, int(d / self.num_head)))
            attention_values.append(attention_final)
            
        attention_concat = tf.concat(attention_values, axis = -1)
        out = self.out(self.dropout(attention_concat))
        
        return out

class FeedForwardNetwork(Layer):
    def __init__(self, dim, dropout = 0.0):
        super(FeedForwardNetwork, self).__init__()
        self.dense = Dense(dim, activation = 'gelu')
        self.dropout = Dropout(dropout)
        
    def call(self, x):
        return self.dropout(self.dense(x))

class CustomEmbedding(Layer):
    def __init__(self, stock_dim, time_dim, dim):
        super(CustomEmbedding, self).__init__()
        self.stock_dim = stock_dim
        self.time_dim = time_dim
        self.dim = dim
        
    def build(self, input_shape):
        b, n = input_shape
        self.embedding_categorical_stock = Embedding(self.stock_dim, self.dim)
        self.embedding_categorical_time = Embedding(self.time_dim, self.dim)

        self.embedding_categorical_stock.build([b, 1])
        self.embedding_categorical_time.build([b, 1])
        
        self.embedding_numerical = Dense(self.dim, activation = 'relu')
        self.embedding_numerical.build([b, int(n - 2), 1])
        
    def call(self, x):
        b, n = x.shape
        categorical_x_stock = x[:, :1]
        categorical_x_time = x[:, 1:2]
        numerical_x = x[:, 2:]
        numerical_x = tf.reshape(numerical_x, (b, int(n - 2), 1))
        
        embedded_cat_stock = self.embedding_categorical_stock(categorical_x_stock)
        embedded_cat_time = self.embedding_categorical_time(categorical_x_time)
        embedded_num = self.embedding_numerical(numerical_x)
    
        embedded_x = tf.concat([embedded_cat_stock, embedded_cat_time, embedded_num], axis = 1)
        
        return embedded_x


class SAINT(Layer):
    def __init__(self, repeat, stock_dim, time_dim, EMB_DIM, MHA_HEADS, IMHA_HEADS):
        super(SAINT, self).__init__()
        self.repeat = repeat
        self.layer_mha = []
        self.layer_imha = []
        self.layer_ffn = []
        self.layer_layernorm = []
        self.embedding = CustomEmbedding(stock_dim, time_dim, EMB_DIM)
        
        for _ in range(repeat):
            mha = MHA(MHA_HEADS)
            imha = IMHA(IMHA_HEADS)
            ffn_1 = FeedForwardNetwork(EMB_DIM)
            ffn_2 = FeedForwardNetwork(EMB_DIM)
            layernorm_1 = LayerNormalization()
            layernorm_2 = LayerNormalization()
            layernorm_3 = LayerNormalization()
            layernorm_4 = LayerNormalization()
            
            self.layer_mha.append(mha)
            self.layer_imha.append(imha)
            self.layer_ffn.append(ffn_1)
            self.layer_ffn.append(ffn_2)
            self.layer_layernorm.append(layernorm_1)
            self.layer_layernorm.append(layernorm_2)
            self.layer_layernorm.append(layernorm_3)
            self.layer_layernorm.append(layernorm_4)
            
    def call(self, x):
        x = self.embedding(x)
        # Depth of SAINT Layer
        for i in range(self.repeat):
            # Multi-Head part
            x = self.layer_layernorm[i](self.layer_mha[i](x)) + x
            x = self.layer_layernorm[i+1](self.layer_ffn[i](x)) + x
            
            # Intersample Multi-Head part
            x = self.layer_layernorm[i+2](self.layer_imha[i](x)) + x
            x = self.layer_layernorm[i+3](self.layer_ffn[i+1](x)) + x
       
        # only using cls token for final output
        out = x[:, 0] # CLS Token
        
        return out

    
