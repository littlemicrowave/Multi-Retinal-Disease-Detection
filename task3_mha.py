'''
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, word_vector_len, heads):
        super().__init__()
        self.word_vector_len = word_vector_len
        self.heads = heads
        if word_vector_len % heads != 0:
            raise "Word vector should be divisible by number of heads"
        self.head_depth = int(word_vector_len / heads)

        #Initializing Dense layers for queries, keys, values and output
        self.wq = tf.keras.layers.Dense(word_vector_len)
        self.wk = tf.keras.layers.Dense(word_vector_len)
        self.wv = tf.keras.layers.Dense(word_vector_len)
        
        self.wo = tf.keras.layers.Dense(word_vector_len)

    def calculateAttention(self, batch_size, sentence_len, q, k, v, mask = None):
        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_depth, dtype=tf.float32)) # (batch_size, heads, sentence_len, sentence_len)
        if mask is not None:
            attention_scores += (mask * -1e9) #masking for decoder
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)  
        attention_output = tf.matmul(attention_scores, v) # (batch_size, heads, sentence_len, head_depth)

        #re-arraning and combining heads back
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])  # (batch, seq_len, heads, head_depth)
        attention_output = tf.reshape(attention_output, [batch_size, sentence_len, self.word_vector_len])

        #regualizer
        attention_output = tf.keras.layers.Dropout(0.2)(attention_output)

        return attention_output
    
    def call(self, q, k, v, mask = None):
        batch_size = tf.shape(q)[0]
        sentence_len = tf.shape(q)[1]
        queries = self.wq(q)
        keys = self.wk(k)
        values = self.wv(v)

        #slicing word vectors into pieces of size head_depth, amount of such pieces is equal to amount of heads. That's why it is neccessary that word_vecor_len is divisible by amount of heads
        #rearraging our matrix to shape so it would be [batch_size, heads, sentence length, head depth]

        queries = tf.reshape(queries, [batch_size, sentence_len, self.heads, self.head_depth])
        queries = tf.transpose(queries, perm = [0, 2, 1, 3]) # (batch_size, heads, sentence_len, head_depth)

        keys = tf.reshape(keys, [batch_size, sentence_len, self.heads, self.head_depth])
        keys= tf.transpose(keys, perm = [0, 2, 1, 3])

        values = tf.reshape(values, [batch_size, sentence_len, self.heads, self.head_depth])
        values = tf.transpose(values, perm = [0, 2, 1, 3])

        attention_output = self.calculateAttention(batch_size, sentence_len, queries, keys, values, mask)
        output = self.wo(attention_output)

        return output
'''
import torch
import torch.nn as nn


class MultiHeadAttentionCNN(nn.Module):
    def __init__(self, input_dim, num_heads, mask=None):
        super().__init__()
        if input_dim % num_heads != 0:
            raise "Projection number of dimensions is indisible by number of heads!"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_depth = input_dim // num_heads
        self.mask = mask
        self.attn_denum = self.head_depth ** (-0.5)

        self.wq = nn.Linear(input_dim, input_dim)
        self.wk = nn.Linear(input_dim, input_dim)
        self.wv = nn.Linear(input_dim, input_dim)
        self.wo = nn.Linear(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)

    def attention(self, q, k, v):
        # q, k, v: (b, heads, seq_len, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.attn_denum

        if self.mask is not None:
            scores = torch.masked_fill(scores, self.mask, -torch.inf)

        scores = self.softmax(scores)
        return torch.matmul(scores, v)

    def forward(self, x):
        # x: (b, c, h, w)
        b, c, h, w = x.shape
        seq_len = h * w

        #flatten spatial dimensions
        x = torch.reshape(x, [b, seq_len, c])

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        #split heads
        q = torch.reshape(q, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)
        k = torch.reshape(k, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)
        v = torch.reshape(v, [b, seq_len, self.num_heads, self.head_depth]).transpose(1, 2)

        #attention compute
        out = self.attention(q, k, v)
        #merge heads
        out = out.transpose(1, 2)
        out = out.reshape(b, seq_len, c)
        #output projection
        out = self.wo(out)
        #rstores dimensions
        out = torch.reshape(out, [b, h, w, c]).permute(0, 3, 1, 2)

        return out

