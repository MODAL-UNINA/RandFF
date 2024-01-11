import tensorflow as tf

#%% DENSE NETWORK
class Block(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, normalize_input=True, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(out_dim, input_dim=in_dim, use_bias=bias)
        self.relu = tf.keras.layers.ReLU(True)
        self.normalize_input = normalize_input

    def call(self, x):
        if self.normalize_input:
            x = tf.math.l2_normalize(x, axis=1)
        x = self.fc(x)
        self.x = x

        return self.relu(x)

class Network(tf.keras.Model):
    def __init__(self, dims, **kwargs):
        super().__init__(**kwargs)

        blocks = []
        blocks.append(Block(dims[0], dims[1], normalize_input=False))
        for i in range(len(dims[1:-1])):
            blocks.append(Block(dims[i+1], dims[i+2], normalize_input=True))
        
        # just for print
        self.blocks = tf.keras.Sequential(blocks)
        self.n_blocks = len(blocks)

    def call(self, x, cat=True):
        x = self.blocks(x)
        
        xs = [b.x for b in self.blocks.layers]

        if not cat:
            return xs
        return tf.stack(xs, axis=1)
    
#%% RECEPTIVE FIELD NETWORK
class BlockRF(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, normalize_input=True, input_shape=None, bias=False, **kwargs):
        super().__init__(**kwargs)
        if input_shape is not None:
            self.fc = tf.keras.layers.LocallyConnected2D(filters, kernel_size, padding='valid',
                                                          use_bias=bias, input_shape=input_shape, strides=1)#, implementation=1)
        else:
            self.fc = tf.keras.layers.LocallyConnected2D(filters, kernel_size, padding='valid',
                                                          use_bias=bias, strides=1)#, implementation=1)
        self.padding = tf.keras.layers.ZeroPadding2D(padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.relu = tf.keras.layers.ReLU(True)
        self.normalize_input = normalize_input

    def call(self, x):
        if self.normalize_input:
            # x = tf.math.l2_normalize(x, axis=1)
            x = tf.keras.layers.BatchNormalization()(x)
        x = self.fc(x)
        # x = self.padding(x)
        self.x = x

        return self.relu(x)

class NetworkRF(tf.keras.Model):
    def __init__(self, dims, input_shape, **kwargs):
        super().__init__(**kwargs)

        blocks = []
        blocks.append(BlockRF(dims[0][0], dims[0][1], normalize_input=False, input_shape=input_shape))
        for f,k in dims[1:]:
            blocks.append(BlockRF(f, k, normalize_input=True))
        
        # just for print
        self.blocks = tf.keras.Sequential(blocks)
        self.n_blocks = len(blocks)

    def call(self, x, cat=True):
        x = self.blocks(x)
        
        xs = [b.x for b in self.blocks.layers]

        if not cat:
            return xs
        return tf.stack(xs, axis=1)

#%% MIXED
#%% MIXED NETWORK
class BlockDense(tf.keras.layers.Layer):
    def __init__(self, out_dim, normalize_input=None, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(out_dim, use_bias=bias)
        self.relu = tf.keras.layers.ReLU(True)
        self.normalize_input = normalize_input

    def call(self, x):
        if len(x.shape) > 2:
            x = tf.reshape(x, [x.shape[0], -1])

        if self.normalize_input:
            x = tf.math.l2_normalize(x, axis=1)
        x = self.fc(x)
        self.x = x
        return self.relu(x)
class NetworkMixed(tf.keras.Model):
    def __init__(self, dims, input_shape, **kwargs):
        super().__init__(**kwargs)

        blocks = []
        blocks.append(BlockRF(dims[0][0], dims[0][1], normalize_input=False, input_shape=input_shape))
        for _, _, d in dims[1:]:
            blocks.append(BlockDense(d, normalize_input=True))
        
        # just for print
        self.blocks = tf.keras.Sequential(blocks)
        self.n_blocks = len(blocks)

    def call(self, x, cat=True):
        x = self.blocks(x)
        
        xs = [b.x for b in self.blocks.layers]

        if not cat:
            return xs
        return tf.stack(xs, axis=1)