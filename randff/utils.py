import tensorflow as tf
import random
import numpy as np
import os

def set_seed(seed, CVD):
    # Set seed for reproducibility and set the GPU to use (-1 for CPU)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CVD)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def accuracy(output, target, topk=(1,)):
    # Computes the accuracy over the k top predictions for the specified values of k
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = tf.math.top_k(output, maxk, sorted=True)
    pred = tf.cast(tf.transpose(pred), tf.uint8)
    correct = tf.math.equal(pred, tf.expand_dims(target,0))

    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], -1), tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        res.append(correct_k * (100.0 / batch_size))
    return res

def p_grads_probs(xf, deg=5, scale=1.5, distribution='uniform'):
    # Computes the probability for the training phase
    if distribution == 'uniform':
        import numpy as np
        return np.ones(xf)/xf
        
    elif distribution == 'chi2':
        import scipy.stats as stats
        x = [(i/xf)*10 for i in list(range(xf))]
        p_grads = stats.chi2.pdf(x, df=deg, scale=scale)
        return p_grads / sum(p_grads)