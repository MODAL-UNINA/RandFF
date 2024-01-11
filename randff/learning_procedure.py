#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:38:41 2023

@author: MODAL
"""

# %%
import tensorflow as tf
from .utils import accuracy

def train(network, optimizer, linear_cf, optimizer_cf, train_loader, start_block, opts, idxs, block_choice):
    running_loss = tf.constant(0.)
    running_ce = tf.constant(0.)
    first_block_loss = tf.constant(0.)

    for i, (x, y_pos) in enumerate(train_loader):
        if opts.training_mode == 'dense':
            x = tf.reshape(x, [x.shape[0], -1])
            
        # elif opts.training_mode == 'receptive_field':
        elif opts.training_mode in ['receptive_field', 'mixed']:
            if len(x.shape)==3:
                x = tf.expand_dims(x, axis=-1)

        # Convert y_pos in one hot encoded vector and concat to x
        y_pos_one_hot = tf.one_hot(y_pos, opts.n_classes)

        if opts.training_mode == 'dense':
            x_pos = tf.concat([x, y_pos_one_hot], axis=1)
            # Create the negative samples
            x_neg = tf.concat([x, tf.ones_like(y_pos_one_hot)*(1/opts.n_classes)], axis=1)
   
        # elif opts.training_mode == 'receptive_field':
        elif opts.training_mode in ['receptive_field', 'mixed']:
            x_pos = tf.Variable(x, trainable=False)
            x_pos = tf.constant(x_pos[:, :opts.n_classes, 0, :].assign(tf.stack([y_pos_one_hot]*x.shape[-1], axis=-1)))
            # Create the negative samples
            x_neg = tf.Variable(x, trainable=False)
            x_neg = tf.constant(x_neg[:, :opts.n_classes, 0, :].assign(tf.stack([tf.ones_like(y_pos_one_hot)*(1/opts.n_classes)]*x.shape[-1], axis=-1)))
            
        # Compute predictions of the network
        ys = network(x_neg, cat=False)
        ys = tf.concat([tf.reshape(k, [k.shape[0], -1]) for k in ys], axis=-1)
        
        # Compute cross-entropy loss
        with tf.GradientTape() as tape:
            logits = linear_cf(ys)
            ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_pos, logits)
            running_ce += ce
        
        # Compute the gradient of the cross-entropy loss
        grads = tape.gradient(ce, linear_cf.trainable_variables)
        optimizer_cf.apply_gradients(zip(grads, linear_cf.trainable_variables))

        # Negative pairs from softmax layer
        probs = tf.nn.softmax(logits, axis=1)
        preds = tf.cast(tf.argmax(probs, axis=1), dtype=tf.uint8)
        idx = tf.where(preds != y_pos)

        # Negative pairs from random labels
        y_rand = tf.cast(tf.random.uniform([x.shape[0]], minval=0, maxval=opts.n_classes, dtype=tf.int32), dtype=tf.uint8)
        idx = tf.where(y_rand != y_pos) # incorrect labels
        y_rand_one_hot = tf.one_hot(y_rand, opts.n_classes)

        if opts.training_mode == 'dense':
            x_rand = tf.concat([x, y_rand_one_hot], axis=1) #[idx] # keeping positives seems to work better
            x_neg = x_rand

        # elif opts.training_mode == 'receptive_field':
        elif opts.training_mode in ['receptive_field', 'mixed']:
            x_rand = tf.Variable(x, trainable=False)
            x_rand = tf.constant(x_rand[:, :opts.n_classes, 0, :].assign(tf.stack([y_rand_one_hot]*x.shape[-1], axis=-1)))
            # x_rand = tf.gather(x_rand, idx[:,0]) # keeping positives seems to work better
            x_neg = x_rand

        if opts.hard_negatives:
            y_hard_one_hot = tf.one_hot(preds, opts.n_classes)
            if opts.training_mode == 'dense':
                # taking the wrong predicted samples
                x_hard = tf.gather(tf.concat([x, y_hard_one_hot], axis=1), idx[:,0])

            # elif opts.training_mode == 'receptive_field':
            elif opts.training_mode in ['receptive_field', 'mixed']:
                # taking the wrong predicted samples
                x_hard = tf.Variable(x, trainable=False)
                x_hard = tf.constant(x_hard[:, :opts.n_classes, 0, :].assign(tf.stack([y_hard_one_hot]*x.shape[-1], axis=-1)))
                x_hard = tf.gather(x_hard, idx[:,0])

            x_neg = tf.concat([x_neg, x_hard], axis=0)       

        # Perform the optimizations
        running_loss_partial, first_block_loss_partial = train_step(network, idxs, x_pos, x_neg, optimizer, start_block, block_choice, opts)
        running_loss += running_loss_partial
        first_block_loss += first_block_loss_partial

    running_loss /= tf.cast(len(train_loader), dtype=tf.float32)
    running_ce /= tf.cast(len(train_loader), dtype=tf.float32)
    first_block_loss /= tf.cast(len(train_loader), dtype=tf.float32)

    return running_loss, running_ce, first_block_loss

def compute_loss_norm(tensor, theta, s=1.):
    return tf.reduce_mean(tf.math.log(1 + tf.math.exp(s*(-tf.norm(tf.reshape(tensor, [tensor.shape[0], -1]), axis=-1) + theta))))


def train_step(network, idxs, x_pos, x_neg, optimizer, start_block, block_choice, opts):
    acc_loss = tf.constant(0.)

    # Evaluate the Gradients and perform the local BackPropagation
    with tf.GradientTape() as tape:
        z_pos = network(x_pos, cat=False)
        z_neg = network(x_neg, cat=False)

        if tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(k)).numpy() for k in z_pos]) or tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(k)).numpy() for k in z_neg]):
            print(tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(k)).numpy() for k in z_pos]),
                    tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(k)).numpy() for k in z_neg]))
            raise ValueError("NAN")
        
        losses = []

        for l, idx in enumerate(idxs[1:]):
            
            zp = z_pos[idx-1]; zn = z_neg[idx-1]
            positive_loss = compute_loss_norm(zp, opts.theta, 1.)
            negative_loss = compute_loss_norm(zn, opts.theta, -1.)

            # positive_loss_test = tf.reduce_mean(tf.math.log(1 + tf.math.exp((-tf.norm(zp, axis=-1) + opts.theta))))
            # negative_loss_test = tf.reduce_mean(tf.math.log(1 + tf.math.exp((tf.norm(zn, axis=-1) - opts.theta))))

            loss = (positive_loss + negative_loss)
            if l==start_block:
                first_block_loss = loss

            acc_loss += loss

            #DEBUG
            # if any element of zp or zn is nan print the tensor
            # if tf.math.reduce_any(tf.math.is_nan(zp)) or tf.math.reduce_any(tf.math.is_nan(zn)):
            #     print("NAN")
            #     print(zp)
            #     print(zn)
            #     raise ValueError("NAN")

            if l in block_choice:
                losses.append(loss)
        

    grads = tape.gradient(losses, [network.trainable_variables[idxs[i-1]:idxs[i]] for i in range(1,len(idxs)) if i-1 in block_choice])

    [opt.apply_gradients(zip(grad, var)) 
            for grad, var, opt in 
                    zip(grads, 
                       [network.trainable_variables[idxs[i-1]:idxs[i]] for i in range(1,len(idxs)) if i-1 in block_choice],
                       [o for j, o in enumerate(optimizer) if j in block_choice])
    ]
    
    return acc_loss, first_block_loss


def test(network_ff, linear_cf, test_dataset, opts):
    # TODO: check if modifications are correct
    all_outputs = []
    all_labels = []
    all_logits = []
    
    for (x_test, y_test) in test_dataset:
        # # x_test, y_test = x_test.to(opts.device), y_test.to(opts.device)
        # x_test = tf.reshape(x_test, [x_test.shape[0], -1])

        # acts_for_labels = []

        # # Slow method
        # for label in range(10):
        #     test_label = tf.ones_like(y_test, dtype=tf.uint8) * label
        #     # test_label = norm_y(tf.one_hot(test_label, depth=10))
        #     test_label = tf.one_hot(test_label, depth=10)
        #     x_with_labels = tf.concat([x_test, test_label], axis=1)
            
        #     acts = network_ff(x_with_labels)
        #     acts = tf.norm(acts, ord='euclidean', axis=-1)
        #     acts_for_labels.append(acts)
        
        # # These are logits
        # acts_for_labels = tf.stack(acts_for_labels, axis=1) #should be BSZxLABELSxLAYERS (10)

        if opts.training_mode == 'dense':
            x_test = tf.reshape(x_test, [x_test.shape[0], -1])
        # elif opts.training_mode == 'receptive_field':
        elif opts.training_mode in ['receptive_field', 'mixed']:
            if len(x_test.shape)==3:
                x_test = tf.expand_dims(x_test, axis=-1)
            else:
                x_test = x_test
            
        acts_for_labels = []
        for label in range(opts.n_classes):
            test_label = tf.ones(y_test.shape, dtype=tf.uint8) * label 
            test_label = tf.one_hot(test_label, depth=opts.n_classes)
            
            if opts.training_mode == 'dense':
                x_with_labels = tf.concat([x_test, test_label], axis=1)

            # elif opts.training_mode == 'receptive_field':
            elif opts.training_mode in ['receptive_field', 'mixed']:
                x_with_labels = tf.Variable(x_test)
                x_with_labels = tf.constant(x_with_labels[:, :opts.n_classes, 0, :].assign(tf.stack([test_label]*x_test.shape[-1], axis=-1)))

            acts = network_ff(x_with_labels, cat=False)[1:]
            acts = tf.stack(
                [tf.norm(tf.reshape(tensor, [tensor.shape[0], -1]), ord='euclidean', axis=-1) for tensor in acts], axis=1
                )
            acts_for_labels.append(acts)

        acts_for_labels = tf.stack(acts_for_labels, axis=1)

        all_outputs.append(acts_for_labels)
        all_labels.append(y_test)

        # Quick method
        neutral_label = tf.ones([x_test.shape[0], opts.n_classes], dtype=tf.float32) * 0.1
        # acts = network_ff(tf.concat([x_test, neutral_label], axis=1))

        if opts.training_mode == 'dense':
            x_neutral = tf.concat([x_test, neutral_label], axis=1)
   
        # elif opts.training_mode == 'receptive_field':
        elif opts.training_mode in ['receptive_field', 'mixed']:
            x_neutral = tf.Variable(x_test, trainable=False)
            x_neutral = tf.constant(x_neutral[:, :opts.n_classes, 0, :].assign(tf.stack([neutral_label]*x_test.shape[-1], axis=-1)))
            
        # Compute predictions of the network
        acts = network_ff(x_neutral, cat=False)
        acts = tf.concat([tf.reshape(k, [k.shape[0], -1]) for k in acts], axis=-1)
        logits = linear_cf(tf.reshape(acts, [acts.shape[0], -1]))
        all_logits.append(logits)

    all_outputs = tf.concat(all_outputs, axis=0)
    all_labels = tf.concat(all_labels, axis=0)
    all_logits = tf.concat(all_logits, axis=0)

    slow_acc = accuracy(tf.reduce_mean(all_outputs, axis=-1), all_labels, topk=(1,))[0]
    fast_acc = accuracy(all_logits, all_labels, topk=(1,))[0]

    return slow_acc, fast_acc

def predict(x_test, y_test, net_ff, idxs, opts, return_acc=False):

    if opts.training_mode == 'dense':
        sample = tf.reshape(x_test, [x_test.shape[0], -1])
    # elif opts.training_mode == 'receptive_field':
    elif opts.training_mode in ['receptive_field', 'mixed']:
        if len(x_test.shape)==3:
            sample = tf.expand_dims(x_test, axis=-1)
        else:
            sample = x_test
        
    acts_for_labels = []
    for label in range(opts.n_classes):
        test_label = tf.ones(y_test.shape, dtype=tf.uint8) * label 
        test_label = tf.one_hot(test_label, depth=opts.n_classes)
        
        if opts.training_mode == 'dense':
            x_with_labels = tf.concat([sample, test_label], axis=1)

        # elif opts.training_mode == 'receptive_field':
        elif opts.training_mode in ['receptive_field', 'mixed']:
            x_with_labels = tf.Variable(sample)
            x_with_labels = tf.constant(x_with_labels[:, :opts.n_classes, 0, :].assign(tf.stack([test_label]*sample.shape[-1], axis=-1)))

        acts = net_ff(x_with_labels, cat=False)[1:]
        acts = tf.stack(
            [tf.norm(tf.reshape(tensor, [tensor.shape[0], -1]), ord='euclidean', axis=-1) for tensor in acts], axis=1
            )
        acts_for_labels.append(acts)

    acts_for_labels = tf.stack(acts_for_labels, axis=1)    
    preds = tf.argmax(tf.reduce_mean(acts_for_labels, axis=-1), axis=1)

    #Compute accuracy between preds and y_test
    if return_acc:
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_test), dtype=tf.float32))
        return preds.numpy(), acc
    
    return preds.numpy()
