#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:39:38 2023

@author: MODAL
"""
#%% IMPORT SECTION
import random
import numpy as np
from randff import set_seed, p_grads_probs

import matplotlib.pyplot as plt

# %% PRELIMINARY DEFINITIONS
from fastcore.all import dict2obj
opts = dict2obj(dict(
                hard_negatives = True,
                n_layers = 15,
                layer_size = (28*28)//2,
                n_blocks = 5,

                batch_size = 3000,
                lr = 1e-3,#5e-4,
                epochs = 2000,
                initial_epochs = 25,
                changing_epochs = 25,
                deg_patience = 15,
                
                deg = 4,
                theta = 10.,
                layer_threshold = 1e-2,
                
                seed = 0,
                dataset = 'cifar10',
                training_mode = 'mixed', # 'dense' or 'receptive_field' or 'mixed'
                receptive_field = (7,7),
                ))

#%% FORWARD FORWAR PROCEDURE
import tensorflow as tf
from randff import set_seed, train, predict

def main(opts):
    # Take the mnist dataset from tensorflow
    if opts.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    elif opts.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    elif opts.dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)

        opts.layer_size = (tf.reduce_prod(x_train[0].shape))//2
        opts.initial_epochs = 50; opts.changing_epochs = 50
        opts.layer_threshold = 1e-3

    opts.n_classes = len(tf.unique(y_train)[0])
    
    # Normalize the dataset
    x_train = tf.cast(x_train, tf.float32) / 255.
    x_test = tf.cast(x_test, tf.float32) / 255.

    # Split in batches
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(opts.batch_size)
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(opts.batch_size) # not used since the FAST and SLOW accuracies are not computed
    
    # Create the model and the needed optimizers
    if opts.training_mode == 'dense':
        from randff import Network
        net_ff = Network(dims=[(tf.reduce_prod(x_train[0].shape))+opts.n_classes, *[opts.layer_size]*opts.n_layers])

    elif opts.training_mode == 'receptive_field':
        from randff import NetworkRF
        net_ff = NetworkRF(dims=[(x_train[0].shape[-1], opts.receptive_field)]*opts.n_layers, input_shape=x_train[0].shape)

    elif opts.training_mode == 'mixed':
        from randff import NetworkMixed
        net_ff = NetworkMixed(dims=[(x_train[0].shape[-1], opts.receptive_field, opts.layer_size)]*opts.n_layers, input_shape=x_train[0].shape)

    optimizers = [tf.optimizers.Adam(opts.lr) for _ in net_ff.blocks.layers]

    # Softmax layer for predicting classes from embeddings (fast method)
    linear_cf = tf.keras.layers.Dense(opts.n_classes)
    optimizer_cf = tf.optimizers.Adam(learning_rate=1e-4)

    start_block = 0; losses = []; ces = []; tec = []; accs = []; fbl = []

    import time

    wait = 0
    # Initialize the blocks - fix the first block
    idxs_blocks = list(range(2,opts.n_layers)); random.shuffle(idxs_blocks)
    fixed_blocks = [0] + [sorted(idxs_blocks[:opts.n_blocks-1])[0]]
    opts.minimal_blen = 0.75*(opts.n_blocks-len(fixed_blocks)+1) #set the minimal block number to be trained

    for epoch in range(1, opts.epochs+1):
        t0 = time.time()
        
        if epoch%opts.changing_epochs == 1:
            # Change the blocks in the training phase each opts.initial_epochs epochs
            idxs_blocks = list(range(fixed_blocks[-1]+1,opts.n_layers)); random.shuffle(idxs_blocks)
            idxs = fixed_blocks + sorted(idxs_blocks[:opts.n_blocks-len(fixed_blocks)]) + [opts.n_layers]
            optimizers = [tf.optimizers.Adam(opts.lr) for _ in range(len(idxs)-1)]

        if epoch > opts.initial_epochs:
            # Change the probabilities of the training phase of the blocks after the initial full training
            layer_probs = p_grads_probs((idxs[-1]-idxs[0])+1, deg=opts.deg, scale=1.5, distribution='chi2')[1:]
            block_probs = [sum(layer_probs[idxs[i]:idxs[i+1]]) for i in range(len(idxs)-1)]
            block_choice = []
            # Select the blocks to be trained
            while len(block_choice) < max(1,int(opts.minimal_blen)):
                try:
                    block_choice = list(set(random.choices(list(range(opts.n_blocks)), 
                                                weights=block_probs, k=opts.n_blocks)))
                except:
                    print(idxs, '\n', block_probs, '\n')
                    raise ValueError('Error in the block choice')
        else:
            block_choice = list(range(len(idxs)-1))

        # Train the model
        running_loss, running_ce, first_block_loss = train(net_ff, optimizers, linear_cf, optimizer_cf,
                                                            train_dataset, start_block, opts, idxs, block_choice)
        
        fbl.append(first_block_loss) # evaluate the loss of the first fixed block
        if epoch > opts.initial_epochs and np.abs(fbl[-1]-fbl[-2]) < opts.layer_threshold:
            wait += 1
            # Evaluate the stationarity of the Loss function on the first fixed block
            if wait>opts.deg_patience:
                if len(fixed_blocks)==opts.n_blocks:
                    # If all the blocks are fixed, stop the training
                    print(f"Training completed in {epoch} epochs")
                    break
                fixed_blocks += [idxs[len(fixed_blocks)]]
                opts.minimal_blen = 0.75*(opts.n_blocks-len(fixed_blocks)+1)
                opts.deg+=1
                start_block +=1
                wait = 0
        else:
            wait = 0

        t1 = time.time()-t0

        # Evaluate the accuracy on the test set - not used by now
        # train_slow_acc, train_fast_acc = test(net_ff, linear_cf, train_dataset)
        # test_slow_acc, test_fast_acc = test(net_ff, linear_cf, test_dataset)    

        t2 = time.time()-t0
        
        # Compute accuracy on the test set.
        acc = predict(x_test, y_test, net_ff, idxs, opts, return_acc=True)[1]

        print(f"{t1:.2f} sec | {t2:.2f} sec -->",
              f"Step {epoch:3d} | Loss: {running_loss:7.4f} | CE: {running_ce:7.4f} | ACCURACY: {acc.numpy()*100:3.2f}",
              f"| {start_block+1}° block loss: {first_block_loss:7.4f}",
              f"| deg: {opts.deg:2d} | BLOCK CHOICE: {block_choice} | EXTREMA: {idxs}",
              f"| FIXED: {fixed_blocks}",
            #   f"-- TRAIN: fast {train_fast_acc:.2f} (err {(100. - train_fast_acc):.2f}) slow {train_slow_acc:.2f} (err {(100. - train_slow_acc):.2f})",
            #   f"-- TEST: fast {test_fast_acc:.2f} (err {(100. - test_fast_acc):.2f}) - slow {test_slow_acc:.2f} (err {(100. - test_slow_acc):.2f})"
              )

        losses.append(running_loss.numpy())
        accs.append(acc.numpy())
        ces.append(running_ce.numpy())
        tec.append(t1)

########################### TEST SHOW IMAGES ##################


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

        if epoch ==1:
            plt.imshow(tf.reshape(sample[0,:], x_train.shape[1:]))
            # dont show axis
            plt.axis('off')
            plt.title(f'Label {y_test[0]}')
            plt.tight_layout()
            plt.savefig('sample.png')
            plt.show()


        if (epoch==1) or (epoch%20 == 1):
            fig, axs = plt.subplots(1, 2, figsize=(20, 5), sharey=True,
                                    gridspec_kw={'width_ratios': [2, 1], 'wspace':-0.225})

            # Plotting the first image with 2/3 width
            axs[0].imshow(acts_for_labels[0, :, :] / tf.reduce_max(acts_for_labels[0, :, :], axis=0))
            axs[0].set_xticks(list(range(0, opts.n_layers-1)))
            axs[0].set_xticklabels(list(range(1, opts.n_layers)))
            #rotate x ticks
            for tick in axs[0].get_xticklabels():
                tick.set_rotation(90)
            axs[0].set_yticks(list(range(opts.n_classes)))

            # Plotting the second image with 1/3 width
            axs[1].barh(list(range(10)), tf.transpose(tf.reduce_mean(acts_for_labels[0, :, :], axis=-1)/tf.reduce_max(tf.reduce_mean(acts_for_labels[0, :, :], axis=-1))))
            for tick in axs[1].get_xticklabels():
                tick.set_rotation(90)
            plt.tight_layout()
            plt.savefig(f'epoch_{epoch}.png')
            plt.show()


################################################################    
    return losses, accs, ces, tec

    
if __name__ == '__main__':

    set_seed(opts.seed, -1)

    losses, accs, ces, tec = main(opts) 
