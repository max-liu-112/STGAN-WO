
"""Loss functions."""

import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def _fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values

#----------------------------------------------------------------------------
# Loss functions advocated by the paper

def G_logistic_nonsaturating(G, D, opt,  training_set, minibatch_size): # pylint: disable=unused-argument
    latents_b = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    latents_c = tf.random_normal([minibatch_size] + G.input_shapes[1][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents_b, latents_c, labels, is_training=True)
    fake_scores_out = _fp32(D.get_output_for(fake_images_out[0], fake_images_out[1], fake_images_out[2],fake_images_out[3],
                                             fake_images_out[4], fake_images_out[5],fake_images_out[6], labels, is_training=True))
    loss = 0
    num = len(fake_scores_out)
    for m in range(num):
        loss += tf.nn.softplus(-fake_scores_out[m])
    loss = loss / float(num)
    with tf.name_scope('orthogonal_norm'):
        orth_penalty = 0
        for item in G.trainables:
            if 'U_weight' in item or 'V_weight' in item:
                if '16x16' in item or '32x32' in item or '64x64' in item:
                    weight = G.trainables[item]
                    weight_mat = tf.matmul(tf.transpose(weight, [1, 0]), weight)
                    orth_penalty += tf.reduce_mean(tf.abs(weight_mat - tf.eye(tf.shape(weight_mat)[0])))

    return loss, orth_penalty


def D_logistic_simplegp(G, D, opt, training_set, minibatch_size,  reals, labels, r1_gamma=10.0, r2_gamma=0.0): # pylint: disable=unused-argument
    latents_b = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    latents_c = tf.random_normal([minibatch_size] + G.input_shapes[1][1:])
    fake_images_out = G.get_output_for(latents_b, latents_c, labels, is_training=True)
    real_scores_out = _fp32(D.get_output_for(reals[0], reals[1],reals[2],reals[3],reals[4], reals[5], reals[6], labels, is_training=True))
    fake_scores_out = _fp32(D.get_output_for(fake_images_out[0], fake_images_out[1], fake_images_out[2],fake_images_out[3],
                                             fake_images_out[4], fake_images_out[5], fake_images_out[6], labels, is_training=True))
    assert len(real_scores_out)== len(fake_scores_out)
    real_score = []
    fake_score = []
    L = len(real_scores_out) # structure and texture independent
    loss = 0
    for num in range(L):
        real_score.append(autosummary('Loss/scores/real_{:0>2}'.format(num), real_scores_out[num]))
        fake_score.append(autosummary('Loss/scores/fake_{:0>2}'.format(num), fake_scores_out[num]))
    for score in fake_score:
        loss += tf.nn.softplus(score)
    for score in real_score:
        loss += tf.nn.softplus(-score)

    resolution = training_set.resolution_log2
    split_resolution = resolution - training_set.tfr_lods[-2]

    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty'):
            r1_penalty = 0
            for num in range(resolution, split_resolution, -1):
                real_loss = opt.apply_loss_scaling(tf.reduce_sum(real_score[0]))
                real_grads = opt.undo_loss_scaling(fp32(tf.gradients(real_loss, [reals[num-3]])[0]))
                r1_penalty += tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
            for num in range(split_resolution, 2, -1):
                real_loss = opt.apply_loss_scaling(tf.reduce_sum(real_score[1]))
                real_grads = opt.undo_loss_scaling(fp32(tf.gradients(real_loss, [reals[num-3]])[0]))
                r1_penalty += tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
            r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss += r1_penalty * (r1_gamma * 0.5)

    return loss/2.0

#----------------------------------------------------------------------------
