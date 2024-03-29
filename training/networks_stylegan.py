﻿

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act


# Get/create weight tensor for a convolutional or fully-connected layer.
def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.
def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)
#----------------------------------------------------------------------------
# Convolutional layer.
def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
    return x

#----------------------------------------------------------------------------
# Apply bias and activation func.
def apply_bias_act(x, act='linear', alpha=None, gain=None, lrmul=1, bias_var='bias'):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain)

#----------------------------------------------------------------------------
# Naive upsampling (nearest neighbor) and downsampling (average pooling).
def naive_upsample_2d(x, factor=2):
    with tf.variable_scope('NaiveUpsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H, 1, W, 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        return tf.reshape(x, [-1, C, H * factor, W * factor])

def naive_downsample_2d(x, factor=2):
    with tf.variable_scope('NaiveDownsample'):
        _N, C, H, W = x.shape.as_list()
        x = tf.reshape(x, [-1, C, H // factor, factor, W // factor, factor])
        return tf.reduce_mean(x, axis=[3,5])



#----------------------------------------------------------------------------
# Modulated convolution layer.
# Weight demodulation proposed by "Analyzing and Improving the Image Quality of StyleGAN"
def modulated_conv2d_layer(x, y, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, fused_modconv=True, weight_var='weight', mod_weight_var='mod_weight', mod_bias_var='mod_bias'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    # Modulate.
    s = dense_layer(y, fmaps=x.shape[1].value, weight_var=mod_weight_var) # [BI] Transform incoming W to style.
    s = apply_bias_act(s, bias_var=mod_bias_var) + 1 # [BI] Add bias (initially 1).
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    # Demodulate.
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    # Reshape/scale input.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

    # Convolution with optional up/downsampling.
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')

    # Reshape/scale output.
    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
    return x

# convolutional layer with weight decomposition
# weight decomposition
def decomposition_conv2d_layer(x, y, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, fused_modconv=True, weight_var='weight', mod_weight_var='U_weight', mod_bias_var='mod_bias'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    # Get weight.
    out_channel = fmaps
    in_channel = kernel * kernel * x.shape[1].value
    s_dimension = min(out_channel, in_channel)
    U = get_weight([out_channel, s_dimension], gain=gain, use_wscale=use_wscale, lrmul=lrmul,
                   weight_var='U_' + weight_var)
    V = get_weight([in_channel, s_dimension], gain=gain, use_wscale=use_wscale, lrmul=lrmul,
                   weight_var='V_' + weight_var)
    # linear and normalization to obtain the style vector s and its diagnonal matrix S
    s = dense_layer(y, fmaps=s_dimension, weight_var=mod_weight_var)  # [BI] Transform incoming W to style.
    s = apply_bias_act(s, bias_var=mod_bias_var) + 1  # [BI] Add bias (initially 1).
    s *= tf.rsqrt(tf.reduce_mean(tf.square(s), axis=1, keepdims=True) + 1e-8)
    # construct diagnol matrix using style s
    S = tf.matrix_diag(s)
    # using S to construct a controllable matrix w
    w = tf.matmul(tf.reshape(S, [-1, s_dimension]), tf.transpose(V, [1, 0]))
    w = tf.reshape(w, [-1, s_dimension, in_channel])
    w = tf.transpose(w, [0, 2, 1])
    w = tf.matmul(tf.reshape(w, [-1, s_dimension]), tf.transpose(U, [1, 0]))
    w = tf.reshape(w, [-1, kernel, kernel, x.shape[1].value, out_channel])
    # normalize w similar to weight demodulation
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(w), axis=[1, 2, 3]) + 1e-8)  # [BO] Scaling factor.
        w *= d[:, np.newaxis, np.newaxis, np.newaxis, :]  # [BkkIO] Scale output feature maps.

    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(w, [1, 2, 3, 0, 4]), [w.shape[1], w.shape[2], w.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

    # Convolution with optional up/downsampling.
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
    return x

# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
        y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
        y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Composed of two sub-networks (G_mapping and G_synthesis) that are defined below.
#
def G_style(
    latents_in_b,                                   # First input: Conditioning labels z1[minibatch, label_size].
    latents_in_c,                                   # Second input: Conditioning labels z2 [minibatch, label_size].
    labels_in,                                      # Third input: Conditioning labels [minibatch, label_size].
    truncation_psi          = 0.7,                  # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = 8,                    # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                 # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                 # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                  # Probability of mixing styles during training. None = disable.
    is_training             = False,                # Network is under training? Enables and disables specific features.
    is_validation           = False,                # Network is under validation? Chooses which value to use for truncation_psi.
    is_template_graph       = False,                # True = template graph constructed by the Network class, False = actual evaluation.
    components              = dnnlib.EasyDict(),    # Container for sub-networks. Retained between calls.
    **kwargs):                                      # Arguments for sub-networks (G_mapping and G_synthesis).

    # Validate arguments.
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:##False
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training or (truncation_cutoff is not None and not tflib.is_tf_expression(truncation_cutoff) and truncation_cutoff <= 0):
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=G_synthesis, **kwargs)

    num_layers_b = components.synthesis.input_shapes[0][1]
    num_layers_c = components.synthesis.input_shapes[1][1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping_b' not in components:
        components.mapping_b = tflib.Network('G_mapping_b', func_name=G_mapping, dlatent_broadcast=num_layers_b, **kwargs)
    if 'mapping_c' not in components:
        components.mapping_c = tflib.Network('G_mapping_c', func_name=G_mapping, dlatent_broadcast=num_layers_c, **kwargs)

    # Setup variables.
    dlatent_avg_b = tf.get_variable('dlatent_avg_b', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)
    dlatent_avg_c = tf.get_variable('dlatent_avg_c', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    dlatents_b = components.mapping_b.get_output_for(latents_in_b, labels_in, **kwargs)
    dlatents_c = components.mapping_c.get_output_for(latents_in_c, labels_in, **kwargs)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg_b = tf.reduce_mean(dlatents_b[:, 0], axis=0)
            batch_avg_c = tf.reduce_mean(dlatents_c[:, 0], axis=0)
            update_op_b = tf.assign(dlatent_avg_b, tflib.lerp(batch_avg_b, dlatent_avg_b, dlatent_avg_beta))
            update_op_c = tf.assign(dlatent_avg_c, tflib.lerp(batch_avg_c, dlatent_avg_c, dlatent_avg_beta))
            with tf.control_dependencies([update_op_b, update_op_c]):
                dlatents_b = tf.identity(dlatents_b)
                dlatents_c = tf.identity(dlatents_c)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.name_scope('StyleMix_b'):
            latents3 = tf.random_normal(tf.shape(latents_in_b))
            dlatents3 = components.mapping_b.get_output_for(latents3, labels_in, is_training=is_training, **kwargs)
            dlatents3 = tf.cast(dlatents3, tf.float32)
            layer_idx = np.arange(num_layers_b)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers_b
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents_b = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents_b)), dlatents_b, dlatents3)
        with tf.name_scope('StyleMix_c'):
            latents4 = tf.random_normal(tf.shape(latents_in_c))
            dlatents4 = components.mapping_c.get_output_for(latents4, labels_in, is_training=is_training, **kwargs)
            dlatents4 = tf.cast(dlatents4, tf.float32)
            layer_idx = np.arange(num_layers_c)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers_c
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents_c = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents_c)), dlatents_c, dlatents4)

    #Apply truncation trick.
    if truncation_psi is not None and truncation_cutoff is not None:
        with tf.variable_scope('Truncation_b'):
            layer_idx = np.arange(num_layers_b)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            dlatents_b = tflib.lerp(dlatent_avg_b, dlatents_b, coefs)
        with tf.variable_scope('Truncation_c'):
            layer_idx = np.arange(num_layers_c)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            dlatents_c = tflib.lerp(dlatent_avg_c, dlatents_c, coefs)

    # Evaluate synthesis network.

    images_out = components.synthesis.get_output_for(dlatents_b, dlatents_c, force_clean_graph=is_template_graph, **kwargs)
    out = []
    num = len(images_out)
    for i in range(num):
        out.append(tf.identity(images_out[i], name='images_out%d'%(i)))
    return tuple(out)

#----------------------------------------------------------------------------
# Mapping network used in the proposed paper.
# 8-MLP is removed and replaced by a linear transformation

def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    labels_in,                              # Second input: Conditioning labels [minibatch, label_size].
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 1,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    normalize_latents       = True,         # Normalize latent vectors (Z) before feeding them to the mapping layers?
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.


    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # Normalize latents.
    if normalize_latents:
        with tf.variable_scope('Normalize'):
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul), lrmul=mapping_lrmul)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

#----------------------------------------------------------------------------
# Synthesis network

def G_synthesis(
    dlatents_in_b,                      # Input: Disentangled latents (W1) [minibatch, num_layers, dlatent_size].
    dlatents_in_c,                      # Input: Disentangled latents (W2) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            # Number of output color channels.
    resolution          = 1024,         # Output resolution.
    split_res_b         = 6,            # 2**6=64, corresponds to the resolution of 64*64
    fmap_base           = 16<<10,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer. halve of that in stylegan
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu'
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    fused_modconv       = True,         # Implement modulated_conv2d_layer() as a single fused op?
    structure           = 'fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    force_clean_graph   = False,        # True = construct a clean graph that looks nice in TensorBoard, False = default behavior.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    if is_template_graph: force_clean_graph = True
    if force_clean_graph: randomize_noise = False
    if structure == 'auto': structure = 'linear' if force_clean_graph else 'recursive'
    act = nonlinearity

    num_layers_b = split_res_b  * 2 - 3
    num_layers_c = (resolution_log2 -  split_res_b) * 3
    num_layers =  num_layers_b + num_layers_c
    images_out = []

    # Primary inputs.

    dlatents_in_b.set_shape([None, num_layers_b, dlatent_size])
    dlatents_in_b = tf.cast(dlatents_in_b, dtype)
    dlatents_in_c.set_shape([None, num_layers_c, dlatent_size])
    dlatents_in_c = tf.cast(dlatents_in_c, dtype)

    # Things to do at the end of each layer.
    # we remove th noise input
    def layer(x, layer_idx, fmaps, kernel, up=False):
        if layer_idx < num_layers_b:
            x = decomposition_conv2d_layer(x, dlatents_in_b[:, layer_idx], fmaps=fmaps, kernel=kernel, up=up,
                                       resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        else:
            x = decomposition_conv2d_layer(x, dlatents_in_c[:, layer_idx  - num_layers_b], fmaps=fmaps, kernel=kernel, up=up,
                                       resample_kernel=resample_kernel, fused_modconv=fused_modconv)
        return apply_bias_act(x, act=act)

    # Building blocks for remaining layers.
    def block_structure(x, res):  # res = 3..split_resolution_log2
        with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
            with tf.variable_scope('Conv0_up'):
                x = layer(x, layer_idx=res * 2 - 5, fmaps=nf(res - 1), kernel=3, up=True)
            with tf.variable_scope('Conv1'):
                x = layer(x, layer_idx=res * 2 - 4, fmaps=nf(res - 1), kernel=3)
            return x
    def block_color(x, res):  # res = split_res_log2 +1..resolution_log2
        with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
            with tf.variable_scope('Conv0_up'):
                x = layer(x, layer_idx=res * 3 - 12, fmaps=nf(res - 1), kernel=3, up=True)
            with tf.variable_scope('Conv1'):
                x = layer(x, layer_idx=res * 3 - 11, fmaps=nf(res - 1), kernel=3)
            with tf.variable_scope('Conv2'):
                x = layer(x, layer_idx=res * 3 - 10, fmaps=nf(res - 1), kernel=3)
            return x
    def torgb(x, res):  # res = 2..resolution_log2
        with tf.variable_scope('ToRGB_res%d' % res):
            return apply_bias_act(conv2d_layer(x, fmaps=num_channels, kernel=1))

    # Early layers.
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in_b)[0], 1, 1, 1])
        with tf.variable_scope('Conv'):
            x = layer(x, layer_idx=0, fmaps=nf(1), kernel=3)

    for res in range(3, split_res_b + 1):
        x = block_structure(x, res)
        out = torgb(x, res)
        images_out.append(out)  ##image_out = [3, 8, 8], [3, 16, 16], ..., [3, 64 , 64]
    for res in range(split_res_b + 1, resolution_log2 + 1):
        x = block_color(x, res)
        out = torgb(x, res)
        images_out.append(out)

    return_img = []
    j = 3
    for img in images_out:
        assert img.dtype == tf.as_dtype(dtype)
        return_img.append(tf.identity(img, name='images_out_{:0>4}'.format(2**j)))
        j = j + 1
    return tuple(return_img)  ##[8*8 16*16 32*32 64*64 128*128 256*256]


#----------------------------------------------------------------------------
# Discriminator

def D_basic(
    images_in_8x8, images_in_16x16, images_in_32x32, images_in_64x64, images_in_128x128, images_in_256x256, images_in_512x512,  #
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    split_res_b         = 6,            # 2**6=64, corresponds to the resolution of 64*64
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'fixed',       # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure == 'auto': structure = 'linear' if is_template_graph else 'recursive'
    act = nonlinearity
    images_in = []
    images_in_8x8.set_shape([None, num_channels, 8, 8])
    images_in_8x8 = tf.cast(images_in_8x8, dtype)
    images_in_16x16.set_shape([None, num_channels, 16, 16])
    images_in_16x16 = tf.cast(images_in_16x16, dtype)
    images_in_32x32.set_shape([None, num_channels, 32, 32])
    images_in_32x32 = tf.cast(images_in_32x32, dtype)
    images_in_64x64.set_shape([None, num_channels, 64, 64])
    images_in_64x64 = tf.cast(images_in_64x64, dtype)
    images_in_128x128.set_shape([None, num_channels, 128, 128])
    images_in_128x128 = tf.cast(images_in_128x128, dtype)
    images_in_256x256.set_shape([None, num_channels, 256, 256])
    images_in_256x256 = tf.cast(images_in_256x256, dtype)
    images_in_512x512.set_shape([None, num_channels, 512, 512])
    images_in_512x512 = tf.cast(images_in_512x512, dtype)

    images_in.append(images_in_8x8)
    images_in.append(images_in_16x16)
    images_in.append(images_in_32x32)
    images_in.append(images_in_64x64)
    images_in.append(images_in_128x128)
    images_in.append(images_in_256x256)
    images_in.append(images_in_512x512)

    labels_in.set_shape([None, label_size])
    labels_in = tf.cast(labels_in, dtype)

    scores_out = []

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=1), act=act)
    def block(x, res, g_img = None): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if g_img is not None:
                with tf.variable_scope('combine'):
                    x = tf.concat((x, g_img), axis= 1)
                    x = conv2d_layer(x, fmaps=nf(res - 1), kernel=1)
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3), act=act)
                with tf.variable_scope('Conv1_down'):
                    x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 2), kernel=3, down=True, resample_kernel=resample_kernel),act=act)

            else: # 4x4
                if mbstd_group_size > 1:
                    with tf.variable_scope('MinibatchStddev'):
                        x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                with tf.variable_scope('Conv'):
                    x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
                with tf.variable_scope('Dense0'):
                    x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)
                with tf.variable_scope('Dense1'):
                    x = apply_bias_act(dense_layer(x, fmaps=max(label_size, 1)))
            return x

    x = fromrgb(images_in[-1], resolution_log2)
    for res in range(resolution_log2, 2, -1):
        if res < resolution_log2 and res> split_res_b:
            x = block(x, res, images_in[res-3])
        else:
            x = block(x, res)
    score = block(x, 2)
    scores_out.append(score)

    x = fromrgb(images_in[resolution_log2 - split_res_b], split_res_b)
    for res in range(split_res_b, 2, -1):
        if res == split_res_b:
            x = block(x, res)
        else:
            x = block(x, res, images_in[res -3])
    score = block(x, 2)
    scores_out.append(score)

    # Label conditioning from "Which Training Methods for GANs do actually Converge?"
    if label_size:
        with tf.variable_scope('LabelSwitch'):
            scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)

    num = len(scores_out)
    for m in range(num):
        assert scores_out[m].dtype == tf.as_dtype(dtype)
        scores_out[m] = tf.identity(scores_out[m], name='scores_out_{:0>2}'.format(m))
    return tuple(scores_out)
#----------------------------------------------------------------------------
