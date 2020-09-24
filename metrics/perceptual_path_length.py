
"""Perceptual Path Length (PPL)."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

# Normalize batch of vectors.
def normalize(v):
    return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))

# Spherical interpolation of a batch of vectors.
def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
    p = t * tf.math.acos(d)
    c = normalize(b - d * a)
    d = a * tf.math.cos(p) + c * tf.math.sin(p)
    return normalize(d)

#----------------------------------------------------------------------------

class PPL(metric_base.MetricBase):
    def __init__(self, num_samples, epsilon, space, sampling, minibatch_per_gpu, crop = True, **kwargs):
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.minibatch_per_gpu = minibatch_per_gpu
        self.crop = crop

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph.
        distance_expr = []
        distance_b_expr = []
        distance_c_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                noise_vars = [var for name, var in Gs_clone.components.synthesis.vars.items() if name.startswith('noise')]

                # Generate random latents and interpolation t-values.

                lat_b_t01 = tf.random_normal([self.minibatch_per_gpu * 2] + Gs_clone.input_shapes[0][1:])
                lat_c_t01 = tf.random_normal([self.minibatch_per_gpu * 2] + Gs_clone.input_shapes[1][1:])
                lerp_t = tf.random_uniform([self.minibatch_per_gpu], 0.0, 1.0 if self.sampling == 'full' else 0.0)

                # Interpolate in W or Z.
                if self.space == 'w':

                    dlat_b_t01 = Gs_clone.components.mapping_b.get_output_for(lat_b_t01, None, is_validation=True)
                    dlat_c_t01 = Gs_clone.components.mapping_c.get_output_for(lat_c_t01, None, is_validation=True)

                    dlat_b_t0, dlat_b_t1 = dlat_b_t01[0::2], dlat_b_t01[1::2]
                    dlat_c_t0, dlat_c_t1 = dlat_c_t01[0::2], dlat_c_t01[1::2]

                    dlat_b_e0 = tflib.lerp(dlat_b_t0, dlat_b_t1, lerp_t[:, np.newaxis, np.newaxis])
                    dlat_c_e0 = tflib.lerp(dlat_c_t0, dlat_c_t1, lerp_t[:, np.newaxis, np.newaxis])
                    dlat_c_e1 = tflib.lerp(dlat_c_t0, dlat_c_t1, lerp_t[:, np.newaxis, np.newaxis] + self.epsilon)

                    tmp_b_e0 = dlat_b_e0[:, 0, :]
                    tmp_b_e1 = tflib.lerp(dlat_b_t0, dlat_b_t1, lerp_t[:, np.newaxis, np.newaxis] + self.epsilon)[:, 0, :]

                    a1 = tf.reduce_sum(tmp_b_e0 * tmp_b_e1, axis=1, keepdims=True)
                    a2 = tf.reduce_sum(tmp_b_e0 * tmp_b_e0, axis=1, keepdims=True)
                    tmp_b_e1 = tmp_b_e1 - a1 / a2 * tmp_b_e0
                    tmp_b_e1 = tmp_b_e1 / tf.sqrt(tf.reduce_sum(tmp_b_e1 * tmp_b_e1,  axis=1, keepdims=True))

                    dlat_b_e1  = tmp_b_e0 +  self.epsilon*tmp_b_e1
                    dlat_b_e1 = tf.reshape(dlat_b_e1, [self.minibatch_per_gpu, 1, -1])
                    sh = dlat_b_e0.shape.as_list()
                    dlat_b_e1 = tf.tile(dlat_b_e1, [1, sh[1], 1])


                    # caculate sololy and whole ppl

                    # change b only
                    dlat_b_e02 = tf.reshape(tf.stack([dlat_b_e0, dlat_b_e1], axis=1), dlat_b_t01.shape)
                    dlat_c_e02 = tf.reshape(tf.stack([dlat_c_e0, dlat_c_e0], axis=1), dlat_c_t01.shape)
                    # change c only

                    dlat_b_e03 = tf.reshape(tf.stack([dlat_b_e0, dlat_b_e0], axis=1), dlat_b_t01.shape)
                    dlat_c_e03 = tf.reshape(tf.stack([dlat_c_e0, dlat_c_e1], axis=1), dlat_c_t01.shape)
                    #  change all
                    dlat_b_e04 = tf.reshape(tf.stack([dlat_b_e0, dlat_b_e1], axis=1), dlat_b_t01.shape)
                    dlat_c_e04 = tf.reshape(tf.stack([dlat_c_e0, dlat_c_e1], axis=1), dlat_c_t01.shape)
                else: # space == 'z'

                    lat_b_t0, lat_b_t1 = lat_b_t01[0::2], lat_b_t01[1::2]
                    lat_c_t0, lat_c_t1 = lat_c_t01[0::2], lat_c_t01[1::2]

                    lat_b_e0 = slerp(lat_b_t0, lat_b_t1, lerp_t[:, np.newaxis])
                    lat_c_e0 = slerp(lat_c_t0, lat_c_t1, lerp_t[:, np.newaxis])

                    lat_b_e1 = slerp(lat_b_t0, lat_b_t1, lerp_t[:, np.newaxis] + self.epsilon)
                    lat_c_e1 = slerp(lat_c_t0, lat_c_t1, lerp_t[:, np.newaxis] + self.epsilon)

                    # chnage b only

                    lat_b_e02 = tf.reshape(tf.stack([lat_b_e0, lat_b_e1], axis=1), lat_b_t01.shape)
                    lat_c_e02 = tf.reshape(tf.stack([lat_c_e0, lat_c_e0], axis=1), lat_c_t01.shape)
                    dlat_b_e02 = Gs_clone.components.mapping_b.get_output_for(lat_b_e02, None, is_validation=True)
                    dlat_c_e02 = Gs_clone.components.mapping_c.get_output_for(lat_c_e02, None, is_validation=True)
                    #  change c only

                    lat_b_e03 = tf.reshape(tf.stack([lat_b_e0, lat_b_e0], axis=1), lat_b_t01.shape)
                    lat_c_e03 = tf.reshape(tf.stack([lat_c_e0, lat_c_e1], axis=1), lat_c_t01.shape)
                    dlat_b_e03 = Gs_clone.components.mapping_b.get_output_for(lat_b_e03, None, is_validation=True)
                    dlat_c_e03 = Gs_clone.components.mapping_c.get_output_for(lat_c_e03, None, is_validation=True)
                    # change a and b and c

                    lat_b_e04 = tf.reshape(tf.stack([lat_b_e0, lat_b_e1], axis=1), lat_b_t01.shape)
                    lat_c_e04 = tf.reshape(tf.stack([lat_c_e0, lat_c_e1], axis=1), lat_c_t01.shape)
                    dlat_b_e04 = Gs_clone.components.mapping_b.get_output_for(lat_b_e04, None, is_validation=True)
                    dlat_c_e04 = Gs_clone.components.mapping_c.get_output_for(lat_c_e04, None, is_validation=True)

                # Synthesize images.
                with tf.control_dependencies([var.initializer for var in noise_vars]): # use same noise inputs for the entire minibatch
                    images_2 = \
                    Gs_clone.components.synthesis.get_output_for(dlat_b_e02, dlat_c_e02, is_validation=True,
                                                                 randomize_noise=False)[-1]
                    images_3 = \
                    Gs_clone.components.synthesis.get_output_for(dlat_b_e03, dlat_c_e03, is_validation=True,
                                                                 randomize_noise=False)[-1]
                    images_4 = \
                    Gs_clone.components.synthesis.get_output_for(dlat_b_e04, dlat_c_e04, is_validation=True,
                                                                 randomize_noise=False)[-1]

                # Crop only the face region.
                if self.crop:
                    c = int(images_2.shape[2] // 8)
                    images_2 = images_2[:, :, c * 3: c * 7, c * 2: c * 6]
                    images_3 = images_3[:, :, c * 3: c * 7, c * 2: c * 6]
                    images_4 = images_4[:, :, c * 3: c * 7, c * 2: c * 6]


                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                if images_2.shape[2] > 256:
                    factor = images_2.shape[2] // 256

                    images_2 = tf.reshape(images_2, [-1, images_2.shape[1], images_2.shape[2] // factor, factor,
                                                     images_2.shape[3] // factor, factor])
                    images_2 = tf.reduce_mean(images_2, axis=[3, 5])
                    images_3 = tf.reshape(images_3, [-1, images_3.shape[1], images_3.shape[2] // factor, factor,
                                                     images_3.shape[3] // factor, factor])
                    images_3= tf.reduce_mean(images_3, axis=[3, 5])
                    images_4 = tf.reshape(images_4, [-1, images_4.shape[1], images_4.shape[2] // factor, factor,
                                                     images_4.shape[3] // factor, factor])
                    images_4 = tf.reduce_mean(images_4, axis=[3, 5])

                # Scale dynamic range from [-1,1] to [0,255] for VGG.

                images_2 = (images_2 + 1) * (255 / 2)
                images_3 = (images_3 + 1) * (255 / 2)
                images_4 = (images_4 + 1) * (255 / 2)

                # Evaluate perceptual distance.

                img_2_e0, img_2_e1 = images_2[0::2], images_2[1::2]
                img_3_e0, img_3_e1 = images_3[0::2], images_3[1::2]
                img_4_e0, img_4_e1 = images_4[0::2], images_4[1::2]
                distance_measure = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl')  # vgg16_zhang_perceptual.pkl
                distance_b_expr.append(distance_measure.get_output_for(img_2_e0, img_2_e1) * (1 / self.epsilon ** 2))
                distance_c_expr.append(distance_measure.get_output_for(img_3_e0, img_3_e1) * (1 / self.epsilon ** 2))
                distance_expr.append(distance_measure.get_output_for(img_4_e0, img_4_e1) * (1 / self.epsilon ** 2))

        # Sampling loop.
        all_distances = []

        b_distance = []
        c_distance = []
        for _ in range(0, self.num_samples, minibatch_size):
            all_distances += tflib.run(distance_expr)

            b_distance += tflib.run(distance_b_expr)
            c_distance += tflib.run(distance_c_expr)
        all_distances = np.concatenate(all_distances, axis=0)

        b_distances = np.concatenate(b_distance, axis=0)
        c_distances = np.concatenate(c_distance, axis=0)
        # Reject outliers.
        lo = np.percentile(all_distances, 1, interpolation='lower')
        hi = np.percentile(all_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= all_distances, all_distances <= hi), all_distances)
        self._report_result(np.mean(filtered_distances), suffix='_bc')



        lo = np.percentile(b_distances, 1, interpolation='lower')
        hi = np.percentile(b_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= b_distances, b_distances <= hi), b_distances)
        self._report_result(np.mean(filtered_distances), suffix='_b')

        lo = np.percentile(c_distances, 1, interpolation='lower')
        hi = np.percentile(c_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= c_distances, c_distances <= hi), c_distances)
        self._report_result(np.mean(filtered_distances), suffix='_c')

#----------------------------------------------------------------------------
