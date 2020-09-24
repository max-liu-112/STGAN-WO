
"""Inception Score (IS)."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class IS(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_softmax.pkl')
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents_b = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                latents_c = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                images = Gs_clone.get_output_for(latents_b, latents_c, None, is_validation=True, randomize_noise=True)
                assert len(images) == 7
                images = tflib.convert_images_to_uint8(images[-1])
                assert images.shape[2] == images.shape[3] == 512
                result_expr.append(inception_clone.get_output_for(images))

        # Calculate activations for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]

        # Calculate IS.
        scores = []
        for i in range(self.num_splits):
            part = activations[i * self.num_images // self.num_splits : (i + 1) * self.num_images // self.num_splits]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        self._report_result(np.mean(scores), suffix='_mean')
        self._report_result(np.std(scores), suffix='_std')

#----------------------------------------------------------------------------
