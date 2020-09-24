
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from training import misc

synthesis_kwargs = dict(minibatch_size=8)

_Gs_cache = dict()

def load_Gs(url):
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]



def draw_figure(png, Gs, seeds):
    avg_dlantents_b = Gs.get_var('dlatent_avg_b')
    avg_dlantents_c = Gs.get_var('dlatent_avg_c')

    for seed in seeds:
        rnd = np.random.RandomState(seed)
        b1 = rnd.randn(Gs.input_shapes[0][1])
        b1 = b1[np.newaxis]
        b1 = Gs.components.mapping_b.run(b1, None)
        b1_v = b1[0, 0, :]
        #
        b1[:, :] = (b1_v - avg_dlantents_b) * 0.9 + avg_dlantents_b
        # change C
        for i in range(20):
            c = rnd.randn(Gs.input_shapes[1][1])
            c = c[np.newaxis]
            c = Gs.components.mapping_c.run(c, None)  # [seed, layer, component]
            c_v = c[0, 0, :]
            c[:, :] = (c_v - avg_dlantents_c) * 0.7 + avg_dlantents_c
            current_png = png + '/seedc_%d_%d' % (seed, i) + '.png'
            gen = Gs.components.synthesis.run(b1, c, randomize_noise=False, **synthesis_kwargs)[-1]
            misc.save_image_grid(gen, current_png, drange=[-1, 1], grid_size=(1, 1))
        b1_v = b1[0, 0, :]
        c = rnd.randn(Gs.input_shapes[1][1])
        c = c[np.newaxis]
        c = Gs.components.mapping_c.run(c, None)  # [seed, layer, component]
        c[:, :] = avg_dlantents_c
        for j in range(80):
            random_b2 = rnd.randn(Gs.input_shapes[0][1])
            random_b2 = random_b2[np.newaxis]
            random_b2 = Gs.components.mapping_b.run(random_b2, None)
            b2_v = (random_b2[0, 0, :] - avg_dlantents_b) * 0.5 + avg_dlantents_b
            print(b2_v.shape)
            # gram-schmidt process

            a1 = np.sum(b1_v * b2_v, dtype=np.float32)
            a2 = np.sum(b1_v * b1_v, dtype=np.float32)
            print(a1)
            print(a2)
            b2_v = b2_v - a1 / a2 * b1_v
            print(b1_v.shape)
            print(b2_v.shape)
            print(np.sum(b1_v * b2_v))
            for i in range(10):
                tmp = np.empty_like(b1)
                tmp[:, :] = b1_v + 0.1 * i * b2_v
                current_png = png + '/seedb%d_%d_%d' % (seed, j, i) + '.png'
                gen = Gs.components.synthesis.run(tmp, c, randomize_noise=False, **synthesis_kwargs)[-1]
                misc.save_image_grid(gen, current_png, drange=[-1, 1], grid_size=(1, 1))

#---------------------------------------------------------------------------
# Main program.

def main():
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
    network_pkl = 'network-snapshot-010000.pkl'
    G, D, Gs = misc.load_pkl(network_pkl)
    draw_figure(config.result_dir, Gs, seeds = [2, 7, 8, 11, 23])

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
