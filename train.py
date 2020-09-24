

import copy
import dnnlib
from dnnlib import EasyDict

import config
from metrics import metric_base

#----------------------------------------------------------------------------

desc          = 'sgan'                                                                 # Description string included in result subdir name.
train         = EasyDict(run_func_name='training.training_loop.training_loop')         # Options for training loop.
G             = EasyDict(func_name='training.networks_stylegan.G_style')               # Options for generator network.
D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # Options for discriminator network.
G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for generator optimizer.
D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for discriminator optimizer.
G_loss        = EasyDict(func_name='training.loss.G_logistic_nonsaturating')           # Options for generator loss.
D_loss        = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0) # Options for discriminator loss.
dataset       = EasyDict()                                                             # Options for load_dataset().
sched         = EasyDict()                                                             # Options for TrainingSchedule.
grid          = EasyDict(size='1080p', layout='random')                                   # Options for setup_snapshot_image_grid().
metrics       = [metric_base.fid50k]                                                   # Options for MetricGroup.
submit_config = dnnlib.SubmitConfig()                                                  # Options for dnnlib.submit_run().
tf_config     = {'rnd.np_random_seed': 1000}                                           # Options for tflib.init_tf().

# Dataset.
desc += '-ffhq';     dataset = EasyDict(tfrecord_dir='ffhq_datasets');                 train.mirror_augment = False

# Number of GPUs.
desc += '-4gpu'; submit_config.num_gpus = 4; sched.minibatch_base = 16
# desc += '-8gpu'; submit_config.num_gpus = 8; sched.minibatch_base = 32; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}

# Default options.
train.total_kimg = 10000
sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)


#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Main entry point for training.

def main():
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
