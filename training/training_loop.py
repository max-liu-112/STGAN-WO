

"""Main training script."""

import os
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

import config
import train
from training import dataset
from training import misc
from metrics import metric_base

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod_from_trainingset, mirror_augment, drange_data, drange_net):
      ##turn the tuple to list
    with tf.name_scope('ProcessReals'):
        reals = []
        x = list(x)
        num = len(x)
        with tf.name_scope('DynamicRange'):
            for m in range(num):
                x[m]= tf.cast(x[m], tf.float32)
                x[m] = misc.adjust_dynamic_range(x[m], drange_data, drange_net)
        with tf.name_scope('UpscaleLOD'):
            x = x[-1::-1]  ##[512, 64, 32, 16, 8]
            lod_max = lod_from_trainingset[0]
            lod_split = lod_from_trainingset[-2]
            down_sampled = tf.identity(x[0])
            dif = 0
            reals.append(tf.identity(x[0]))
            for lod in range(1, lod_max+1):
                down_sampled = (down_sampled[:,:,0::2,0::2] + down_sampled[:,:,0::2,1::2] + down_sampled[:,:,1::2,0::2] + down_sampled[:,:,1::2,1::2]) * 0.25
                if lod < lod_split:
                    reals.append(tf.identity(down_sampled))
                    dif = dif + 1
                else:
                    reals.append(tf.identity(down_sampled - x[lod - dif])) ## obtain the structure compenent

        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(reals[0])
                mask = np.random.uniform(0.0, 1.0, [s.shape[0].value, 1, 1, 1])
                num = len(reals)
                for m in range(num):
                    s = tf.shape(reals[m])
                    _mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                    reals[m] = tf.where(_mask < 0.5, reals[m], tf.reverse(reals[m], axis=[3]))
        assert len(reals) == lod_max + 1
        reals = reals[-1::-1]
        return tuple(reals)
#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    num_gpus,
    minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
    max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
    G_lrate_base            = 0.001,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.001,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 160,      # Default interval of progress snapshots.
    tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:30, 1024:20}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0
    s.resolution = 2 ** training_set.resolution_log2

    # Minibatch size.
    s.minibatch = minibatch_base
    s.minibatch -= s.minibatch % num_gpus
    if s.resolution in max_minibatch_per_gpu:
        s.minibatch = min(s.minibatch, max_minibatch_per_gpu[s.resolution] * num_gpus)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup
    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

#----------------------------------------------------------------------------
# Main training script.

def training_loop(
    submit_config,
    G_args                  = {},       # Options for generator network.
    D_args                  = {},       # Options for discriminator network.
    G_opt_args              = {},       # Options for generator optimizer.
    D_opt_args              = {},       # Options for discriminator optimizer.
    G_loss_args             = {},       # Options for generator loss.
    D_loss_args             = {},       # Options for discriminator loss.
    dataset_args            = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
    D_repeats               = 1,        # How many times the discriminator is trained per G iteration.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    total_kimg              = 15000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 1,        # How often to export image snapshots?
    network_snapshot_ticks  = 10,       # How often to export network snapshots?
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_run_id           = None,     # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,     # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0):     # Assumed wallclock time at the beginning. Affects reporting.

    # Initialize dnnlib and TensorFlow.
    ctx = dnnlib.RunContext(submit_config, train)
    tflib.init_tf(tf_config)

    # Load training set.
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            G, D, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
            Gs = G.clone('Gs')
    G.print_layers(); D.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // submit_config.num_gpus
        Gs_beta         = 0.5 ** tf.div(tf.cast(minibatch_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    G_opt = tflib.Optimizer(name='TrainG', learning_rate=lrate_in, **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', learning_rate=lrate_in, **D_opt_args)
    lod_from_trainingset = training_set.tfr_lods  ##[ 5, 4, 3, 0]
    for gpu in range(submit_config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            reals, labels = training_set.get_minibatch_tf()
            reals = process_reals(reals, lod_from_trainingset, mirror_augment, training_set.dynamic_range, drange_net)
            with tf.name_scope('G_loss'):
                G_loss , on_penalty= dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set, minibatch_size=minibatch_split, **G_loss_args)
            with tf.name_scope('D_loss'):
                D_loss = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split,  reals=reals, labels=labels, **D_loss_args)
            G_opt.register_gradients(tf.reduce_mean(G_loss) + on_penalty, G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)

    print('Setting up snapshot image grid...')
    grid_size, grid_reals, grid_labels,  grid_latents = misc.setup_snapshot_image_grid(G, training_set, **grid_args)
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
    print('Setting up run dir...')
    grid_reals = process_reals(grid_reals, lod_from_trainingset, mirror_augment, training_set.dynamic_range, drange_net)
    grid_fakes = Gs.run(grid_latents[0], grid_latents[1], grid_labels, is_validation=True,
                        minibatch_size=sched.minibatch // submit_config.num_gpus)

    assert len(grid_reals)   == len(grid_fakes)
    num = len(grid_reals)
    for m in range(num):
        if m < lod_from_trainingset[0] - lod_from_trainingset[-2] + 1:
            # save the structure component
            misc.save_image_grid(np.absolute(grid_fakes[m])-1.0,
                                 os.path.join(submit_config.run_dir, 'fakes_{:0>6}_{:0>3}.png'.format(resume_kimg, m)),
                                 drange=drange_net, grid_size=grid_size)
            misc.save_image_grid(np.absolute(grid_reals[m].eval())-1.0,
                                 os.path.join(submit_config.run_dir, 'reals_{:0>3}.png'.format(m)),
                                 drange=drange_net, grid_size=grid_size)
        else:
            # save rgb
            misc.save_image_grid(grid_fakes[m],
                                 os.path.join(submit_config.run_dir, 'fakes_{:0>6}_{:0>3}.png'.format(resume_kimg, m)),
                                 drange=drange_net, grid_size=grid_size)
            misc.save_image_grid(grid_reals[m].eval(),
                                 os.path.join(submit_config.run_dir, 'reals_{:0>3}.png'.format(m)),
                                 drange=drange_net, grid_size=grid_size)

    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training...\n')
    ctx.update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = ctx.get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    while cur_nimg < total_kimg * 1000:
        if ctx.should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
        training_set.configure(sched.minibatch // submit_config.num_gpus)

        # Run training ops.
        for _mb_repeat in range(minibatch_repeats):
            for _D_repeat in range(D_repeats):
                tflib.run([D_train_op, Gs_update_op], {lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
                cur_nimg += sched.minibatch
            tflib.run([G_train_op], { lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})


        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = ctx.get_time_since_last_update()
            total_time = ctx.get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %-4.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/minibatch', sched.minibatch),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if cur_tick % image_snapshot_ticks == 0 or done:

                grid_fakes = Gs.run(grid_latents[0],grid_latents[1], grid_labels,  is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus)
                num = len(grid_fakes)
                for m in range(num):
                    if m < lod_from_trainingset[0] - lod_from_trainingset[-2] + 1:
                        misc.save_image_grid(np.absolute(grid_fakes[m])-1.0,
                                             os.path.join(submit_config.run_dir,
                                                          'fakes_{:0>6}_{:0>3}.png'.format(cur_nimg // 1000, m)),
                                             drange=drange_net, grid_size=grid_size)
                    else:
                        misc.save_image_grid(grid_fakes[m],
                                             os.path.join(submit_config.run_dir,
                                                          'fakes_{:0>6}_{:0>3}.png'.format(cur_nimg // 1000, m)),
                                             drange=drange_net, grid_size=grid_size)

            if cur_tick % network_snapshot_ticks == 0 or done or cur_tick == 1:
                pkl = os.path.join(submit_config.run_dir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl)
                metrics.run(pkl, run_dir=submit_config.run_dir, num_gpus=submit_config.num_gpus, tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            ctx.update(cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = ctx.get_last_update_interval() - tick_time

    # Write final results.
    misc.save_pkl((G, D, Gs), os.path.join(submit_config.run_dir, 'network-final.pkl'))
    summary_log.close()

    ctx.close()

#----------------------------------------------------------------------------
