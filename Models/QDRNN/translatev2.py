
"""Simple code for training an RNN for motion prediction."""
# Mainly adopted from https://github.com/una-dinosauria/human-motion-prediction

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import h5py

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
from ops.loaddata import load_data_ as ld
from ops.loaddata import norm
import prediction_modelv2 as prediction_model


# Learning
tf.app.flags.DEFINE_float("learning_rate", .05, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 10000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 1, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", 50000, "Iterations to train for.")
# Architecture
tf.app.flags.DEFINE_integer("size", 30, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 32, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 32, "Number of frames that the decoder has to predict. 25fps")
tf.app.flags.DEFINE_integer("max_diffusion_step", 3, "Number of maximum diffusion steps in the model.")
tf.app.flags.DEFINE_string("filter_type", "dual_random_walk", "laplacian/random_walk/dual_random_walk")
tf.app.flags.DEFINE_boolean("omit_one_hot", True, "Whether to remove one-hot encoding from the data")
tf.app.flags.DEFINE_boolean("train_on_euler", False, "Train using euler angle")
tf.app.flags.DEFINE_boolean("velocity", True, "Train using velocity")
tf.app.flags.DEFINE_string("action","all", "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
# Directories
tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/h3.6m/dataset"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("./experiments/"), "Training directory.")
# Evaluations
tf.app.flags.DEFINE_boolean("eval_pose", True, "Training evaluation on pose")
tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("save_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.normpath(os.path.join( FLAGS.train_dir, FLAGS.action,
  'out_{0}'.format(FLAGS.seq_length_out),
  'iterations_{0}'.format(FLAGS.iterations),
  'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
  'size_{0}'.format(FLAGS.size),
  'lr_{0}'.format(FLAGS.learning_rate)))

summaries_dir = os.path.normpath(os.path.join( train_dir, "log" )) # Directory for TB summaries

def create_model(session, actions, sampling=False):
  """Create translation model and initialize or load parameters in session."""

  model = prediction_model.Seq2SeqModel(
      FLAGS.seq_length_in if not sampling else 64,
      FLAGS.seq_length_out if not sampling else 30,
      FLAGS.size, # hidden layer size
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      summaries_dir,
      len( actions ),
      FLAGS.max_diffusion_step,
      FLAGS.filter_type,
      not FLAGS.omit_one_hot,
      FLAGS.eval_pose,
      dtype=tf.float32)

  if FLAGS.load <= 0:
    print("Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model

  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.normpath(os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) ))
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt_name )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model


def train():
  """Train a seq2seq model on human motion"""

  train=ld('test')
  train_set=norm(train)
  ns,fr,jn=np.shape(train_set)
  actions=list(range(0,ns))
  train_set=dict(zip(actions,train_set))
  

  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:

    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    model = create_model( sess, actions )
    model.train_writer.add_graph( sess.graph )
    print( "Model created" )

   
    #=== This is the training loop ===
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1


    for _ in xrange( FLAGS.iterations ):

      # === Training step ===
      action_prefix, action_postfix_input, action_postfix_output, action_poses = model.get_batch(train_set, not FLAGS.omit_one_hot)

      _, step_mse_fw_loss, step_mse_bw_loss, mse_loss_summary, lr_summary = model.step( sess, action_prefix[:,:,:], action_postfix_input[:,:,:], action_postfix_output[:,:,:], action_poses[:,:,:], False )
      model.train_writer.add_summary( mse_loss_summary, current_step)
      model.train_writer.add_summary( lr_summary, current_step )

      if current_step % 10 == 0:
        print("step {0:04d}; mse_loss_fw: {1:.4f}; mse_loss_bw: {2:.4f}".format(current_step, step_mse_fw_loss, step_mse_bw_loss))
      current_step += 1

      # === step decay ===
      if current_step % FLAGS.learning_rate_step == 0:
        sess.run(model.learning_rate_decay_op)

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.test_every == 0:

        # === Validation with randomly chosen seeds ===
        forward_only = True

        action_prefix, action_postfix_input, action_postfix_output, action_poses = model.get_batch(train_set, not FLAGS.omit_one_hot)

        step_mse_loss_fw, mse_loss_summary = model.step(sess, action_prefix[:,:,:], action_postfix_input[:,:,:], action_postfix_output[:,:,:], action_poses[:,:,:], forward_only)
        val_mse_loss_fw = step_mse_loss_fw
        model.test_writer.add_summary(mse_loss_summary, current_step)

        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 560, 1000]:
          print(" {0:5d} |".format(ms), end="")
        print()



        # Save the model
        if current_step % FLAGS.save_every == 0:
          print( "Saving the model..." )
          model.saver.save(sess, os.path.normpath(os.path.join(train_dir, 'checkpoint')), global_step=current_step )


        sys.stdout.flush()


def main(_):
    train()

if __name__ == "__main__":
  tf.app.run()
