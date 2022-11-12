
"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import copy
from dcgru import DCGRUCell

class Seq2SeqModel(object):
  """Sequence-to-sequence model for human motion prediction"""

  def __init__(self,
               source_seq_len,
               target_seq_len,
               rnn_size, # hidden recurrent layer size
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               summaries_dir,
               number_of_actions,
               max_diffusion_step,
               filter_type,
               one_hot=True,
               eval_pose=False,
               dtype=tf.float32):
    """Create the model.

    Args:
      source_seq_len: lenght of the input sequence.
      target_seq_len: lenght of the target sequence.
      rnn_size: number of units in the rnn.
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      summaries_dir: where to log progress for tensorboard.
      number_of_actions: number of classes we have.
      one_hot: whether to use one_hot encoding during train/test (sup models).
      eval_pose: whether to evaluate on poses.
      dtype: the data type to use to store internal variables.
    """
    self.input_size_target = 30
    self.input_size = 30

    print( "One hot is ", one_hot )
    print( "Input size is %d" % self.input_size )

    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join( summaries_dir, 'train')))
    self.test_writer  = tf.summary.FileWriter(os.path.normpath(os.path.join( summaries_dir, 'test')))

    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.rnn_size = rnn_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype )
    self.learning_rate_decay_op = self.learning_rate.assign( self.learning_rate * learning_rate_decay_factor )
    self.global_step = tf.Variable(0, trainable=False)

    # === Transform the inputs ===
    with tf.name_scope("inputs"):
      act_pre_fw = tf.placeholder(dtype, shape=[None, source_seq_len-2, self.input_size], name="act_pre_fw")
      act_post_in_fw = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="act_post_in_fw")
      act_post_out_fw = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="act_post_out_fw")
      act_pose_fw = tf.placeholder(dtype, shape=[None, source_seq_len + target_seq_len, self.input_size], name="act_pose_fw")
      outputs_fake_fw_fix = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="outputs_fake_regroup")

      self.action_prefix_fw = act_pre_fw
      self.action_postfix_input_fw = act_post_in_fw
      self.action_postfix_output_fw = act_post_out_fw
      self.action_pose_fw = act_pose_fw
      self.outputs_fake_fw_fix = outputs_fake_fw_fix

      act_post_out_pose_fw = act_pose_fw[:, source_seq_len:, :]

      act = tf.concat([act_pre_fw,tf.expand_dims(act_post_in_fw[:,0,:], axis=1), act_post_out_fw], axis=1)
      act = tf.reverse(act, [1])
      act_pre_bw = act[:, :source_seq_len-2, :]
      act_post_in_bw = act[:, source_seq_len-2:source_seq_len+target_seq_len-2, :]
      act_post_out_bw = act[:, source_seq_len-1:, :]

      act_pose_bw = tf.reverse(act_pose_fw, [1])
      act_post_out_pose_bw = act_pose_bw[:, source_seq_len:, :]

      act_pre_fw = tf.transpose(act_pre_fw, [1, 0, 2])
      act_post_in_fw = tf.transpose(act_post_in_fw, [1, 0, 2])
      act_post_out_fw = tf.transpose(act_post_out_fw, [1, 0, 2])
      act_post_out_pose_fw = tf.transpose(act_post_out_pose_fw, [1, 0, 2])
      outputs_fake_fw_fix = tf.transpose(outputs_fake_fw_fix, [1, 0, 2])

      act_pre_bw = tf.transpose(act_pre_bw, [1, 0, 2])
      act_post_in_bw = tf.transpose(act_post_in_bw, [1, 0, 2])
      act_post_out_bw = tf.transpose(act_post_out_bw, [1, 0, 2])
      act_post_out_pose_bw = tf.transpose(act_post_out_pose_bw, [1, 0, 2])

      act_pre_fw = tf.reshape(act_pre_fw, [-1, self.input_size])
      act_post_in_fw = tf.reshape(act_post_in_fw, [-1, self.input_size])
      act_post_out_fw = tf.reshape(act_post_out_fw, [-1, self.input_size])
      outputs_fake_fw_fix = tf.reshape(outputs_fake_fw_fix, [-1, self.input_size])

      act_pre_bw = tf.reshape(act_pre_bw, [-1, self.input_size])
      act_post_in_bw = tf.reshape(act_post_in_bw, [-1, self.input_size])
      act_post_out_bw = tf.reshape(act_post_out_bw, [-1, self.input_size])

      act_pre_fw = tf.split(act_pre_fw, source_seq_len-2, axis=0)
      act_post_in_fw = tf.split(act_post_in_fw, target_seq_len, axis=0)
      act_post_out_fw = tf.split(act_post_out_fw, target_seq_len, axis=0)
      outputs_fake_fw_fix = tf.split(outputs_fake_fw_fix, target_seq_len, axis=0)

      act_pre_bw = tf.split(act_pre_bw, source_seq_len - 2, axis=0)
      act_post_in_bw = tf.split(act_post_in_bw, target_seq_len, axis=0)
      act_post_out_bw = tf.split(act_post_out_bw, target_seq_len, axis=0)

    # === Create the RNN that will keep the state ===
    print('rnn_size = {0}'.format(rnn_size))
    adj_mx = tf.get_variable('train_g_fw', shape=(self.input_size, self.input_size),
                             initializer=tf.random_uniform_initializer(minval=0, maxval=1))
    cell_fw = DCGRUCell(self.rnn_size, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=self.input_size,
                        filter_type=filter_type, num_proj=1)
    cell_bw = DCGRUCell(self.rnn_size, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=self.input_size,
                        filter_type=filter_type, num_proj=1)

    if num_layers == 2:
      cell_fw_no_projection = DCGRUCell(self.rnn_size, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=self.input_size,
                                        filter_type=filter_type)
      cell_bw_no_projection = DCGRUCell(self.rnn_size, adj_mx, max_diffusion_step=max_diffusion_step, num_nodes=self.input_size,
                                        filter_type=filter_type)
      cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw_no_projection] + [cell_fw] )
      cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw_no_projection] + [cell_bw])

    # for training
    outputs_fake_fw, enc_state_fw, dec_state_generated_fw = self.generator(cell_fw, act_pre_fw, act_post_in_fw, decoder_architecture='self_feeding', name='train_g_fw')
    outputs_fake_bw, enc_state_bw, _ = self.generator(cell_bw, act_pre_bw, act_post_in_bw, decoder_architecture='self_feeding', name='train_g_bw')

    dec_state_real_fw = self.discriminator(cell_fw, act_post_out_fw[:-1], tf.zeros_like(enc_state_fw), decoder_architecture='supervised', name='train_g_fw')
    act_post_out_fw_reverse = act_post_out_fw[:-1]
    dec_state_real_bw = self.discriminator(cell_bw, act_post_out_fw_reverse[::-1], tf.zeros_like(enc_state_fw), decoder_architecture='supervised', name='train_g_bw')

    self.outputs_fake_fw = outputs_fake_fw
    dec_state_fake_fw = self.discriminator(cell_fw, outputs_fake_fw_fix[:-1], tf.zeros_like(enc_state_fw), decoder_architecture='supervised', name='train_g_fw')
    outputs_fake_fw_fix_reverse = outputs_fake_fw_fix[:-1]
    dec_state_fake_bw = self.discriminator(cell_bw, outputs_fake_fw_fix_reverse[::-1], tf.zeros_like(enc_state_fw), decoder_architecture='supervised', name='train_g_bw')
    outputs_fake_fw_reverse = outputs_fake_fw[:-1]
    dec_state_generated_bw = self.discriminator(cell_bw, outputs_fake_fw_reverse[::-1], tf.zeros_like(enc_state_fw), decoder_architecture='supervised', name='train_g_bw')

    digit_real = self.dense(dec_state_real_fw, dec_state_real_bw)
    digit_fake = self.dense(dec_state_fake_fw, dec_state_fake_bw, reuse=True)
    digit_generated = self.dense(dec_state_generated_fw, dec_state_generated_bw, reuse=True)

    # for sampling
    outputs_fake_fw_test, _, _ = self.generator(cell_fw, act_pre_fw, act_post_in_fw, decoder_architecture='self_feeding', name='train_g_fw', reuse=True)

    self.outputs = outputs_fake_fw_test

    # losses
    if eval_pose:
      outputs_fake_poses_fw = tf.cumsum(outputs_fake_fw, axis=0)
      outputs_fake_poses_fw = outputs_fake_poses_fw + tf.tile(tf.expand_dims(act_pose_fw[:,source_seq_len-1,:], 0), [target_seq_len, 1, 1])
      with tf.name_scope("loss_angles_fw"):
        loss_angles_fw = tf.reduce_mean(tf.square(tf.subtract(act_post_out_pose_fw, outputs_fake_poses_fw)))
      self.mse_loss_fw = loss_angles_fw

      outputs_fake_poses_bw = tf.cumsum(outputs_fake_bw, axis=0)
      outputs_fake_poses_bw = tf.tile(tf.expand_dims(act_pose_bw[:,source_seq_len-1,:], 0), [target_seq_len, 1, 1]) - outputs_fake_poses_bw
      with tf.name_scope("loss_angles_bw"):
        loss_angles_bw = tf.reduce_mean(tf.square(tf.subtract(act_post_out_pose_bw, outputs_fake_poses_bw)))
      self.mse_loss_bw = loss_angles_bw

      self.mse_loss = loss_angles_fw + loss_angles_bw
      self.mse_loss_summary = tf.summary.scalar('loss/mse_loss', self.mse_loss)
    else:
      with tf.name_scope("loss_angles_fw"):
        loss_angles_fw = tf.reduce_mean(tf.square(tf.subtract(act_post_out_fw, outputs_fake_fw)))
      self.mse_loss_fw = loss_angles_fw
      self.mse_loss_summary_fw = tf.summary.scalar('loss/mse_loss_fw', self.mse_loss_fw)
      with tf.name_scope("loss_angles_bw"):
        loss_angles_bw = tf.reduce_mean(tf.square(tf.subtract(act_post_out_bw, outputs_fake_bw)))
      self.mse_loss_bw = loss_angles_bw
      self.mse_loss_summary_bw = tf.summary.scalar('loss/mse_loss_bw', self.mse_loss_bw)

    with tf.name_scope("g_loss"):
      g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=digit_generated, labels=tf.ones_like(digit_generated))) # activation goes here
    self.g_loss = g_loss
    self.g_loss_summary = tf.summary.scalar('loss/g_loss', self.g_loss)

    with tf.name_scope("d_loss"):
      d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=digit_real, labels=tf.ones_like(digit_real)))
      d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=digit_fake, labels=tf.zeros_like(digit_fake)))
      d_loss = d_loss_real + d_loss_fake
    self.d_loss = d_loss
    self.d_loss_summary = tf.summary.scalar('loss/d_loss', self.d_loss)


    tvars = tf.trainable_variables()
    mse_vars = [var for var in tvars if 'train_g' in var.name]
    d_vars = [var for var in tvars if 'bw' not in var.name]
    g_vars = [var for var in tvars if 'train_g_fw' in var.name]

    # separate gradient optimization
    gradients_mse = tf.gradients(self.mse_loss, mse_vars)

    clipped_gradients_mse, norm_mse = tf.clip_by_global_norm(gradients_mse, max_gradient_norm)
    self.gradient_norms_mse = norm_mse
    self.updates_mse = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(
      zip(clipped_gradients_mse, mse_vars), global_step=self.global_step)


    gradients_d = tf.gradients(self.d_loss, d_vars)

    clipped_gradients_d, norm_d = tf.clip_by_global_norm(gradients_d, max_gradient_norm)
    self.gradient_norms_d = norm_d
    self.updates_d = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(
      zip(clipped_gradients_d, d_vars), global_step=self.global_step)


    gradients_g = tf.gradients(self.g_loss, g_vars)

    clipped_gradients_g, norm_g = tf.clip_by_global_norm(gradients_g, max_gradient_norm)
    self.gradient_norms_g = norm_g
    self.updates_g = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(
      zip(clipped_gradients_g, g_vars), global_step=self.global_step)

    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    # === variables for loss in Euler Angles -- for each action

    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=10 )


  def step(self, session, action_prefix_fw, action_postfix_input_fw, action_postfix_output_fw, action_pose_fw,
             forward_only, srnn_seeds=False ):  # train or evaluate
    print('Outs from the batch')
    print(np.shape(action_prefix_fw))
    print(np.shape(action_postfix_input_fw))
    print(np.shape(action_postfix_output_fw))
    print(np.shape(action_pose_fw))
    """Run a step of the model feeding the given inputs.
      
    Args
      session: tensorflow session to use.
      encoder_inputs: list of numpy vectors to feed as encoder inputs.
      decoder_inputs: list of numpy vectors to feed as decoder inputs.
      decoder_outputs: list of numpy vectors that are the expected decoder outputs.
      forward_only: whether to do the backward step or only forward.
      srnn_seeds: True if you want to evaluate using the sequences of SRNN
    Returns
      A triple consisting of gradient norm (or None if we did not do backward),
      mean squared error, and the outputs.
    Raises
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    input_feed = {self.action_prefix_fw: action_prefix_fw,
                  self.action_postfix_input_fw: action_postfix_input_fw,
                  self.action_postfix_output_fw: action_postfix_output_fw,
                  self.action_pose_fw: action_pose_fw}

    # Output feed: depends on whether we do a backward step or not.
    if not srnn_seeds:
      if not forward_only:
        # Training step
        output_feed_mse = [self.updates_mse,
                           self.gradient_norms_mse,
                           self.mse_loss_fw,
                           self.mse_loss_bw,
                           self.mse_loss,
                           self.mse_loss_summary,
                           self.outputs_fake_fw]
        # training two times of mse
        print("TENSOR 1",np.shape(action_prefix_fw),"\n")
        print('>>>>>>>>>>>>>>>>>>>>>>>')
        print("TENSOR 2",np.shape(action_postfix_input_fw),"\n")

        outputs_mse = session.run(output_feed_mse, input_feed)
        outputs_mse = session.run(output_feed_mse, input_feed)
        outputs_fake_fw_fix = np.transpose(outputs_mse[-1], (1, 0, 2))
        # training one time of d
        input_feed_d = copy.copy(input_feed)
        input_feed_d[self.outputs_fake_fw_fix] = outputs_fake_fw_fix

        output_feed_d = [self.updates_d,         # Update Op that does SGD.
                       self.gradient_norms_d,  # Gradient norm.
                       self.d_loss,
                       self.d_loss_summary]

        outputs_d = session.run(output_feed_d, input_feed_d)
        # adjusting one time of g
        output_feed_g = [self.updates_g,         # Update Op that does SGD.
                       self.gradient_norms_g,  # Gradient norm.
                       self.g_loss,
                       self.g_loss_summary,
                       self.learning_rate_summary]

        outputs_g = session.run(output_feed_g, input_feed)

        return outputs_mse[1], outputs_mse[2], outputs_mse[3], outputs_mse[5], outputs_g[4]

      else:
        # Validation step, not on SRNN's seeds
        output_feed_mse = [self.mse_loss_fw,
                           self.mse_loss_summary,
                           self.outputs_fake_fw]
        outputs_mse = session.run(output_feed_mse, input_feed)
        outputs_fake_fw_fix = np.transpose(outputs_mse[-1], (1, 0, 2))

        input_feed_d = copy.copy(input_feed)
        input_feed_d[self.outputs_fake_fw_fix] = outputs_fake_fw_fix
        output_feed_d = [self.d_loss, # Loss for this batch.
                       self.d_loss_summary]

        outputs_d = session.run(output_feed_d, input_feed_d)

        output_feed_g = [self.g_loss,  # Loss for this batch.
                         self.g_loss_summary]

        outputs_g = session.run(output_feed_g, input_feed)
        return outputs_mse[0], outputs_mse[1]
    else:
      # Validation on SRNN's seeds
      output_feed = [self.mse_loss_fw,
                     self.mse_loss_summary,
                     self.outputs]

      outputs = session.run(output_feed, input_feed)

      return outputs[0], outputs[1], outputs[2]



  def get_batch( self, data, actions ):
    """Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    # Select entries at random
    all_keys    = list(data.keys())
    chosen_keys = np.random.choice( len(all_keys), self.batch_size )
    # How many frames in total do we need?
    total_frames = self.source_seq_len + self.target_seq_len

    encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-2, self.input_size_target), dtype=float)
    decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size_target), dtype=float)
    decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size_target), dtype=float)
    all_poses = np.zeros((self.batch_size, total_frames, self.input_size_target), dtype=float)


    for i in xrange( self.batch_size ):

      the_key = all_keys[ chosen_keys[i] ]

      # Get the number of frames
      n, _ = data[ the_key ].shape

      # Sample somewherein the middle
      idx = np.random.randint( 16, n-total_frames )

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames ,:]

      # Add the data
      encoder_inputs[i,:,0:self.input_size_target] = data_sel[1:self.source_seq_len-1, :] - data_sel[0:self.source_seq_len-2, :]
      decoder_inputs[i,:,0:self.input_size_target] = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :] - data_sel[self.source_seq_len-2:self.source_seq_len+self.target_seq_len-2, :]
      decoder_outputs[i,:,0:self.input_size_target] = data_sel[self.source_seq_len:, 0:self.input_size_target] - data_sel[self.source_seq_len-1:-1, 0:self.input_size_target]
      all_poses[i, :, 0:self.input_size_target] = data_sel[:, 0:self.input_size_target]


    return encoder_inputs, decoder_inputs, decoder_outputs, all_poses







  def generator(self, cell, act_pre, act_post_in, decoder_architecture=None, name=None, reuse=False):
    lf = None
    if decoder_architecture == 'self_feeding':
      def lf(prev, i):
        return prev
    elif decoder_architecture == "supervised":
      pass
    else:
      raise(ValueError, "unknown decoder architecture: %s" % decoder_architecture)

    with tf.variable_scope(name) as scope:
      if reuse:
        tf.get_variable_scope().reuse_variables()
      _, enc_state = tf.contrib.rnn.static_rnn(cell, act_pre, dtype=tf.float32, scope=scope)
      variable_scope.get_variable_scope().reuse_variables()
      outputs, dec_state = tf.contrib.legacy_seq2seq.rnn_decoder(act_post_in, enc_state, cell,
                                                                 loop_function=lf, scope=scope)

    return outputs, enc_state, dec_state

  def discriminator(self, cell, act_post_in, enc_state, decoder_architecture=None, name=None):
    lf = None
    if decoder_architecture == 'self_feeding':
      def lf(prev, i):
        return prev
    elif decoder_architecture == "supervised":
      pass
    else:
      raise(ValueError, "unknown decoder architecture: %s" % decoder_architecture)
    with tf.variable_scope(name) as scope:
      outputs, dec_state = tf.contrib.legacy_seq2seq.rnn_decoder(act_post_in, enc_state, cell, loop_function=lf, scope=scope)
    return dec_state


  def dense(self, state_fw, state_bw, reuse=False):
    with tf.variable_scope('train_d') as scope:  # name control linear
      if reuse:
        tf.get_variable_scope().reuse_variables()
      digit = tf.layers.dense(tf.concat([state_fw, state_bw], axis=1), 1)
    return digit
