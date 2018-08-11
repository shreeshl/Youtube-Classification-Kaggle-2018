# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

import scipy.io as sio
import numpy as np

FLAGS = flags.FLAGS


flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")



flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", False,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 16384,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 2048,
                    "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("dbof_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("dbof_var_features", 0,
                     "Variance features on top of Dbof cluster layer.")

flags.DEFINE_string("dbof_activation", "relu", 'dbof activation')

flags.DEFINE_bool("softdbof_maxpool", False, 'add max pool to soft dbof')

flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                   "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                   "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                   "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")


flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")



flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("fv_relu", True,
                     "ReLU after the NetFV hidden layer.")


flags.DEFINE_bool("fv_couple_weights", True,
                     "Coupling cluster weights or not")
 
flags.DEFINE_float("fv_coupling_factor", 0.01,
                     "Coupling factor")


flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_cells_video", 1024, "Number of LSTM cells (video).")
flags.DEFINE_integer("lstm_cells_audio", 128, "Number of LSTM cells (audio).")



flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_cells_video", 1024, "Number of GRU cells (video).")
flags.DEFINE_integer("gru_cells_audio", 128, "Number of GRU cells (audio).")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                     "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                     "Random sequence input for gru.")
flags.DEFINE_bool("gru_backward", False, "BW reading for GRU")
flags.DEFINE_bool("lstm_backward", False, "BW reading for LSTM")


flags.DEFINE_bool("fc_dimred", True, "Adding FC dimred after pooling")



class NetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        V = tf.get_variable("V",
              [self.feature_size, self.feature_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        U = tf.get_variable("U",
              [self.feature_size, self.feature_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))

        reshaped_input = tf.multiply(tf.nn.tanh(tf.matmul(reshaped_input, V)), tf.nn.sigmoid(tf.matmul(reshaped_input, U)))
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        a = tf.multiply(a_sum,cluster_weights2)
        #bs x cs x nm
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input) #bs x cs x fs
        vlad = tf.transpose(vlad,perm=[0,2,1]) #bs x fs x cs
        vlad = tf.subtract(vlad,a)
        
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size]) #bs X (cs*fs)
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad




class NetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """
  def vlad_create(self, video_scope, audio_scope, reshaped_input, max_frames, add_batch_norm, is_training, video_size, audio_size, clusters, pooling='vlad'):
    
    if pooling=='vlad':
      print 'vlad'
      video_NetVLAD = NetVLAD(video_size, max_frames, clusters, add_batch_norm, is_training)
      audio_NetVLAD = NetVLAD(audio_size, max_frames, clusters/2, add_batch_norm, is_training)
    else:
      print 'fv'
      video_NetVLAD = NetFV(video_size, max_frames, clusters, add_batch_norm, is_training)
      audio_NetVLAD = NetFV(audio_size, max_frames, clusters/2, add_batch_norm, is_training)
    
    with tf.variable_scope(video_scope):
      vlad_video = video_NetVLAD.forward(reshaped_input[:,0:video_size])
      alpha = tf.get_variable("alpha", [1], initializer = tf.random_normal_initializer(mean=1.0,stddev=0.01))
      beta  = tf.get_variable("beta", [1], initializer  = tf.random_normal_initializer(mean=0.0,stddev=0.01))
      # tf.summary.histogram(alpha.name, alpha)
      # tf.summary.histogram(beta.name, beta)
      self.histo.append(alpha)
      self.histo.append(beta)
      vlad_video = tf.nn.l2_normalize(alpha*vlad_video + beta, 1)
    
    with tf.variable_scope(audio_scope):
      vlad_audio = audio_NetVLAD.forward(reshaped_input[:,video_size:])
      alpha = tf.get_variable("alpha", [1], initializer = tf.random_normal_initializer(mean=1.0,stddev=0.01))
      beta  = tf.get_variable("beta", [1], initializer  = tf.random_normal_initializer(mean=0.0,stddev=0.01))
      # tf.summary.histogram(alpha.name, alpha)
      # tf.summary.histogram(beta.name, beta)
      self.histo.append(alpha)
      self.histo.append(beta)
      vlad_audio = tf.nn.l2_normalize(alpha*vlad_audio + beta, 1)
      
    return vlad_video, vlad_audio

  def multiple_vlad(self, reshaped_input, max_frames, add_batch_norm, is_training, feature_size, video_size, audio_size, clusters, pooling_type=['vlad','vlad'], dimred=True):
    
    N = len(video_size)
    if dimred:
      dimred_weights = tf.get_variable("dimred_weights",
            [feature_size, feature_size/2],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
      
      reshaped_input = tf.matmul(reshaped_input, dimred_weights)
  
    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")
    
    audio, video = [], []
    for i in range(N):
      v, a = self.vlad_create("video_VLAD"+str(i), "audio_VLAD"+str(i), reshaped_input, max_frames, add_batch_norm, \
                              is_training, video_size[i], audio_size[i], clusters[i], pooling_type[i])
      video.append(v)
      audio.append(a)
    
    vlad = tf.concat([tf.concat(video, 1), tf.concat(audio, 1)],1)
    # print vlad.get_shape()
    vlad = vlad/(N**0.5)
    return vlad

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    dimred_video = tf.get_variable("dimred_video",
      [1024, 400],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(1024)))
    dimred_audio = tf.get_variable("dimred_audio",
      [128, 50],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(1024)))

    reshaped_input  = tf.reshape(model_input, [-1, feature_size])
    reshaped_input1_video = tf.matmul(tf.reshape(model_input[:,:100,:1024], [-1, 1024]), dimred_video)
    reshaped_input2_video = tf.matmul(tf.reshape(model_input[:,100:200,:1024], [-1, 1024]), dimred_video)
    reshaped_input3_video = tf.matmul(tf.reshape(model_input[:,200:,:1024], [-1, 1024]), dimred_video)

    reshaped_input1_audio = tf.matmul(tf.reshape(model_input[:,:100,1024:], [-1, 128]), dimred_audio)
    reshaped_input2_audio = tf.matmul(tf.reshape(model_input[:,100:200,1024:], [-1, 128]), dimred_audio)
    reshaped_input3_audio = tf.matmul(tf.reshape(model_input[:,200:,1024:], [-1, 128]), dimred_audio)
    
    video_NetVLAD = NetVLAD(1024,max_frames,128, add_batch_norm, is_training)
    audio_NetVLAD = NetVLAD(128,max_frames,128/2, add_batch_norm, is_training)

    video_NetVLAD1 = NetVLAD(400,max_frames/3,64, add_batch_norm, is_training)
    audio_NetVLAD1 = NetVLAD(50,max_frames/3,64/2, add_batch_norm, is_training)

    video_NetVLAD2 = NetVLAD(400,max_frames/3,64, add_batch_norm, is_training)
    audio_NetVLAD2 = NetVLAD(50,max_frames/3,64/2, add_batch_norm, is_training)

    video_NetVLAD3 = NetVLAD(400,max_frames/3,64, add_batch_norm, is_training)
    audio_NetVLAD3 = NetVLAD(50,max_frames/3,64/2, add_batch_norm, is_training)
    
    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024])

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])
    
    with tf.variable_scope("video_VLAD1"):
        vlad_video1 = video_NetVLAD1.forward(reshaped_input1_video)

    with tf.variable_scope("audio_VLAD1"):
        vlad_audio1 = audio_NetVLAD1.forward(reshaped_input1_audio)

    with tf.variable_scope("video_VLAD2"):
        vlad_video2 = video_NetVLAD2.forward(reshaped_input2_video)

    with tf.variable_scope("audio_VLAD2"):
        vlad_audio2 = audio_NetVLAD2.forward(reshaped_input2_audio)

    with tf.variable_scope("video_VLAD3"):
        vlad_video3 = video_NetVLAD3.forward(reshaped_input3_video)

    with tf.variable_scope("audio_VLAD3"):
        vlad_audio3 = audio_NetVLAD3.forward(reshaped_input3_audio)

    vlad = tf.concat([vlad_video,vlad_video1,vlad_video2,vlad_video3,vlad_audio,vlad_audio1,vlad_audio2,vlad_audio3],1)
    vlad_dim = vlad.get_shape().as_list()[1]
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(vlad_dim)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    if relu:
      activation = tf.nn.relu6(activation)
   

    if gating:
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
        
        gates = tf.matmul(activation, gating_weights)
 
        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)

       
        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    # activation = tf.layers.dropout(activation, rate = 0.1, training=is_training)
    
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)


class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
    random_frames = FLAGS.lstm_random_sequence
    iterations = FLAGS.iterations
    backward = FLAGS.lstm_backward

    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)
    if backward:
      model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1) 
 
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

    loss = 0.0
    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)