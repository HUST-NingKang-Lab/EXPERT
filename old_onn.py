# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import sys
import json
import tensorflow as tf
import utils
import pickle

class model(object):
  def __init__(self, feature=None, feature_size=None, label=None, label_size=None, lr=1e-4, is_training=True, reuse=False, gpu_mode=True):
    self.lr = lr
    self.is_training = is_training
    self.reuse = reuse
    self.feature = feature
    self.feature_size = feature_size
    self.label = label
    self.label_size = label_size
    
    with tf.compat.v1.variable_scope('conv_vae', reuse=self.reuse):
      if (not gpu_mode):
        with tf.device('/cpu:0'):
          tf.compat.v1.logging.info('model using cpu!')
      else:
        tf.compat.v1.logging.info('model using gpu!')
        self.build_graph()
    self.init_session()

  def build_graph(self):
    self.g = tf.Graph()
    with self.g.as_default():
      #input abundance vector
      self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, self.feature_size])
      #ground truth
      self.y_0 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.label_size[0]])
      self.y_1 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.label_size[1]])
      self.y_2 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.label_size[2]])
      self.y_3 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.label_size[3]])
      self.y_4 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.label_size[4]])
      self.y_5 = tf.compat.v1.placeholder(tf.float32, shape=[None, self.label_size[5]])
      
      
      #fully connected layers using for extract underlying logic
      base0_1 = tf.compat.v1.layers.dense(self.x, 2048, activation=tf.nn.relu)
      base0_2 = tf.compat.v1.layers.dense(base0_1, 1024, activation=tf.nn.relu)
      #base0_3 = tf.layers.dense(base0_2, 1024, activation=tf.nn.relu)
      base0 = tf.compat.v1.layers.dense(base0_2, 512, activation=tf.nn.relu)
      
      #Ontology layer0 for label0 classifying
      otlg0_0 = tf.compat.v1.layers.dense(base0, 256, activation=tf.nn.relu)
      otlg0_1 = tf.compat.v1.layers.dense(otlg0_0, 128, activation=tf.nn.relu)
      self.otlg0 = tf.compat.v1.layers.dense(otlg0_1, self.label_size[0])
      self.pred0 = tf.nn.sigmoid(self.otlg0)
      
      #Ontology layer1 for label1 classifying
      otlg1_0 = tf.compat.v1.layers.dense(base0, 256, activation=tf.nn.relu)
      otlg1_1 = tf.compat.v1.layers.dense(otlg1_0, 128, activation=tf.nn.relu)
      otlg1_2 = tf.compat.v1.layers.dense(otlg1_1, 64, activation=tf.nn.relu)
      otlg1_c = tf.concat([otlg1_2, otlg0_1], axis=1)
      otlg1_3 = tf.compat.v1.layers.dense(otlg1_c, 128, activation=tf.nn.relu)
      self.otlg1 = tf.compat.v1.layers.dense(otlg1_3, self.label_size[1])
      self.pred1 = tf.nn.sigmoid(self.otlg1)
      
      #Ontology layer2 for label2 classifying
      otlg2_0 = tf.compat.v1.layers.dense(base0, 256, activation=tf.nn.relu)
      otlg2_1 = tf.compat.v1.layers.dense(otlg2_0, 128, activation=tf.nn.relu)
      otlg2_2 = tf.compat.v1.layers.dense(otlg2_1, 64, activation=tf.nn.relu)
      otlg2_c = tf.concat([otlg2_2, otlg1_3], axis=1)
      otlg2_3 = tf.compat.v1.layers.dense(otlg2_c, 128, activation=tf.nn.relu)
      self.otlg2 = tf.compat.v1.layers.dense(otlg2_3, self.label_size[2])
      self.pred2 = tf.nn.sigmoid(self.otlg2)
      
      #Ontology layer3 for label3 classifying
      otlg3_0 = tf.compat.v1.layers.dense(base0, 256, activation=tf.nn.relu)
      otlg3_1 = tf.compat.v1.layers.dense(otlg3_0, 128, activation=tf.nn.relu)
      otlg3_2 = tf.compat.v1.layers.dense(otlg3_1, 64, activation=tf.nn.relu)
      otlg3_c = tf.concat([otlg3_2, otlg2_3], axis=1)
      otlg3_3 = tf.compat.v1.layers.dense(otlg3_c, 128, activation=tf.nn.relu)
      self.otlg3 = tf.compat.v1.layers.dense(otlg3_3, self.label_size[3])
      self.pred3 = tf.nn.sigmoid(self.otlg3)
      
      #Ontology layer4 for label4 classifying
      otlg4_0 = tf.compat.v1.layers.dense(base0, 256, activation=tf.nn.relu)
      otlg4_1 = tf.compat.v1.layers.dense(otlg4_0, 128, activation=tf.nn.relu)
      otlg4_2 = tf.compat.v1.layers.dense(otlg4_1, 64, activation=tf.nn.relu)
      otlg4_c = tf.concat([otlg4_2, otlg3_3], axis=1)
      otlg4_3 = tf.compat.v1.layers.dense(otlg4_c, 128, activation=tf.nn.relu)
      self.otlg4 = tf.compat.v1.layers.dense(otlg4_3, self.label_size[4])
      self.pred4 = tf.nn.sigmoid(self.otlg4)
      
      #Ontology layer5 for label5 classifying
      otlg5_0 = tf.compat.v1.layers.dense(base0, 256, activation=tf.nn.relu)
      otlg5_1 = tf.compat.v1.layers.dense(otlg5_0, 128, activation=tf.nn.relu)
      otlg5_2 = tf.compat.v1.layers.dense(otlg5_1, 64, activation=tf.nn.relu)
      otlg5_c = tf.concat([otlg5_2, otlg4_3], axis=1)
      otlg5_3 = tf.compat.v1.layers.dense(otlg5_c, 128, activation=tf.nn.relu)
      self.otlg5 = tf.compat.v1.layers.dense(otlg5_3, self.label_size[5])
      self.pred5 = tf.nn.sigmoid(self.otlg5)
      '''
      #Ontology layer6 for label6 classifying
      otlg6_0 = tf.layers.dense(base0, 256, activation=tf.nn.relu)
      otlg6_1 = tf.layers.dense(otlg6_0, 128, activation=tf.nn.relu)
      otlg6_2 = tf.layers.dense(otlg6_1, 64, activation=tf.nn.relu)
      otlg6_c = tf.concat([otlg6_2, otlg5_3], axis=1)
      otlg6_3 = tf.layers.dense(otlg6_c, 128, activation=tf.nn.relu)
      self.otlg6 = tf.layers.dense(otlg6_3, self.label_size[6])
      self.pred6 = tf.nn.sigmoid(self.otlg6)
      '''
      self.y_pred = tf.concat([self.pred0, self.pred1, self.pred2, self.pred3, self.pred4, self.pred5], axis=1)
      self.logits = tf.concat([self.otlg0, self.otlg1, self.otlg2, self.otlg3, self.otlg4, self.otlg5], axis=1)
      self.y_true = tf.concat([self.y_0, self.y_1, self.y_2, self.y_3, self.y_4, self.y_5], axis=1)

      #train options
      if(self.is_training):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #set eps to avoid zero in the loss calculating process
        eps=1e-6
        self.losses =tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_true, logits = self.logits))
        #training
        self.lr = tf.Variable(self.lr, trainable=False)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        grads = self.optimizer.compute_gradients(self.losses)
        self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')
        
      #initialize variables
      self.init = tf.compat.v1.global_variables_initializer()
      t_vars = tf.compat.v1.trainable_variables()
      self.assign_ops = {}
      for var in t_vars:
        #if var.name.startswith('conv_vae'):
        pshape = var.get_shape()
        pl = tf.compat.v1.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
        assign_op = var.assign(pl)
        self.assign_ops[var] = (assign_op, pl)

  def init_session(self):
    self.sess = tf.compat.v1.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    self.sess.close()
  def encode(self, x):
    return self.sess.run(self.z, feed_dict={self.x: x})
  def encode_mu_logvar(self, x):
    (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.x: x})
    return mu, logvar
  def decode(self, z):
    return self.sess.run(self.y, feed_dict={self.z: z})
  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.compat.v1.trainable_variables()
      for var in t_vars:
        #if var.name.startswith('conv_vae'):
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
    return rparam
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.compat.v1.trainable_variables()
      idx = 0
      for var in t_vars:
        #if var.name.startswith('conv_vae'):
        pshape = tuple(var.get_shape().as_list())
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op, pl = self.assign_ops[var]
        self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
        idx += 1
  def load_json(self, jsonfile='vae.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)
  def save_json(self, jsonfile='vae.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))
  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)
  def save_model(self, model_save_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'vae')
    tf.compat.v1.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, 0) # just keep one
  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.compat.v1.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

