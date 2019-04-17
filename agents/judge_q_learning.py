import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random

from agents.agent import Agent
from networks.jnn import JNN
from agents.judgeExperience import judgeExperience
from logging import getLogger

logger = getLogger(__name__)


class JDQN(Agent):
    def __init__(self, sess, pred_network , env, stat, conf, target_network = None):
        super(JDQN, self).__init__(sess,pred_network,env,stat,conf,target_network = target_network)
        self.oP= np.ones((256))
        self.oN= np.zeros((256))
        self.j_t = 0
        self.judge_network = JNN(sess=sess,
                            data_format=conf.data_format,
                            history_length=conf.history_length,
                            observation_dims=conf.observation_dims,
                            output_size=256,
                            network_output_type='judgement',
                            network_header_type=conf.network_header_type,
                            name='JNN', trainable=True)
        with tf.variable_scope('judgeOp'):
            self.optP = tf.placeholder('float32', [256],name= 'optP')
            self.optN = tf.placeholder('float32', [256],name='optN')
            self.r = tf.placeholder('float32',[None,256],name='r')
            self.Jloss = self.r * (tf.log(tf.squared_difference(self.optP,self.judge_network.outputs)) - tf.log(tf.squared_difference(self.optN,self.judge_network.outputs)))

            self.judgeoptim = tf.train.RMSPropOptimizer(1e-5).minimize(loss=self.Jloss)


        self.experience = judgeExperience(conf.data_format,
        conf.batch_size, conf.history_length, conf.memory_size, conf.observation_dims)

        with tf.variable_scope('optimizer'):
            self.targets = tf.placeholder('float32', [None], name='target_q_t')
            self.actions = tf.placeholder('int64' , [None] , name='action')

            actions_one_hot = tf.one_hot(self.actions,self.env.action_size ,1.0, 0.0, name="action_one_hot")
            pred_q = tf.reduce_sum(self.pred_network.outputs * actions_one_hot, reduction_indices=1,
                                   name='q_acted')

            self.delta= self.targets - pred_q
            self.clipped_error = tf.where(tf.abs(self.delta)<1.0,
                                          0.5*tf.square(self.delta),
                                          tf.abs(self.delta)-0.5,name='clipped_error')

            self.loss = tf.reduce_mean(self.clipped_error,name='loss')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.stat.t_op,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True
                                               ))

            optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op,momentum=0.95,epsilon=0.01
            )
            if self.max_grad_norm!=None:
                grads_and_vars  = optimizer.compute_gradients(self.loss)
                for idx,(grad,var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm),var)
                self.optim = optimizer.apply_gradients(grads_and_vars)
            else:
                self.optim = optimizer.minimize(self.loss)

    def observe(self,observation,reward,action,terminal):
        reward = max(self.min_r, min(self.max_r, reward))

        self.history.add(observation)
        self.experience.add(observation, reward, action, terminal)
        # we train judge network here
        posS,posA,negS,negA = self.experience.getJtrainData()

        self.j_t = self.j_t +1
        #both has value
        if self.j_t%64==0:
            if len(posS)!=0 and len(negS)!=0:
                _1,jloss1 = self.sess.run([self.judgeoptim,self.Jloss],feed_dict ={
                    self.optP:self.opP,
                    self.optN:self.opN,
                    self.r:np.ones((len(posS),256)),
                    self.judge_network.inputa:posA,
                    self.judge_network.inputs:posS
                })
                _2,jloss2 = self.sess.run([self.judgeoptim,self.Jloss],feed_dict ={
                    self.optP:self.opP,
                    self.optN:self.opN,
                    self.r:np.full((len(negA),256),-1),
                    self.judge_network.inputs:negS,
                    self.judge_network.inputa:negA
                })

        result = [], 0, False

        if self.t > self.t_learn_start:
            if self.t % self.t_train_freq == 0:
                result = self.q_learning_minibatch()

            if self.t % self.t_target_q_update_freq == self.t_target_q_update_freq - 1:
                self.update_target_q_network()

        return result
    def q_learning_minibatch(self):
        if self.experience.count < self.history_length:
            return [],0,False
        else:
            s_t , aciton , reward , s_t_plus_1, terminal = self.experience.sample()

        terminal = np.array(terminal) + 0.



        judgement = np.zeros((self.pred_network.output_size,self.experience.batch_size))
        at = 0
        for i in range(self.pred_network.output_size):
            a = np.array([at]*self.experience.batch_size)
            judgement[i]= self.getPseudoRs(s_t_plus_1,a)
            at =at+1
        judgement = np.transpose(judgement)
        mqt1 = self.sess.run(self.target_network.outputs,{self.target_network.inputs:s_t_plus_1})
        max_q_t_plus_1 = np.zeros((32))
        judge = np.zeros((32))
        for i in range(32):
          max_q_t_plus_1[i]=mqt1[i][np.argmax(judgement[i])]
        target_q_t = (1. - terminal) * self.discount_r * max_q_t_plus_1 + reward 
        _,q_t, loss = self.sess.run([self.optim,self.pred_network.outputs,self.loss],{
                                    self.targets:target_q_t,
                                    self.actions:aciton,
                                    self.pred_network.inputs:s_t
        })

        return q_t,loss,True

    def getPseudoR(self,s,a):
        a = np.array([a]).reshape((1,1))
        r = self.sess.run([self.judge_network.outputs],feed_dict = {
            self.judge_network.inputa:a,
            self.judge_network.inputs:s
        })
        od = cal_dist(r,self.oP)
        on = cal_dist(r,self.oN)

        return np.exp((-od+on)/(od+on))

    def getPseudoRs(self,s,a):
        a= np.array([a]).reshape((-1,1))
        r = self.sess.run([self.judge_network.outputs],feed_dict={
            self.judge_network.inputs:s,
            self.judge_network.inputa:a
        })
        r = np.array(r).reshape((-1,256))
        d = np.zeros((r.shape[0]))
        for i in range(r.shape[0]):
            od = cal_dist(r[i],self.oP)
            on = cal_dist(r[i],self.oN)
            d[i] = np.exp((-od+on)/(od+on))
        return d 


    def predict(self, s_t, ep):
        action = random.randrange(self.env.action_size)     
        if random.random() < ep:
            if self.j_t>5000+20000:
                 z = np.zeros((self.env.action_size))
                 for i in range(self.env.action_size):
                      z[i] = self.getPseudoRs([s_t],i)
                 action = np.argmax(z)
           
        else:
            action = self.pred_network.calc_actions([s_t])[0]
        return action


def cal_dist(x,y):
    x=np.array(x).reshape((256))
    y=np.array(y).reshape((256))
    return np.linalg.norm(x-y)
