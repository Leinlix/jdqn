import random
import tensorflow as tf
import numpy as np
from networks.cnn import CNN

class judgeExperience(object):
    def __init__(self, data_format, batch_size, history_length, memory_size, observation_dims):
        self.data_format = data_format
        self.batch_size = batch_size
        self.history_length = history_length
        self.memory_size = memory_size

        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.observations = np.empty([self.memory_size] + observation_dims, dtype=np.uint8)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)

        self.tempAct = np.empty(self.memory_size,dtype=np.uint8)
        self.tempObs = np.empty([self.memory_size] + observation_dims, dtype=np.uint8)
        self.tempCount = 0

        self.posAct = np.empty(self.memory_size, dtype=np.uint8)
        self.posObs = np.empty([self.memory_size] + observation_dims, dtype=np.uint8)
        self.prestatesPos = np.empty([self.memory_size,self.history_length] + observation_dims,dtype=np.float16)
        self.posCount = 0

        self.negAct = np.empty(self.memory_size, dtype=np.uint8)
        self.negObs = np.empty([self.memory_size] + observation_dims, dtype=np.uint8)
        self.prestatesNeg = np.empty([self.memory_size,self.history_length] + observation_dims,dtype=np.float16)
        self.negCount = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty([self.batch_size, self.history_length] + observation_dims, dtype=np.float16)
        self.poststates = np.empty([self.batch_size, self.history_length] + observation_dims, dtype=np.float16)

        self.count = 0
        self.current = 0
        self.tempR = 0

    def add(self, observation, reward, action, terminal):
        self.actions[self.current] = action
        self.tempR = self.tempR + reward
        self.tempAct[self.tempCount] = action
        self.rewards[self.current] = reward
        self.observations[self.current, ...] = observation
        self.tempObs[self.tempCount] = observation
        self.terminals[self.current] = terminal
        self.tempCount = self.tempCount + 1
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size
        if terminal==True:
            #add pos
            if self.tempR >200:
                for i in range(self.tempCount):
                    self.posAct[self.posCount] = self.tempAct[i]
                    self.posObs[self.posCount] = self.tempObs[i]
                    self.posCount = (self.posCount +1)%self.memory_size
                    if self.posCount>3 :
                        idxp = [self.posCount-4,self.posCount-3,self.posCount-2,self.posCount-1]
                    elif self.posCount>2:
                        idxp =[0, self.posCount - 3, self.posCount - 2, self.posCount-1]
                    elif self.posCount>1:
                        idxp = [0,0, self.posCount - 2, self.posCount-1]
                    else:
                        idxp = [0,0,0,0]
                    self.prestatesPos[self.posCount-1] = self.posObs[idxp,...] 
            #add neg
            elif self.tempR<50:
                for i in range(self.tempCount):
                    self.negAct[self.negCount] = self.tempAct[i]
                    self.negObs[self.negCount] = self.tempObs[i]
                    self.negCount = (self.negCount +1)%self.memory_size
                    if self.negCount>3 :
                        idxn=[self.negCount-4,self.negCount-3,self.negCount-2,self.negCount-1]
                    elif self.negCount>2:
                        idxn=[0, self.negCount - 3, self.negCount - 2, self.negCount-1]
                    elif self.negCount>1:
                        idxn=[0,0, self.negCount - 2, self.negCount-1]
                    else:
                        idxn = [0,0,0,0]
                    self.prestatesNeg[self.negCount-1] = self.negObs[idxn,...]
            if terminal:
                self.tempCount = 0
                self.tempR = 0

    def sample(self):
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(self.history_length, self.count - 1)
                if index >= self.current and index - self.history_length < self.current:
                    continue
                if self.terminals[(index - self.history_length):index].any():
                    continue
                break

            self.prestates[len(indexes), ...] = self.retreive(index - 1)
            self.poststates[len(indexes), ...] = self.retreive(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        if self.data_format == 'NHWC' and len(self.prestates.shape) == 4:
            return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
                   rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
        else:
            return self.prestates, actions, rewards, self.poststates, terminals

    def retreive(self, index):
        index = index % self.count
        if index >= self.history_length - 1:
            return self.observations[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.observations[indexes, ...]


    def getJtrainData(self):
        if self.posCount==0 or self.negCount == 0:
            return [],[],[],[]
        else:
            pindexs = []
            nindexs = []
            for i in range(32):
                pindexs.append(random.randint(0,self.posCount))
                nindexs.append(random.randint(0,self.negCount))
            if self.data_format == 'NHWC' and len(self.prestates.shape) == 4:
                return np.transpose(self.prestatesPos[pindexs], (0, 2, 3, 1)),np.array(self.posAct[pindexs]).reshape((32,1)),np.transpose(self.prestatesNeg[nindexs],(0,2,3,1)),np.array(self.negAct[nindexs]).reshape((32,1))
            else:
                return np.array(self.prestatesPos[pindexs]),np.array(self.posAct[pindexs]).reshape((32,1)),np.array(self.prestatesNeg[nindexs]),np.array(self.negAct[nindexs]).reshape((32,1))

