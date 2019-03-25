import os
import pickle 
from inter_utils import tprint
import numpy as np
from data_utils import load_dataset_input
from distance.get_dis import edit_distance

def load_pickle(path):
    with open(path,'rb') as files:
        return pickle.load(files)

def dump_pickle(path,thing):
    with open(path,'wb') as files:
        pickle.dump(thing,files)

'''
    first we need to load all the distance data from scratch
'''

pos2idx = load_pickle('./temp/pos2idx.bin')

path_patch_train_train = './patch_train_train.dat'
if not os.path.exists(path_patch_train_train):
    patch = dict()
else:
    patch = load_pickle(path_patch_train_train)

def seqpos2id(seq):
    return list(map(lambda x:pos2idx[x[7]],seq))

class final_dataset:
    '''
        This is the final interface for loading dataset, which is modified on load_data_input in data_utils.
        
        The old result is a 3D lists, with shape : #sentence * #words * #informations

        The new result is a list of dictionary, with every dictionary having following structures:
        {
            'ori_sentence': #words * #informations
            'aux_sentences': #sentence * #words *#informations
        }
    '''
    def __init__(self,path_train,path_dev,path_test,aug_num,distance_train,distance_dev,distance_test):
        self.path_train = path_train
        self.path_dev = path_dev
        self.path_test = path_test
        self.aug_num = aug_num
        self.old_train = load_dataset_input(self.path_train)
        self.old_dev = load_dataset_input(self.path_dev)
        self.old_test = load_dataset_input(self.path_test)
        self.train_train = load_pickle(distance_train)
        self.dev_train = load_pickle(distance_dev)
        self.test_train = load_pickle(distance_test)

        tprint('load is done, begin transfer data to new format')
        self._load_new()
        tprint('dataset load done')

    def _load_new(self):
        train = []
        total_false = 0
        for i,ele in enumerate(self.old_train):
            cur_train = {}
            cur_train['ori_sentence'] = ele
            cur_train['aux_sentences'] = []
            sort_list = self.train_train[i]
            sort_id = 0
            sort_id_list = []
            while len(cur_train['aux_sentences']) < self.aug_num:
                sort_sentence_id = sort_list[sort_id]
                if self._get_actual_sentence_from_train(sort_sentence_id) != self._get_actual_sentence_from_train(i):
                    cur_train['aux_sentences'].append(self.old_train[sort_sentence_id].copy())
                    sort_id_list.append(sort_list[sort_id])
                sort_id += 1
                if sort_id >= len(sort_list):
                    total_false += 1
                    raise ValueError('Need more sentences id!')
            
            train.append(cur_train)
        
        dev = list(map(lambda x:{
            'ori_sentence':x[0].copy(),
            'aux_sentences':list(map(lambda y:self.old_train[y].copy(),x[1][:self.aug_num]))
            },zip(self.old_dev,self.dev_train)))
        
        test = list(map(lambda x:{
            'ori_sentence':x[0].copy(),
            'aux_sentences':list(map(lambda y:self.old_train[y].copy(),x[1][:self.aug_num]))
            },zip(self.old_test,self.test_train)))

        self.train = train
        self.dev = dev
        self.test = test

    def _get_actual_sentence_from_train(self,id):
        return ' '.join(list(map(lambda x:x[5],self.old_train[id])))
