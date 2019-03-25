# -*- coding: utf-8 -*-

from distance.get_dis import *
from inter_utils import tprint
import pickle
import numpy as np
import sys

train_path = ''
test_path = ''

def pickle_save(thing,path):
    with open(path,'wb') as files:
        pickle.dump(thing,files)

def pickle_load(path):
    with open(path,'rb') as files:
        return pickle.load(files)

def init_sent(id):
    tmp = dict()
    tmp['word'] = []
    tmp['pos'] = []
    tmp['lemma'] = []
    tmp['label'] = []
    tmp['flag'] = []
    tmp['pred'] = ''
    tmp['seqlen'] = 0
    tmp['id'] = id
    return tmp

def get_sentences(path):
    with open(path,'r') as files:
        content = files.readlines()
    raw_sentences = []
    curRawSentence = []
    for line in content:
        items = line.strip().split('\t')
        if len(items) <= 1:
            if len(curRawSentence) != 0:
                raw_sentences.append(curRawSentence)
                curRawSentence = []
        else:
            curRawSentence.append(items)
    
    if len(curRawSentence) != 0:
        raw_sentences.append(curRawSentence)
    
    final_sentences = []
    curid = 0
    for rawSent in raw_sentences:
        cur_predid = 0
        for item in rawSent:
            if item[12] == 'Y':
                cur_sent = init_sent(curid)
                for it in rawSent:
                    cur_sent['word'].append(it[1])
                    cur_sent['lemma'].append(it[3])
                    cur_sent['pos'].append(it[5])
                    cur_sent['label'].append(it[14+cur_predid])
                    cur_sent['flag'].append(1 if it[0]==item[0] else 0)
                cur_predid += 1
                cur_sent['seqlen'] = len(cur_sent['word'])
                final_sentences.append(cur_sent)
                curid += 1

    return final_sentences

def get_seq_id(input_seq):
    pos2id = pickle_load('./temp/pos2idx.bin')
    return list(map(lambda x:pos2id['<UNK>'] if x not in pos2id else pos2id[x],input_seq))

def get_max_similar(sent1,sent2):
    pos_seq_1 = []
    for sent in sent1:
        pos_seq_1.extend(get_seq_id(sent['pos']))
    pos_seq_2 = []
    for sent in sent2:
        pos_seq_2.extend(get_seq_id(sent['pos']))
    len_seq_1 = list(map(lambda x:x['seqlen'],sent1))
    len_seq_2 = list(map(lambda x:x['seqlen'],sent2))
    test = get_ed_btwn_lists(pos_seq_1,pos_seq_2,len_seq_1,len_seq_2)
    return test

def analyse_train_file(path_to_data,result_data_path):
    train_train_data = pickle_load(path_to_data)
    train_train_data = np.array(train_train_data)
    train_train_data = train_train_data.reshape([-1,179014])
    train_tmp = []
    for ele in train_train_data:
        res_tmp = []
        for i,e in enumerate(ele):
            res_tmp.append([i,e])
        res_tmp = sorted(res_tmp,key = lambda x:x[1])[:300]
        train_tmp.append([ele[0] for ele in res_tmp])
    pickle_save(train_tmp,result_data_path )
     
if __name__ == '__main__':
    target_name = sys.argv[1]
    if target_name == 'train':
        train = get_sentences('./data/train.txt')
        # train file is so large, we need to split them to 18 file for process
        target_2 = get_sentences('./data/train.txt')
        cur_idx = 0
        while cur_idx < 179014:
            tprint('begin processing %d/18' % (cur_idx // 10000 + 1))
            target = target_2[cur_idx:cur_idx+10000]
            result = get_max_similar(target, train)
            pickle_save(result, './temp/train-train-%d-result.bin' % (cur_idx))
            analyse_train_file('./temp/train-train-%d-result.bin' % (cur_idx), './temp/train-train-%d-part.bin' % (cur_idx))
            cur_idx += 10000
        total = []
        for i in range(18):
            total.extend(pickle_load('./temp/train-train-%d-part.bin' % (i)))
        pickle_save(total, './temp/train_train.bin')

    elif target_name == 'dev':
        target = get_sentences('./data/dev.txt')
        train = get_sentences('./data/train.txt')
        result = get_max_similar(target, train)
        pickle_save(result, './temp/dev-train-result.bin')
        analyse_train_file('./temp/dev-train-result.bin','./temp/train_dev.bin')
    
    elif target_name == 'test':
        target = get_sentences('./data/test.txt')
        train = get_sentences('./data/train.txt')
        result = get_max_similar(target, train)
        pickle_save(result, './temp/test-train-result.bin')
        analyse_train_file('./temp/test-train-result.bin','./temp/train_test.bin')
    
    # used for testing
    elif target_name == 'ts3':
        target = get_sentences('./data/3.txt')
        train = get_sentences('./data/train.txt')
        result = get_max_similar(target, train)
        pickle_save(result, './temp/3-train-result.bin')
        analyse_train_file('./temp/3-train-result.bin','./temp/train_3.bin')