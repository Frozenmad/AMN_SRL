from __future__ import print_function
import datetime
from data_utils import _PAD_,_UNK_,_ROOT_,_NUM_
import math
import numpy as np
import random

save_file = None

def tprint(str,end='\n',quiet = False):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not quiet:
        print(now+" "+str,end=end)
    if save_file is not None:
        with open(save_file,'a') as files:
            files.write(now+" " + str + end)
    return now+ " " + str + end

def quick_pad_seq(seq_len, max_len, pad_value):
    return seq_len + [pad_value for _ in range(max_len - len(seq_len))]

def pad_batch(batch_data, batch_size, pad_int):
    if len(batch_data) < batch_size:
        batch_data += [[pad_int]] * (batch_size - len(batch_data))
    max_length = max([len(item) for item in batch_data])
    return [item + [pad_int]*(max_length-len(item)) for item in batch_data]

def pad_batch_by_max_length(batch_data, batch_size, pad_int, max_length):
    if len(batch_data) < batch_size:
        batch_data += [[pad_int]] * (batch_size - len(batch_data))
    return [item + [pad_int]*(max_length-len(item)) for item in batch_data]

def pad_batch_new_max_length(batch_data, batch_size, pad_int, max_length):
    # 这个函数的输入是[B*B'*L]的元素，我们要把最后的L补全成max_length, 把B补全成batch_size.
    middle_size = len(batch_data[0])
    if len(batch_data) < batch_size:
        batch_data += [[[pad_int] for _ in range(middle_size)] for _ in range(batch_size - len(batch_data))]
    return [[ item + [pad_int for _ in range(max_length - len(item))] for item in batch_sentence] for batch_sentence in batch_data]


def get_batch(input_data, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx):

    for batch_i in range(math.ceil(len(input_data)/batch_size)):
        
        start_i = batch_i * batch_size
        end_i = start_i + batch_size
        if end_i > len(input_data):
            end_i = len(input_data)

        data_batch = input_data[start_i:end_i]

        sentence_id_batch = [sentence[0][0] for sentence in data_batch]
        predicate_id_batch = [sentence[0][1] for sentence in data_batch]
        setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
        id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]

        seq_len_batch = [len(sentence) for sentence in data_batch]

        flag_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]
        pad_flag_batch = np.array(pad_batch(flag_batch, batch_size, 0),dtype=int)

        text_batch = [[item[5] for item in sentence] for sentence in data_batch]
        if len(text_batch) < batch_size:
            text_batch += [[_PAD_]] * (batch_size - len(text_batch))

        word_batch = [[word2idx.get(item[5],word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_word_batch = np.array(pad_batch(word_batch, batch_size, word2idx[_PAD_]))

        lemma_batch = [[lemma2idx.get(item[6],lemma2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_lemma_batch = np.array(pad_batch(lemma_batch, batch_size, lemma2idx[_PAD_]))

        pos_batch = [[pos2idx.get(item[7],pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pos_batch = np.array(pad_batch(pos_batch, batch_size, pos2idx[_PAD_]))

        argument_batch = [[argument2idx.get(item[9],argument2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_argument_batch = np.array(pad_batch(argument_batch, batch_size, argument2idx[_PAD_]))
        flat_argument_batch = np.array([item for line in pad_argument_batch for item in line])

        pretrain_word_batch = [[pretrain2idx.get(item[5],pretrain2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))

        batch = {
            "sentence_id":sentence_id_batch,
            "predicate_id":predicate_id_batch,
            "word_id":id_batch,
            "flag":pad_flag_batch,
            "word":pad_word_batch,
            "lemma":pad_lemma_batch,
            "pos":pad_pos_batch,
            "pretrain":pad_pretrain_word_batch,
            "argument":pad_argument_batch,
            "flat_argument":flat_argument_batch,
            "batch_size":pad_argument_batch.shape[0],
            "pad_seq_len":pad_argument_batch.shape[1],
            "text":text_batch,
            "sentence_len":setence_len_batch,
            "seq_len":seq_len_batch
        }

        yield batch

def get_all_batch(input_data,word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx,batch_size=-1):
    
    n_batch_size = len(input_data)
    data_batch = input_data[:]
    sentence_id_batch = [sentence[0][0] for sentence in data_batch]
    predicate_id_batch = [sentence[0][1] for sentence in data_batch]
    setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
    id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]

    seq_len_batch = [len(sentence) for sentence in data_batch]

    flag_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]
    pad_flag_batch = np.array(pad_batch(flag_batch, n_batch_size, 0),dtype=int)

    text_batch = [[item[5] for item in sentence] for sentence in data_batch]
    if len(text_batch) < n_batch_size:
        text_batch += [[_PAD_]] * (n_batch_size - len(text_batch))

    word_batch = [[word2idx.get(item[5],word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
    pad_word_batch = np.array(pad_batch(word_batch, n_batch_size, word2idx[_PAD_]))

    lemma_batch = [[lemma2idx.get(item[6],lemma2idx[_UNK_]) for item in sentence] for sentence in data_batch]
    pad_lemma_batch = np.array(pad_batch(lemma_batch, n_batch_size, lemma2idx[_PAD_]))

    pos_batch = [[pos2idx.get(item[7],pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
    pad_pos_batch = np.array(pad_batch(pos_batch, n_batch_size, pos2idx[_PAD_]))

    argument_batch = [[argument2idx.get(item[9],argument2idx[_UNK_]) for item in sentence] for sentence in data_batch]
    pad_argument_batch = np.array(pad_batch(argument_batch, n_batch_size, argument2idx[_PAD_]))
    flat_argument_batch = np.array([item for line in pad_argument_batch for item in line])

    pretrain_word_batch = [[pretrain2idx.get(item[5],pretrain2idx[_UNK_]) for item in sentence] for sentence in data_batch]
    pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, n_batch_size, pretrain2idx[_PAD_]))

    batch = {
        "sentence_id":sentence_id_batch,
        "predicate_id":predicate_id_batch,
        "word_id":id_batch,
        "flag":pad_flag_batch,
        "word":pad_word_batch,
        "lemma":pad_lemma_batch,
        "pos":pad_pos_batch,
        "pretrain":pad_pretrain_word_batch,
        "argument":pad_argument_batch,
        "flat_argument":flat_argument_batch,
        "batch_size":pad_argument_batch.shape[0],
        "pad_seq_len":pad_argument_batch.shape[1],
        "text":text_batch,
        "sentence_len":setence_len_batch,
        "seq_len":seq_len_batch
    }

    return batch

def get_batch_equal_length(input_data, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx):

    max_length = max(list(map(lambda x:len(x),input_data)))
    for batch_i in range(math.ceil(len(input_data)/batch_size)):

        start_i = batch_i * batch_size
        end_i = start_i + batch_size
        if end_i > len(input_data):
            end_i = len(input_data)

        data_batch = input_data[start_i:end_i]

        sentence_id_batch = [sentence[0][0] for sentence in data_batch]
        predicate_id_batch = [sentence[0][1] for sentence in data_batch]
        setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
        id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]

        seq_len_batch = [len(sentence) for sentence in data_batch]

        flag_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]
        pad_flag_batch = np.array(pad_batch_by_max_length(flag_batch, batch_size, 0,max_length),dtype=int)

        text_batch = [[item[5] for item in sentence] for sentence in data_batch]
        pad_text_batch = np.array(pad_batch_by_max_length(text_batch,batch_size,_PAD_,max_length))
        if len(text_batch) < batch_size:
            text_batch += [[_PAD_]] * (batch_size - len(text_batch))


        word_batch = [[word2idx.get(item[5],word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_word_batch = np.array(pad_batch_by_max_length(word_batch, batch_size, word2idx[_PAD_],max_length))

        lemma_batch = [[lemma2idx.get(item[6],lemma2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_lemma_batch = np.array(pad_batch_by_max_length(lemma_batch, batch_size, lemma2idx[_PAD_],max_length))

        pos_batch = [[pos2idx.get(item[7],pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pos_batch = np.array(pad_batch_by_max_length(pos_batch, batch_size, pos2idx[_PAD_],max_length))

        argument_batch = [[argument2idx.get(item[9],argument2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_argument_batch = np.array(pad_batch_by_max_length(argument_batch, batch_size, argument2idx[_PAD_],max_length))
        flat_argument_batch = np.array([item for line in pad_argument_batch for item in line])

        pretrain_word_batch = [[pretrain2idx.get(item[5],pretrain2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pretrain_word_batch = np.array(pad_batch_by_max_length(pretrain_word_batch, batch_size, pretrain2idx[_PAD_],max_length))

        batch = {
            "sentence_id":sentence_id_batch,
            "predicate_id":predicate_id_batch,
            "word_id":id_batch,
            "flag":pad_flag_batch,
            "word":pad_word_batch,
            "lemma":pad_lemma_batch,
            "pos":pad_pos_batch,
            "pretrain":pad_pretrain_word_batch,
            "argument":pad_argument_batch,
            "flat_argument":flat_argument_batch,
            "batch_size":pad_argument_batch.shape[0],
            "pad_seq_len":pad_argument_batch.shape[1],
            "text":pad_text_batch,
            "sentence_len":setence_len_batch,
            "seq_len":seq_len_batch
        }

        yield batch

def predicate_get_batch(input_data, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, label2idx, rand_flag = False):

    if rand_flag:
        random.shuffle(input_data)

    for batch_i in range(math.ceil(len(input_data)/batch_size)):
        
        start_i = batch_i * batch_size
        end_i = start_i + batch_size
        if end_i > len(input_data):
            end_i = len(input_data)

        raw_data_batch = input_data[start_i:end_i]

        # 我们需要得到的是两个batch，最终返回一个dict。原始batch应该具有[B*L]的信息，辅助batch应该具有[B * B' *L]的信息。
        data_batch = raw_data_batch

        sentence_id_batch = [sentence[0][0] for sentence in data_batch]
        predicate_id_batch = [sentence[0][1] for sentence in data_batch]
        setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
        pred_id_batch = []
        for sentence in data_batch:
            for idx, line in enumerate(sentence):
                if line[4] == '1':
                    pred_id_batch.append(idx)
                    break
        assert(len(pred_id_batch) == len(data_batch))
        label_id_batch = [label2idx[sentence[0][8]] for sentence in data_batch]
        id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]

        seq_len_batch = [len(sentence) for sentence in data_batch]

        max_seq_len = max(seq_len_batch)

        flag_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]
        pad_flag_batch = [quick_pad_seq(ele, max_seq_len, 0) for ele in flag_batch]
        pad_flag_batch = np.array(pad_flag_batch)
        # pad_flag_batch = np.array(pad_batch(flag_batch, batch_size, 0),dtype=int)

        text_batch = [[item[5] for item in sentence] for sentence in data_batch]

        word_batch = [[word2idx.get(item[5],word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_word_batch = [quick_pad_seq(ele, max_seq_len, word2idx[_PAD_]) for ele in word_batch]
        pad_word_batch = np.array(pad_word_batch)
        # pad_word_batch = np.array(pad_batch(word_batch, batch_size, word2idx[_PAD_]))

        lemma_batch = [[lemma2idx.get(item[6],lemma2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_lemma_batch = [quick_pad_seq(ele, max_seq_len, lemma2idx[_PAD_]) for ele in lemma_batch]
        pad_lemma_batch = np.array(pad_lemma_batch)
        # pad_lemma_batch = np.array(pad_batch(lemma_batch, batch_size, lemma2idx[_PAD_]))

        pos_batch = [[pos2idx.get(item[7],pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pos_batch = [quick_pad_seq(ele, max_seq_len, pos2idx[_PAD_]) for ele in pos_batch]
        pad_pos_batch = np.array(pad_pos_batch)
        # pad_pos_batch = np.array(pad_batch(pos_batch, batch_size, pos2idx[_PAD_]))

        pretrain_word_batch = [[pretrain2idx.get(item[5],pretrain2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pretrain_word_batch = [quick_pad_seq(ele, max_seq_len, pretrain2idx[_PAD_]) for ele in pretrain_word_batch]
        pad_pretrain_word_batch = np.array(pad_pretrain_word_batch)
        # pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))

        batch = {
            "sentence_id":sentence_id_batch,
            "predicate_id":predicate_id_batch,
            "word_id":id_batch,
            "flag":pad_flag_batch,
            "word":pad_word_batch,
            "lemma":pad_lemma_batch,
            "pos":pad_pos_batch,
            "pretrain":pad_pretrain_word_batch,
            "batch_size":pad_pos_batch.shape[0],
            "pad_seq_len":max_seq_len,
            "text":text_batch,
            "sentence_len":setence_len_batch,
            "seq_len":seq_len_batch,
            'pred_id': pred_id_batch,
            'label_id': label_id_batch
        }

        yield batch


def get_all_batch(input_data, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx, deprel2idx = None):
    raw_data_batch = input_data

    # 我们需要得到的是两个batch，最终返回一个dict。原始batch应该具有[B*L]的信息，辅助batch应该具有[B * B' *L]的信息。
    data_batch = list(map(lambda x: x['ori_sentence'], raw_data_batch))

    sentence_id_batch = [sentence[0][0] for sentence in data_batch]
    predicate_id_batch = [sentence[0][1] for sentence in data_batch]
    setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
    id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]

    seq_len_batch = [len(sentence) for sentence in data_batch]

    flag_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]
    pad_flag_batch = np.array(pad_batch(flag_batch, batch_size, 0), dtype=int)

    text_batch = [[item[5] for item in sentence] for sentence in data_batch]
    if len(text_batch) < batch_size:
        text_batch += [[_PAD_]] * (batch_size - len(text_batch))

    word_batch = [[word2idx.get(item[5], word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
    pad_word_batch = np.array(pad_batch(word_batch, batch_size, word2idx[_PAD_]))

    lemma_batch = [[lemma2idx.get(item[6], lemma2idx[_UNK_]) for item in sentence] for sentence in data_batch]
    pad_lemma_batch = np.array(pad_batch(lemma_batch, batch_size, lemma2idx[_PAD_]))

    pos_batch = [[pos2idx.get(item[7], pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
    pad_pos_batch = np.array(pad_batch(pos_batch, batch_size, pos2idx[_PAD_]))

    argument_batch = [[argument2idx.get(item[9], argument2idx[_UNK_]) for item in sentence] for sentence in
                      data_batch]
    pad_argument_batch = np.array(pad_batch(argument_batch, batch_size, argument2idx[_PAD_]))
    flat_argument_batch = np.array([item for line in pad_argument_batch for item in line])

    pretrain_word_batch = [[pretrain2idx.get(item[5], pretrain2idx[_UNK_]) for item in sentence] for sentence in
                           data_batch]
    pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))

    if deprel2idx is not None:
        deprel_batch = [[deprel2idx.get(item[8], deprel2idx[_UNK_]) for item in sentence] for sentence in
                        data_batch]
        pad_deprel_batch = np.array(pad_batch(deprel_batch, batch_size, argument2idx[_PAD_]))
    else:
        deprel_batch = None
        pad_deprel_batch = None

    batch = {
        "sentence_id": sentence_id_batch,
        "predicate_id": predicate_id_batch,
        "word_id": id_batch,
        "flag": pad_flag_batch,
        "word": pad_word_batch,
        "lemma": pad_lemma_batch,
        "pos": pad_pos_batch,
        "pretrain": pad_pretrain_word_batch,
        "argument": pad_argument_batch,
        "flat_argument": flat_argument_batch,
        "deprel": pad_deprel_batch,
        "batch_size": pad_argument_batch.shape[0],
        "pad_seq_len": pad_argument_batch.shape[1],
        "text": text_batch,
        "sentence_len": setence_len_batch,
        "seq_len": seq_len_batch
    }

    # 接下来便需要我们得到[B * B' *L]的信息

    aug_data_batch = list(map(lambda x: x['aux_sentences'], raw_data_batch))
    middle_size = len(aug_data_batch[0])
    max_length = max(list(
        map(lambda batch_sentence: max(list(map(lambda sentence: len(sentence), batch_sentence))), aug_data_batch)))
    aug_sentence_id_batch = [[sentence[0][0] for sentence in batch_sentence] for batch_sentence in aug_data_batch]
    aug_predicate_id_batch = [[sentence[0][1] for sentence in batch_sentence] for batch_sentence in aug_data_batch]
    aug_sentence_len_batch = [[int(sentence[0][2]) for sentence in batch_sentence] for batch_sentence in
                              aug_data_batch]

    aug_id_batch = [[[int(item[3]) for item in sentence] for sentence in batch_sentence] for batch_sentence in
                    aug_data_batch]

    aug_flag_batch = [[[int(item[4]) for item in sentence] for sentence in batch_sentence] for batch_sentence in
                      aug_data_batch]
    aug_pad_flag_batch = np.array(pad_batch_new_max_length(aug_flag_batch, batch_size, 0, max_length), dtype=int)

    aug_text_batch = [[[item[5] for item in sentence] for sentence in batch_sentence] for batch_sentence in
                      aug_data_batch]
    if len(aug_text_batch) < batch_size:
        aug_text_batch += [[[_PAD_] for _ in range(middle_size)] for _ in range(batch_size - len(aug_text_batch))]

    aug_word_batch = [[[word2idx.get(item[5], word2idx[_UNK_]) for item in sentence] for sentence in batch_sentence]
                      for batch_sentence in aug_data_batch]
    aug_pad_word_batch = np.array(pad_batch_new_max_length(aug_word_batch, batch_size, word2idx[_PAD_], max_length))

    aug_lemma_batch = [
        [[lemma2idx.get(item[6], lemma2idx[_UNK_]) for item in sentence] for sentence in batch_sentence] for
        batch_sentence in aug_data_batch]
    aug_pad_lemma_batch = np.array(
        pad_batch_new_max_length(aug_lemma_batch, batch_size, lemma2idx[_PAD_], max_length))

    aug_pos_batch = [[[pos2idx.get(item[7], pos2idx[_UNK_]) for item in sentence] for sentence in batch_sentence]
                     for batch_sentence in aug_data_batch]
    aug_pad_pos_batch = np.array(pad_batch_new_max_length(aug_pos_batch, batch_size, pos2idx[_PAD_], max_length))

    aug_argument_batch = [
        [[argument2idx.get(item[9], argument2idx[_UNK_]) for item in sentence] for sentence in batch_sentence] for
        batch_sentence in aug_data_batch]
    aug_pad_argument_batch = np.array(
        pad_batch_new_max_length(aug_argument_batch, batch_size, argument2idx[_PAD_], max_length))

    aug_pretrain_word_batch = [
        [[pretrain2idx.get(item[5], pretrain2idx[_UNK_]) for item in sentence] for sentence in batch_sentence] for
        batch_sentence in aug_data_batch]
    aug_pad_pretrain_word_batch = np.array(
        pad_batch_new_max_length(aug_pretrain_word_batch, batch_size, pretrain2idx[_PAD_], max_length))

    if deprel2idx is not None:
        aug_deprel_batch = [
            [[deprel2idx.get(item[8], deprel2idx[_UNK_]) for item in sentence] for sentence in batch_sentence] for
            batch_sentence in aug_data_batch]
        aug_pad_deprel_batch = np.array(
            pad_batch_new_max_length(aug_deprel_batch, batch_size, deprel2idx[_PAD_], max_length))
    else:
        aug_pad_deprel_batch = []

    aug_batch = {
        "flag": aug_pad_flag_batch,
        "word": aug_pad_word_batch,
        "argument": aug_pad_argument_batch,
        "lemma": aug_pad_lemma_batch,
        "pretrain": aug_pad_pretrain_word_batch,
        "pos": aug_pad_pos_batch,
        "deprel": aug_pad_deprel_batch,
        "text": aug_text_batch
    }

    res = {
        'ori': batch,
        'aug': aug_batch
    }

    return res

def final_get_batch(input_data, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx, deprel2idx = None):
    for batch_i in range(math.ceil(len(input_data) / batch_size)):
        start_i = batch_i * batch_size
        end_i = start_i + batch_size
        if end_i > len(input_data):
            end_i = len(input_data)

        raw_data_batch = input_data[start_i:end_i]

        # 我们需要得到的是两个batch，最终返回一个dict。原始batch应该具有[B*L]的信息，辅助batch应该具有[B * B' *L]的信息。
        data_batch = list(map(lambda x: x['ori_sentence'], raw_data_batch))

        sentence_id_batch = [sentence[0][0] for sentence in data_batch]
        predicate_id_batch = [sentence[0][1] for sentence in data_batch]
        setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
        id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]

        seq_len_batch = [len(sentence) for sentence in data_batch]

        flag_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]
        pad_flag_batch = np.array(pad_batch(flag_batch, batch_size, 0), dtype=int)

        text_batch = [[item[5] for item in sentence] for sentence in data_batch]
        if len(text_batch) < batch_size:
            text_batch += [[_PAD_]] * (batch_size - len(text_batch))

        word_batch = [[word2idx.get(item[5], word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_word_batch = np.array(pad_batch(word_batch, batch_size, word2idx[_PAD_]))

        lemma_batch = [[lemma2idx.get(item[6], lemma2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_lemma_batch = np.array(pad_batch(lemma_batch, batch_size, lemma2idx[_PAD_]))

        pos_batch = [[pos2idx.get(item[7], pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pos_batch = np.array(pad_batch(pos_batch, batch_size, pos2idx[_PAD_]))

        argument_batch = [[argument2idx.get(item[9], argument2idx[_UNK_]) for item in sentence] for sentence in
                          data_batch]
        pad_argument_batch = np.array(pad_batch(argument_batch, batch_size, argument2idx[_PAD_]))
        flat_argument_batch = np.array([item for line in pad_argument_batch for item in line])

        pretrain_word_batch = [[pretrain2idx.get(item[5], pretrain2idx[_UNK_]) for item in sentence] for sentence in
                               data_batch]
        pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))

        if deprel2idx is not None:
            deprel_batch = [[deprel2idx.get(item[8], deprel2idx[_UNK_]) for item in sentence] for sentence in
                            data_batch]
            pad_deprel_batch = np.array(pad_batch(deprel_batch, batch_size, argument2idx[_PAD_]))
        else:
            deprel_batch = None
            pad_deprel_batch = None

        batch = {
            "sentence_id": sentence_id_batch,
            "predicate_id": predicate_id_batch,
            "word_id": id_batch,
            "flag": pad_flag_batch,
            "word": pad_word_batch,
            "lemma": pad_lemma_batch,
            "pos": pad_pos_batch,
            "pretrain": pad_pretrain_word_batch,
            "argument": pad_argument_batch,
            "flat_argument": flat_argument_batch,
            "deprel": pad_deprel_batch,
            "batch_size": pad_argument_batch.shape[0],
            "pad_seq_len": pad_argument_batch.shape[1],
            "text": text_batch,
            "sentence_len": setence_len_batch,
            "seq_len": seq_len_batch
        }

        # 接下来便需要我们得到[B * B' *L]的信息

        aug_data_batch = list(map(lambda x: x['aux_sentences'], raw_data_batch))
        middle_size = len(aug_data_batch[0])
        max_length = max(list(
            map(lambda batch_sentence: max(list(map(lambda sentence: len(sentence), batch_sentence))), aug_data_batch)))
        aug_sentence_id_batch = [[sentence[0][0] for sentence in batch_sentence] for batch_sentence in aug_data_batch]
        aug_predicate_id_batch = [[sentence[0][1] for sentence in batch_sentence] for batch_sentence in aug_data_batch]
        aug_sentence_len_batch = [[int(sentence[0][2]) for sentence in batch_sentence] for batch_sentence in
                                  aug_data_batch]

        aug_id_batch = [[[int(item[3]) for item in sentence] for sentence in batch_sentence] for batch_sentence in
                        aug_data_batch]

        aug_flag_batch = [[[int(item[4]) for item in sentence] for sentence in batch_sentence] for batch_sentence in
                          aug_data_batch]
        aug_pad_flag_batch = np.array(pad_batch_new_max_length(aug_flag_batch, batch_size, 0, max_length), dtype=int)

        aug_text_batch = [[[item[5] for item in sentence] for sentence in batch_sentence] for batch_sentence in
                          aug_data_batch]
        if len(aug_text_batch) < batch_size:
            aug_text_batch += [[[_PAD_] for _ in range(middle_size)] for _ in range(batch_size - len(aug_text_batch))]

        aug_word_batch = [[[word2idx.get(item[5], word2idx[_UNK_]) for item in sentence] for sentence in batch_sentence]
                          for batch_sentence in aug_data_batch]
        aug_pad_word_batch = np.array(pad_batch_new_max_length(aug_word_batch, batch_size, word2idx[_PAD_], max_length))

        aug_lemma_batch = [
            [[lemma2idx.get(item[6], lemma2idx[_UNK_]) for item in sentence] for sentence in batch_sentence] for
            batch_sentence in aug_data_batch]
        aug_pad_lemma_batch = np.array(
            pad_batch_new_max_length(aug_lemma_batch, batch_size, lemma2idx[_PAD_], max_length))

        aug_pos_batch = [[[pos2idx.get(item[7], pos2idx[_UNK_]) for item in sentence] for sentence in batch_sentence]
                         for batch_sentence in aug_data_batch]
        aug_pad_pos_batch = np.array(pad_batch_new_max_length(aug_pos_batch, batch_size, pos2idx[_PAD_], max_length))

        aug_argument_batch = [
            [[argument2idx.get(item[9], argument2idx[_UNK_]) for item in sentence] for sentence in batch_sentence] for
            batch_sentence in aug_data_batch]
        aug_pad_argument_batch = np.array(
            pad_batch_new_max_length(aug_argument_batch, batch_size, argument2idx[_PAD_], max_length))

        aug_pretrain_word_batch = [
            [[pretrain2idx.get(item[5], pretrain2idx[_UNK_]) for item in sentence] for sentence in batch_sentence] for
            batch_sentence in aug_data_batch]
        aug_pad_pretrain_word_batch = np.array(
            pad_batch_new_max_length(aug_pretrain_word_batch, batch_size, pretrain2idx[_PAD_], max_length))

        if deprel2idx is not None:
            aug_deprel_batch = [
                [[deprel2idx.get(item[8], deprel2idx[_UNK_]) for item in sentence] for sentence in batch_sentence] for
                batch_sentence in aug_data_batch]
            aug_pad_deprel_batch = np.array(
                pad_batch_new_max_length(aug_deprel_batch, batch_size, deprel2idx[_PAD_], max_length))
        else:
            aug_pad_deprel_batch = []

        aug_batch = {
            "flag": aug_pad_flag_batch,
            "word": aug_pad_word_batch,
            "argument": aug_pad_argument_batch,
            "lemma": aug_pad_lemma_batch,
            "pretrain": aug_pad_pretrain_word_batch,
            "pos": aug_pad_pos_batch,
            "deprel": aug_pad_deprel_batch,
            "text": aug_text_batch
        }

        res = {
            'ori': batch,
            'aug': aug_batch
        }

        yield res