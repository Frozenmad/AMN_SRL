from data_utils import _PAD_, _UNK_
import inter_utils
from inter_utils import tprint
from final_model import get_torch_variable_from_np, get_data
import torch
import numpy as np
import os

def sem_f1_score(target, predict, predicate_correct, predicate_sum ,argument2idx, output_to_file = None):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    total = 0
    for i in range(len(target)):
        pred_i = predict[i]
        golden_i = target[i]
        if golden_i == argument2idx[_PAD_]:
            continue
        total += 1
        if pred_i == argument2idx[_UNK_]:
            pred_i = argument2idx['_']
        if golden_i == argument2idx[_UNK_]:
            golden_i = argument2idx['_']
        if pred_i != argument2idx['_']:
            predict_args += 1
        if golden_i != argument2idx['_']:
            golden_args += 1
        if golden_i != argument2idx['_'] and pred_i == golden_i:
            correct_args += 1
        if pred_i == golden_i:
            num_correct += 1

    P = (correct_args + predicate_correct) / (predict_args + predicate_sum + 1e-13)

    R = (correct_args + predicate_correct) / (golden_args + predicate_sum + 1e-13)

    NP = correct_args / (predict_args + 1e-13)

    NR = correct_args / (golden_args + 1e-13)
        
    F1 = 2 * P * R / (P + R + 1e-13)

    NF1 = 2 * NP * NR / (NP + NR + 1e-13)

    outs = tprint('eval accurate:{:.2f} predict:{} golden:{} correct:{} P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(num_correct/total*100, predict_args, golden_args, correct_args, P*100, R*100, F1*100, NP*100, NR *100, NF1 * 100))

    if output_to_file is not None:
        with open(output_to_file,'a') as files:
            files.write(outs)

    return (P, R, F1, NP, NR, NF1)

def pruning_sem_f1_score(target, predict, predicate_correct, predicate_sum, out_of_pruning, argument2idx):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    total = 0
    for i in range(len(target)):
        pred_i = predict[i]
        golden_i = target[i]
        if golden_i == argument2idx[_PAD_]:
            continue
        total += 1
        if pred_i == argument2idx[_UNK_]:
            pred_i = argument2idx['_']
        if golden_i == argument2idx[_UNK_]:
            golden_i = argument2idx['_']
        if pred_i != argument2idx['_']:
            predict_args += 1
        if golden_i != argument2idx['_']:
            golden_args += 1
        if golden_i != argument2idx['_'] and pred_i == golden_i:
            correct_args += 1
        if pred_i == golden_i:
            num_correct += 1

    P = (correct_args + predicate_correct) / (predict_args + predicate_sum + 1e-13)

    R = (correct_args + predicate_correct) / (golden_args + out_of_pruning + predicate_sum + 1e-13)
        
    NP = correct_args / (predict_args + 1e-13)

    NR = correct_args / (golden_args + 1e-13)
        
    F1 = 2 * P * R / (P + R + 1e-13)

    NF1 = 2 * NP * NR / (NP + NR + 1e-13)

    tprint('\teval accurate:{:.2f} predict:{} golden:{} correct:{} P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(num_correct/total*100, predict_args, golden_args + out_of_pruning, correct_args, P*100, R*100, F1*100, NP*100, NR *100, NF1 * 100))
    
    return (P, R, F1, NP, NR, NF1)

def eval_train_batch(epoch,batch_i,loss,golden_batch,predict_batch,argument2idx,output_to_file = None):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    batch_total = 0
    for i in range(len(golden_batch)):
        pred_i = predict_batch[i]
        golden_i = golden_batch[i]
        if golden_i == argument2idx[_PAD_]:
            continue
        batch_total += 1
        if pred_i == argument2idx[_UNK_]:
            pred_i = argument2idx['_']
        if golden_i == argument2idx[_UNK_]:
            golden_i = argument2idx['_']
        if pred_i != argument2idx['_']:
            predict_args += 1
        if golden_i != argument2idx['_']:
            golden_args += 1
        if golden_i != argument2idx['_'] and pred_i == golden_i:
            correct_args += 1
        if pred_i == golden_i:
            num_correct += 1

    recall = correct_args / (golden_args + 1e-13)
    precision = correct_args / (predict_args + 1e-13)
    F = 2 * recall * precision / (recall + precision + 1e-13)

    outs = tprint('epoch {} batch {} loss:{:4f} accurate:{:.2f} precision:{} recall:{} F:{}'.format(epoch, batch_i, loss, num_correct/batch_total*100, precision, recall, F))

    if output_to_file is not None:
        with open(output_to_file,'a') as out_file:
            out_file.write(outs)

    return correct_args,golden_args,predict_args

def my_eval_batch(golden_batch,predict_batch,argument2idx):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    batch_total = 0
    for i in range(len(golden_batch)):
        pred_i = predict_batch[i]
        golden_i = golden_batch[i]
        if golden_i == argument2idx[_PAD_]:
            continue
        batch_total += 1
        if pred_i == argument2idx[_UNK_]:
            pred_i = argument2idx['_']
        if golden_i == argument2idx[_UNK_]:
            golden_i = argument2idx['_']
        if pred_i != argument2idx['_']:
            predict_args += 1
        if golden_i != argument2idx['_']:
            golden_args += 1
        if golden_i != argument2idx['_'] and pred_i == golden_i:
            correct_args += 1
        if pred_i == golden_i:
            num_correct += 1

    return correct_args,golden_args,predict_args,batch_total,num_correct

def eval_data_predicate(model, criterion, dataset, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, label2idx, idx2label):

    model.eval()

    pred_all = []
    target_label_all = []
    for batch_i, input_data in enumerate(inter_utils.predicate_get_batch(dataset, batch_size, word2idx,
                                                                         lemma2idx, pos2idx, pretrain2idx, label2idx)):
        target_label = np.array(input_data['label_id'])
        target_label = get_torch_variable_from_np(target_label)
        out = model(input_data)
        loss = criterion(out, target_label)
        _,pred = torch.max(out, 1)
        pred = get_data(pred)
        pred_all.extend(pred.tolist())
        target_label_all.extend(target_label)

    judge_list = (np.array(pred_all) == np.array(target_label_all))

    model.train()

    return judge_list


def eval_data(model, criterion, dataset, batch_size, predicate_correct, predicate_sum ,word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx, deprel2idx, idx2argument, out_of_pruning = 0, output_to_file = None, use_attention = True):

    model.eval()
    golden = []
    predict = []

    output_data = []
    cur_sentence = None
    cur_sentence_data = None

    for batch_i, input_data in enumerate(inter_utils.final_get_batch(dataset, batch_size, word2idx,
                                                             lemma2idx, pos2idx, pretrain2idx, argument2idx, deprel2idx)):
        
        target_argument = input_data['ori']['argument']
        
        flat_argument = input_data['ori']['flat_argument']

        target_batch_variable = get_torch_variable_from_np(flat_argument)

        sentence_id = input_data['ori']['sentence_id']
        predicate_id = input_data['ori']['predicate_id']
        word_id = input_data['ori']['word_id']
        sentence_len =  input_data['ori']['sentence_len']
        seq_len = input_data['ori']['seq_len']
        bs = input_data['ori']['batch_size']
        psl = input_data['ori']['pad_seq_len']
        
        out = model(input_data)
            
        loss = criterion(out, target_batch_variable)

        _, pred = torch.max(out, 1)

        pred = get_data(pred)

        pred = pred.tolist()

        golden += flat_argument.tolist()

        predict += pred

        pre_data = []
        for b in range(len(seq_len)):
            line_data = ['_' for _ in range(sentence_len[b])]
            for s in range(seq_len[b]):
                wid = word_id[b][s]
                line_data[wid-1] = idx2argument[pred[b * psl + s]]
            pre_data.append(line_data)

        for b in range(len(sentence_id)):
            if cur_sentence != sentence_id[b]:
                if cur_sentence_data is not None:
                    output_data.append(cur_sentence_data)
                cur_sentence_data = [[sentence_id[b]]*len(pre_data[b]),pre_data[b]]
                cur_sentence = sentence_id[b]
            else:
                assert cur_sentence_data is not None
                cur_sentence_data.append(pre_data[b])
    if cur_sentence_data is not None and len(cur_sentence_data)>0:
        output_data.append(cur_sentence_data)
    
    score = pruning_sem_f1_score(golden, predict, predicate_correct, predicate_sum, out_of_pruning, argument2idx)

    model.train()

    return score, output_data

def pruning_eval_data(model, criterion, dataset, batch_size, predicate_correct, predicate_sum, out_of_pruning, word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx, idx2argument):

    model.eval()
    golden = []
    predict = []

    output_data = []
    cur_sentence = None
    cur_sentence_data = None

    for batch_i, input_data in enumerate(inter_utils.get_batch(dataset, batch_size, word2idx,
                                                             lemma2idx, pos2idx, pretrain2idx, argument2idx)):
        
        target_argument = input_data['argument']
        
        flat_argument = input_data['flat_argument']

        target_batch_variable = get_torch_variable_from_np(flat_argument)

        sentence_id = input_data['sentence_id']
        predicate_id = input_data['predicate_id']
        word_id = input_data['word_id']
        sentence_len =  input_data['sentence_len']
        seq_len = input_data['seq_len']
        bs = input_data['batch_size']
        psl = input_data['pad_seq_len']
        
        out = model(input_data)

        loss = criterion(out, target_batch_variable)

        _, pred = torch.max(out, 1)

        pred = get_data(pred)

        pred = pred.tolist()

        golden += flat_argument.tolist()

        predict += pred

        pre_data = []
        for b in range(len(seq_len)):
            line_data = ['_' for _ in range(sentence_len[b])]
            for s in range(seq_len[b]):
                wid = word_id[b][s]
                line_data[wid-1] = idx2argument[pred[b * psl + s]]
            pre_data.append(line_data)

        for b in range(len(sentence_id)):
            if cur_sentence != sentence_id[b]:
                if cur_sentence_data is not None:
                    output_data.append(cur_sentence_data)
                cur_sentence_data = [[sentence_id[b]]*len(pre_data[b]),pre_data[b]]
                cur_sentence = sentence_id[b]
            else:
                assert cur_sentence_data is not None
                cur_sentence_data.append(pre_data[b])
    
    if cur_sentence_data is not None and len(cur_sentence_data)>0:
        output_data.append(cur_sentence_data)

    score = pruning_sem_f1_score(golden, predict, predicate_correct, predicate_sum, out_of_pruning, argument2idx)

    model.train()

    return score, output_data

def print_result(rd):
    print('epoch:{} step:{} accurate:{:.2f} predict:{} golden:{} correct:{} P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'
        .format(rd['e'],rd['s'],rd['acc']*100, rd['p'], rd['g'], rd['c'], rd['P']*100, rd['R']*100, 
            rd['F1']*100, rd['NP']*100, rd['NR']*100, rd['NF1']*100))

def my_eval(epoch,step,model,criterion,dataset,batch_size,predicate_correct,predicate_sum,
            word2idx,lemma2idx,pos2idx,pretrain2idx,argument2idx,use_attention=False,aug_data=None):
    model.eval()
    gold = predict = correct = total = num_correct = 0
    for batch_i, input_data in enumerate(inter_utils.get_batch(dataset,batch_size,word2idx,
                                                        lemma2idx,pos2idx,pretrain2idx,argument2idx)):
        target_argument = input_data["argument"]
        flat_argument = input_data["flat_argument"]
        target_batch_variable = get_torch_variable_from_np(flat_argument)
        
        if use_attention:
            out = model((input_data, aug_data))
        else:
            out = model(input_data)
        
        _, pred = torch.max(out, 1)

        pred = get_data(pred)

        loss = criterion(out, target_batch_variable)
        cur_correct,cur_golden,cur_predict,cur_total,cur_nc = my_eval_batch(flat_argument, pred, argument2idx)
        gold += cur_golden
        predict += cur_predict
        correct += cur_correct
        total += cur_total
        num_correct += cur_nc

    P = (correct + predicate_correct) / (predict + predicate_sum + 1e-13)
    R = (correct + predicate_correct) / (gold + predicate_sum + 1e-13)
    NP = correct / (predict + 1e-13)
    NR = correct / (gold + 1e-13)
    F1 = 2 * P * R / (P + R + 1e-13)
    NF1 = 2 * NP * NR / (NP + NR + 1e-13)
    acc = num_correct / total
    print_dict = {'e':epoch,'s':step,'acc':acc,'p':predict,'g':gold,'c':correct,'P':P,'R':R,'F1':F1,'NP':NP,'NR':NR,'NF1':NF1}
    print_result(print_dict)
    model.train()
    return print_dict
