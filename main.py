from __future__ import print_function
import final_model
import data_utils
import inter_utils
from inter_utils import tprint
import pickle
import time
import os
import torch
from torch import nn
from torch import optim

from final_model import USE_CUDA
from final_model import get_torch_variable_from_np, get_data
from scorer import *
from data_utils import output_predict

from final_inter_helper import *

#torch.cuda.set_device(1)

if not os.path.exists(os.path.exists('./log/')):
    os.makedirs('./log/')

model_name = 'best_model'

aug_num = 4

inter_utils.save_file = './log/%s.log' % model_name

if not os.path.exists('./models/%s/' % (model_name)):
    os.makedirs('./models/%s/' % (model_name))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tprint('SRL basic model')

tprint('start loading data...')
start_t = time.time()

train_input_file = os.path.join(os.path.dirname(__file__),'temp/train.flat.txt')
dev_input_file = os.path.join(os.path.dirname(__file__),'temp/dev.flat.txt')
test_input_file = os.path.join(os.path.dirname(__file__),'temp/test.flat.txt')

all_data = final_dataset(train_input_file,dev_input_file,test_input_file,aug_num,'temp/train_train.bin','temp/train_dev.bin','temp/train_test.bin')

train_dataset = all_data.train
dev_dataset = all_data.dev
test_dataset = all_data.test

word2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/word2idx.bin'))
idx2word = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2word.bin'))

lemma2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/lemma2idx.bin'))
idx2lemma = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2lemma.bin'))

pos2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pos2idx.bin'))
idx2pos = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2pos.bin'))

pretrain2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pretrain2idx.bin'))
idx2pretrain = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2pretrain.bin'))

argument2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/argument2idx.bin'))
idx2argument = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2argument.bin'))

pretrain_emb_weight = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pretrain.emb.bin'))

tprint('data loading finished! consuming {} s'.format(int(time.time()-start_t)))

result_path = os.path.join(os.path.dirname(__file__),'result/')

tprint('start building model...')
start_t = time.time()

if not os.path.exists(os.path.join('./models/',model_name)):
    os.makedirs(os.path.join('./models/',model_name))

# hyper parameters
max_epoch = 500000
learning_rate = 0.001
batch_size = 32
dropout = 0.1
word_embedding_size = 100
pos_embedding_size = 32
pretrained_embedding_size = 100
lemma_embedding_size = 100
input_layer_size1 = 512
input_layer_size2 = 512
hidden_layer1_size = 256
hidden_layer2_size = 128
hidden_layer3_size = None
bilstm_hidden_size1 = 512
bilstm_hidden_size2 = 512
bilstm_num_layers1 = 2
bilstm_num_layers2 = 3
show_train_steps = 100
show_dev_steps = 400
show_test_steps = 800
flag_embedding_size = 16
a_batch_size = 100
argument_emb_size = 128
elmo_embedding_size = 300
highway_layers = 10
only_attention = False

merge_type = 'MEAN'

use_highway = False
use_elmo = True
use_attention = True

# TODO: need modify for predicate accuracy
dev_predicate_sum = 6390
dev_predicate_correct = int(6390 * 0.95)

test_predicate_sum = 10498
test_predicate_correct = int(10498 * 0.95)

model_params = {
    "dropout":dropout,
    "batch_size":batch_size,
    "word_vocab_size":len(word2idx),
    "lemma_vocab_size":len(lemma2idx),
    "pos_vocab_size":len(pos2idx),
    "pretrain_vocab_size":len(pretrain2idx),
    "word_emb_size":word_embedding_size,
    "lemma_emb_size":lemma_embedding_size,
    "pos_emb_size":pos_embedding_size,
    "pretrain_emb_size":pretrained_embedding_size,
    "pretrain_emb_weight":pretrain_emb_weight,
    'argument_emb_size':argument_emb_size,
    "input_layer_size1":input_layer_size1,
    "input_layer_size2":input_layer_size2,
    "bilstm_num_layers1":bilstm_num_layers1,
    "bilstm_num_layers2":bilstm_num_layers2,
    "bilstm_hidden_size1":bilstm_hidden_size1,
    "bilstm_hidden_size2":bilstm_hidden_size2,
    "hidden_layer1_size":hidden_layer1_size,
    "hidden_layer2_size":hidden_layer2_size,
    "hidden_layer3_size":hidden_layer3_size,
    "target_vocab_size":len(argument2idx),
    "use_highway":use_highway,
    'use_elmo':use_elmo,
    "aug_num":aug_num,
    "highway_layers": highway_layers,
    "flag_embedding_size":flag_embedding_size,
    "elmo_embedding_size":elmo_embedding_size,
    'use_attention' : use_attention,
    'only_attention' : only_attention,
    'merge_type' : merge_type,
    'use_syntax' : False,
    'deprel_vocab_size' : 0,
    'deprel_emb_size' : 0
}

# build model
srl_model = final_model.finalAttentionModel(model_params)

if USE_CUDA:
    srl_model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(srl_model.parameters(),lr=learning_rate)


tprint('model build finished! consuming {} s'.format(int(time.time()-start_t)))

tprint('Start training...')

dev_best_score = None
test_best_score = None

for epoch in range(max_epoch):
    for batch_i, train_input_data in enumerate(inter_utils.final_get_batch(train_dataset, batch_size,word2idx,
                                                             lemma2idx, pos2idx, pretrain2idx, argument2idx)):
        


        target_argument = train_input_data['ori']['argument']
        
        flat_argument = train_input_data['ori']['flat_argument']

        target_batch_variable = get_torch_variable_from_np(flat_argument)
        
        out = srl_model(train_input_data)

        loss = criterion(out, target_batch_variable)


        if batch_i % show_train_steps == 0: 

            _, pred = torch.max(out, 1)

            pred = get_data(pred)

            # pred = pred.reshape([bs, sl])

            tprint('\n')
            tprint('*'*80)

            eval_train_batch(epoch, batch_i, loss.item(), flat_argument, pred, argument2idx)

        cur_log = {'epoch':epoch,'batch':batch_i}

        if batch_i % show_dev_steps == 0:
            tprint('dev:')
            score, dev_output = eval_data(srl_model, criterion, dev_dataset, batch_size, dev_predicate_correct, dev_predicate_sum, word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx, None, idx2argument, use_attention=use_attention)
            if dev_best_score is None or score[5] > dev_best_score[5]:
                dev_best_score = score
                output_predict(os.path.join(result_path,'dev_argument_{:.2f}.pred'.format(dev_best_score[2]*100)),dev_output)
                dev_log = cur_log
            tprint('dev best epoch:{} batch:{} P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(
                                                                                            dev_log['epoch'],dev_log['batch'],
                                                                                            dev_best_score[0] * 100, dev_best_score[1] * 100,
                                                                                            dev_best_score[2] * 100, dev_best_score[3] * 100,
                                                                                            dev_best_score[4] * 100, dev_best_score[5] * 100))

            tprint('test:')
            score, test_output = eval_data(srl_model, criterion, test_dataset, batch_size, test_predicate_correct, test_predicate_sum, word2idx, lemma2idx, pos2idx, pretrain2idx, argument2idx, None, idx2argument,use_attention=use_attention)
            if test_best_score is None or score[5] > test_best_score[5]:
                test_best_score = score
                output_predict(os.path.join(result_path,'test_argument_{:.2f}.pred'.format(test_best_score[2]*100)),test_output)
                test_log = cur_log
            tprint('test best epoch:{} batch:{} P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(
                                                                                            test_log['epoch'],test_log['batch'],
                                                                                            test_best_score[0] * 100, test_best_score[1] * 100,
                                                                                            test_best_score[2] * 100, test_best_score[3] * 100,
                                                                                            test_best_score[4] * 100, test_best_score[5] * 100))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
