import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from highway import HighwayMLP

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "./model/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "./model/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available() and True

class ElmoEmbedding(nn.Module):
    def __init__(self, params):
        super(ElmoEmbedding, self).__init__()
        self.weight_file = weight_file
        self.options_file = options_file
        self.elmo_emb_size = params['emb_elmo_size']
        self.layer_weight = nn.Parameter(torch.tensor([0.5,0.5], device = device))
        self.gamma = nn.Parameter(torch.ones(1,device = device))
        self.mlp = nn.Sequential(nn.Linear(1024, self.elmo_emb_size), nn.ReLU())
        self.elmo = Elmo(self.options_file, self.weight_file, 2)
        if USE_CUDA:
            self.elmo.cuda()

    def forward(self, input_sentence_batch):
        character_ids = batch_to_ids(input_sentence_batch)
        character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)
        layer1 = embeddings['elmo_representations'][0]
        layer2 = embeddings['elmo_representations'][1]
        lw = F.softmax(self.layer_weight,  dim = 0)
        final_layer = self.gamma * (layer1 * lw[0] + layer2 * lw[1])
        return self.mlp(final_layer)

def get_torch_variable_from_np(v):
    if USE_CUDA:
        return Variable(torch.from_numpy(v)).cuda()
    else:
        return Variable(torch.from_numpy(v))

def get_torch_variable_from_tensor(t):
    if USE_CUDA:
        return Variable(t).cuda()
    else:
        return Variable(t)

def get_data(v):
    if USE_CUDA:
        return v.data.cpu().numpy()
    else:
        return v.data.numpy()

class finalAttentionModel(nn.Module):
    def __init__(self, model_params):

        super(finalAttentionModel,self).__init__()
        self.dropout = model_params['dropout']
        self.batch_size = model_params['batch_size']
        self.a_batch_size = self.batch_size * model_params['aug_num']
        self.use_highway = model_params['use_highway']
        self.use_elmo = model_params['use_elmo']
        self.aug_num = model_params['aug_num']

        self.word_vocab_size = model_params['word_vocab_size']
        self.lemma_vocab_size = model_params['lemma_vocab_size']
        self.pos_vocab_size = model_params['pos_vocab_size']
        self.pretrain_vocab_size = model_params['pretrain_vocab_size']

        self.flag_emb_size = model_params['flag_embedding_size']
        self.word_emb_size = model_params['word_emb_size']
        self.lemma_emb_size = model_params['lemma_emb_size']
        self.pos_emb_size = model_params['pos_emb_size']
        self.pretrain_emb_size = model_params['pretrain_emb_size']

        self.pretrain_emb_weight = model_params['pretrain_emb_weight']

        self.predicate_emb = False

        # self.input_layer_size = model_params['input_layer_size']
        self.bilstm_num_layers1 = model_params['bilstm_num_layers1']
        self.bilstm_num_layers2 = model_params['bilstm_num_layers2']
        self.bilstm_hidden_size1 = model_params['bilstm_hidden_size1']
        self.bilstm_hidden_size2 = model_params['bilstm_hidden_size2']
        self.hidden_layer1_size = model_params['hidden_layer1_size']
        self.hidden_layer2_size = model_params['hidden_layer2_size']
        self.hidden_layer3_size = model_params['hidden_layer3_size']
        self.target_vocab_size = model_params['target_vocab_size']
        self.argument_emb_size = model_params['argument_emb_size']
        self.use_attention = model_params['use_attention']
        self.only_attention = model_params['only_attention']
        self.flag_embedding = nn.Embedding(2, 16)
        self.elmo_emb_size = model_params['elmo_embedding_size']
        self.merge_type = model_params['merge_type']
        self.use_syntax = model_params['use_syntax']
        self.flat_attention = self.merge_type == 'ORDI'
        self.deprel_vocab_size = model_params['deprel_vocab_size']
        self.deprel_emb_size = model_params['deprel_emb_size']

        self.highway_layers = model_params['highway_layers']

        self.word_embedding = nn.Embedding(self.word_vocab_size, self.word_emb_size)
        self.word_embedding.weight.data.uniform_(-1.0,1.0)

        self.lemma_embedding = nn.Embedding(self.lemma_vocab_size, self.lemma_emb_size)
        self.lemma_embedding.weight.data.uniform_(-1.0,1.0)

        self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_emb_size)
        self.pos_embedding.weight.data.uniform_(-1.0,1.0)

        self.argument_embedding = nn.Embedding(self.target_vocab_size, self.argument_emb_size)
        self.argument_embedding.weight.data.uniform_(-1.0,1.0)

        self.pretrained_embedding = nn.Embedding(self.pretrain_vocab_size,self.pretrain_emb_size)
        self.pretrained_embedding.weight.data.copy_(torch.from_numpy(np.array(self.pretrain_emb_weight)))

        if 'use_affine' in model_params:
            self.use_affine = model_params['use_affine']
        else:
            self.use_affine = False

        if self.use_syntax:
            self.deprel_embedding = nn.Embedding(self.deprel_vocab_size,self.deprel_emb_size)
            self.deprel_embedding.weight.data.uniform_(-1.0,1.0)

        input_emb_size = self.flag_emb_size  + self.pretrain_emb_size + self.word_emb_size  + self.lemma_emb_size + self.pos_emb_size

        if self.use_elmo:
            self.ElmoLayer = ElmoEmbedding({'emb_elmo_size':self.elmo_emb_size})
            input_emb_size += self.elmo_emb_size

        if self.use_syntax:
            input_emb_size += self.deprel_emb_size


        lstm2_input_emb_size = input_emb_size * (1 - self.only_attention) + self.argument_emb_size * self.aug_num * self.flat_attention + self.argument_emb_size * (1 - self.flat_attention)

        if USE_CUDA:
            self.bilstm_hidden_state1_1 = Variable(torch.randn(2 * self.bilstm_num_layers1, 1, self.bilstm_hidden_size1), requires_grad = True).cuda()
            self.bilstm_hidden_state1_2 = Variable(torch.randn(2 * self.bilstm_num_layers1, 1, self.bilstm_hidden_size1), requires_grad = True).cuda()
            self.bilstm_hidden_state2_1 = Variable(torch.randn(2 * self.bilstm_num_layers2, 1, self.bilstm_hidden_size2), requires_grad = True).cuda()
            self.bilstm_hidden_state2_2 = Variable(torch.randn(2 * self.bilstm_num_layers2, 1, self.bilstm_hidden_size2), requires_grad = True).cuda()
        else:
            self.bilstm_hidden_state1_1 = Variable(torch.randn(2 * self.bilstm_num_layers1, 1, self.bilstm_hidden_size1), requires_grad = True)
            self.bilstm_hidden_state1_2 = Variable(torch.randn(2 * self.bilstm_num_layers1, 1, self.bilstm_hidden_size1), requires_grad = True)
            self.bilstm_hidden_state2_1 = Variable(torch.randn(2 * self.bilstm_num_layers2, 1, self.bilstm_hidden_size2), requires_grad = True)
            self.bilstm_hidden_state2_2 = Variable(torch.randn(2 * self.bilstm_num_layers2, 1, self.bilstm_hidden_size2), requires_grad = True)

        if self.use_attention:
            if USE_CUDA:
                self.attention_hidden_state = (Variable(torch.randn(2 * self.bilstm_num_layers1, self.a_batch_size, self.bilstm_hidden_size1),requires_grad=True).cuda(),
                                            Variable(torch.randn(2 * self.bilstm_num_layers1, self.a_batch_size, self.bilstm_hidden_size1),requires_grad=True).cuda())

            else:
                self.attention_hidden_state = (Variable(torch.randn(2 * self.bilstm_num_layers1, self.a_batch_size, self.bilstm_hidden_size1),requires_grad=True),
                                        Variable(torch.randn(2 * self.bilstm_num_layers1, self.a_batch_size, self.bilstm_hidden_size1),requires_grad=True))

        self.bilstm_layer1 = nn.LSTM(input_size=input_emb_size,
                                    hidden_size = self.bilstm_hidden_size1, num_layers = self.bilstm_num_layers1,
                                    dropout = self.dropout, bidirectional = True,
                                    bias = True, batch_first=True)

        if self.use_attention:
            self.bilstm_layer2 = nn.LSTM(input_size=lstm2_input_emb_size,
                                    hidden_size = self.bilstm_hidden_size2, num_layers = self.bilstm_num_layers2,
                                    dropout = self.dropout, bidirectional = True,
                                    bias = True, batch_first=True)
        else:
            self.bilstm_layer2 = nn.LSTM(input_size=2*self.bilstm_hidden_size1,
                                    hidden_size = self.bilstm_hidden_size2, num_layers = self.bilstm_num_layers2,
                                    dropout = self.dropout, bidirectional = True,
                                    bias = True, batch_first=True)

        if self.use_highway:
            self.highway_layers = nn.ModuleList([HighwayMLP(self.bilstm_hidden_size2*2, activation_function=F.relu)
                                             for _ in range(self.highway_layers)])

            self.output_layer = nn.Linear(self.bilstm_hidden_size2*2, self.target_vocab_size)

        else:
            if self.hidden_layer1_size is None:
                self.output_layer = nn.Linear(self.bilstm_hidden_size2*2, self.target_vocab_size)

            else:
                self.hidden_layer1 = nn.Sequential(nn.Linear(self.bilstm_hidden_size2*2,self.hidden_layer1_size),nn.BatchNorm1d(self.hidden_layer1_size),nn.ReLU(True))
                if self.hidden_layer2_size is None:
                    self.output_layer = nn.Linear(self.hidden_layer1_size,self.target_vocab_size)
                else: 
                    self.hidden_layer2 = nn.Sequential(nn.Linear(self.hidden_layer1_size,self.hidden_layer2_size),nn.BatchNorm1d(self.hidden_layer2_size), nn.ReLU(True))
                    if self.hidden_layer3_size is None:
                        self.output_layer = nn.Linear(self.hidden_layer2_size,self.target_vocab_size)
                    else:
                        self.hidden_layer3 = nn.Sequential(nn.Linear(self.hidden_layer2_size,self.hidden_layer3_size),nn.BatchNorm1d(self.hidden_layer3_size), nn.ReLU(True))
                        self.output_layer = nn.Linear(self.hidden_layer3_size,self.target_vocab_size)



    def _init_hidden(self, num_flag, batch_size):
        if num_flag == 0:
            return (torch.cat([self.bilstm_hidden_state1_1] * batch_size, dim=1),
                    torch.cat([self.bilstm_hidden_state1_2] * batch_size, dim=1))
        else:
            return (torch.cat([self.bilstm_hidden_state2_1] * batch_size, dim=1),
                    torch.cat([self.bilstm_hidden_state2_2] * batch_size, dim=1))

    def forward(self, tuple_input):
        """
            for attention model, the input are like this:
            batch_input : batch_size * input_length * dimension
            aug_input: batch_size * [1,2] * aug_length * dimension
        """
        if self.use_attention:
            batch_input,aug_input=tuple_input['ori'],tuple_input['aug']
        else:
            batch_input = tuple_input['ori']

        flag_batch = get_torch_variable_from_np(batch_input['flag'])
        word_batch = get_torch_variable_from_np(batch_input['word'])
        lemma_batch = get_torch_variable_from_np(batch_input['lemma'])
        pos_batch = get_torch_variable_from_np(batch_input['pos'])
        pretrain_batch = get_torch_variable_from_np(batch_input['pretrain'])

        batch_size = len(flag_batch)
        seqLen = flag_batch.size(1)

        flag_emb = self.flag_embedding(flag_batch)
        word_emb = self.word_embedding(word_batch)
        lemma_emb = self.lemma_embedding(lemma_batch)
        pos_emb = self.pos_embedding(pos_batch)
        pretrain_emb = self.pretrained_embedding(pretrain_batch)
        input_emb = torch.cat([flag_emb, word_emb, pretrain_emb, lemma_emb, pos_emb], 2)
        if self.use_syntax:
            deprel_batch = get_torch_variable_from_np(batch_input['deprel'])
            deprel_emb = self.deprel_embedding(deprel_batch)
            input_emb = torch.cat([input_emb, deprel_emb], 2)

        if self.use_elmo:
            batch_text = batch_input['text']
            elmo_part = self.ElmoLayer(batch_text)
            input_emb = torch.cat([input_emb,elmo_part],2)

        bilstm_output, bilstm_hidden_state = self.bilstm_layer1(input_emb,self._init_hidden(0, batch_size))
        hidden_input = bilstm_output.contiguous()
        hidden_input = hidden_input.view(bilstm_output.shape[0],bilstm_output.shape[1],-1)

        if self.use_attention:
            # now we need to add the attention to this net
            a_flag_batch = get_torch_variable_from_np(aug_input['flag'])
            a_word_batch = get_torch_variable_from_np(aug_input['word'])
            a_lemma_batch = get_torch_variable_from_np(aug_input['lemma'])
            a_pos_batch = get_torch_variable_from_np(aug_input['pos'])
            a_pretrain_batch = get_torch_variable_from_np(aug_input['pretrain'])
            a_aug_batch = get_torch_variable_from_np(aug_input['argument'])
            if self.use_syntax:
                a_deprel_batch = get_torch_variable_from_np(aug_input['deprel'])

            detect_batch_size = a_flag_batch.shape[0]
            detect_a_batch_size = a_flag_batch.shape[1]

            if detect_a_batch_size * detect_batch_size != self.a_batch_size:
                raise ValueError('Total shape not match!')

            a_flag_batch = a_flag_batch.view(detect_batch_size*detect_a_batch_size,a_flag_batch.shape[2])
            a_word_batch = a_word_batch.view(detect_batch_size*detect_a_batch_size,a_word_batch.shape[2])
            a_lemma_batch = a_lemma_batch.view(detect_batch_size*detect_a_batch_size,a_lemma_batch.shape[2])
            a_pos_batch = a_pos_batch.view(detect_batch_size*detect_a_batch_size,a_pos_batch.shape[2])
            a_pretrain_batch = a_pretrain_batch.view(detect_batch_size*detect_a_batch_size,a_pretrain_batch.shape[2])
            a_aug_batch = a_aug_batch.view(detect_batch_size*detect_a_batch_size,a_aug_batch.shape[2])

            a_flag_emb = self.flag_embedding(a_flag_batch)
            a_word_emb = self.word_embedding(a_word_batch)
            a_lemma_emb = self.lemma_embedding(a_lemma_batch)
            a_pos_emb = self.pos_embedding(a_pos_batch)
            a_pretrain_emb = self.pretrained_embedding(a_pretrain_batch)
            a_aug_emb = self.argument_embedding(a_aug_batch)
            a_input_emb = torch.cat([a_flag_emb, a_word_emb, a_pretrain_emb, a_lemma_emb, a_pos_emb], 2)
            if self.use_syntax:
                a_deprel_batch = a_deprel_batch.view(detect_batch_size*detect_a_batch_size,a_deprel_batch.shape[2])
                a_deprel_emb = self.deprel_embedding(a_deprel_batch)
                a_input_emb = torch.cat([a_input_emb,a_deprel_emb], 2)
    


            if self.use_elmo:
                raw_a_batch_text = aug_input['text']
                a_batch_text = []
                for e_batch_text in raw_a_batch_text:
                    a_batch_text.extend(e_batch_text)
                a_elmo_emb = self.ElmoLayer(a_batch_text)
                a_input_emb = torch.cat([a_input_emb,a_elmo_emb],2)

            a_aug_emb = a_aug_emb.view(detect_batch_size,detect_a_batch_size,a_aug_emb.shape[1],a_aug_emb.shape[2])

            a_bilstm_output, a_bilstm_hidden_state = self.bilstm_layer1(a_input_emb,self._init_hidden(0, self.aug_num * batch_size))
            a_hidden_input = a_bilstm_output.contiguous()
            a_hidden_input = a_hidden_input.view(detect_batch_size,detect_a_batch_size,a_bilstm_output.shape[1],-1)
            """
            ========================================================================================================================
            here we also get the final aug embedding input
            """
            """
                here are some versions of merging:
                ORIG : as usual.
                FLAT : concatenate all the words of auxiliary sentences.
                MEAN : use mean value instead of weighted sum.
                ORDI : just concatenate the result of all. [] == Concatenate
                now we get the two batches, relatively:
                ```hidden_input``` : ```batch_size * input_length * dimension```
                ```a_hidden_input``` : ```batch_size * aug_num * aug_length * dimension```
            """
            if self.merge_type == 'FLAT':
                a_hidden_input = a_hidden_input.view(a_hidden_input.shape[0], a_hidden_input.shape[1] * a_hidden_input.shape[2], a_hidden_input.shape[3])
                a_hidden_input = torch.transpose(a_hidden_input,1,2)
                temp_feature = torch.matmul(hidden_input, a_hidden_input)
                prob_feature = F.softmax(temp_feature,dim=2)
            else:
                hidden_input = hidden_input.view(hidden_input.shape[0],1,hidden_input.shape[1],hidden_input.shape[2])
                a_hidden_input = torch.transpose(a_hidden_input,2,3)
                temp_feature = torch.matmul(hidden_input,a_hidden_input)
                prob_feature = F.softmax(temp_feature,dim=3)
            # so, we get the B * 2 * t * a attention matrix
            if self.merge_type == 'ORIG':
                weight = torch.sum(temp_feature,dim=3)
                weight = torch.sum(weight,dim=2)
                weight = F.softmax(weight,dim=1)
            if self.merge_type == 'FLAT':
                a_aug_emb = a_aug_emb.view(a_aug_emb.shape[0], a_aug_emb.shape[1] * a_aug_emb.shape[2], a_aug_emb.shape[3])
            aug_represent = torch.matmul(prob_feature,a_aug_emb)
            if self.merge_type == "ORDI":
                aug_represent = torch.transpose(aug_represent,1,2)
                aug_represent = aug_represent.contiguous()
                aug_represent = aug_represent.view(aug_represent.shape[0],-1,self.aug_num * self.argument_emb_size)
            if self.merge_type == 'ORIG':
                weight = weight.view(weight.shape[0],weight.shape[1],1,1)
                aug_represent = aug_represent * weight
                aug_represent = torch.sum(aug_represent,dim=1)
            if self.merge_type == 'MEAN':
                aug_represent = torch.mean(aug_represent,dim=1)
            # now we need to use this weight map to get the final refined_things
            if self.only_attention:
                final_represent = aug_represent
            else:
                final_represent = torch.cat([input_emb,aug_represent],2)
        else:
            final_represent = hidden_input

        bilstm_output2, bilstm_hidden_state2 = self.bilstm_layer2(final_represent,self._init_hidden(1, batch_size))
        hidden_input2 = bilstm_output2.contiguous()
        hidden_input2 = hidden_input2.view(bilstm_output2.shape[0]*bilstm_output2.shape[1],-1)
        #hidden_input2 = torch.transpose(hidden_input2,0,1)


        if self.use_highway:
            for current_layer in self.highway_layers:
                hidden_input2 = current_layer(hidden_input2)

            output = self.output_layer(hidden_input2)

        else:
            if self.hidden_layer1_size is None:
                output = self.output_layer(hidden_input2)
            else:
                hidden_layer1_output = self.hidden_layer1(hidden_input2)
                if self.hidden_layer2_size is None:
                    output = self.output_layer(hidden_layer1_output)
                else:
                    hidden_layer2_output = self.hidden_layer2(hidden_layer1_output)
                    if self.hidden_layer3_size is None:
                        output = self.output_layer(hidden_layer2_output)
                    else:
                        hidden_layer3_output = self.hidden_layer3(hidden_layer2_output)
                        output = self.output_layer(hidden_layer3_output)

        return output

