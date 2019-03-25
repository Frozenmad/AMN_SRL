import os
import pickle
import collections
import numpy as np
import random
from tqdm import tqdm

_UNK_ = '<UNK>'
_PAD_ = '<PAD>'
_ROOT_ = '<ROOT>'
_NUM_ = '<NUM>'

class Vertex:
    def __init__(self, id, head):
        self.id = id
        self.head = head
        self.children = []

def is_valid_tree(sentence, rd_node, cur_node):
    if rd_node == 0:
        return True
    if rd_node == cur_node:
        return False
    cur_head = int(sentence[rd_node-1][9])
    if cur_head == cur_node:
        return False
    while cur_head != 0:
        cur_head = int(sentence[cur_head-1][9])
        if cur_head == cur_node:
            return False
    return True


def is_scientific_notation(s):
    s = str(s)
    if s.count(',')>=1:
        sl = s.split(',')
        for item in sl:
            if not item.isdigit():
                return False
        return True   
    return False

def is_float(s):
    s = str(s)
    if s.count('.')==1:
        sl = s.split('.')
        left = sl[0]
        right = sl[1]
        if left.startswith('-') and left.count('-')==1 and right.isdigit():
            lleft = left.split('-')[1]
            if lleft.isdigit() or is_scientific_notation(lleft):
                return True
        elif (left.isdigit() or is_scientific_notation(left)) and right.isdigit():
            return True
    return False

def is_fraction(s):
    s = str(s)
    if s.count('\/')==1:
        sl = s.split('\/')
        if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
            return True  
    if s.count('/')==1:
        sl = s.split('/')
        if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
            return True    
    if s[-1]=='%' and len(s)>1:
        return True
    return False


def is_number(s):
    s = str(s)
    if s.isdigit() or is_float(s) or is_fraction(s) or is_scientific_notation(s):
        return True
    else:
        return False

def make_word_vocab(file_name, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False):

    with open(file_name,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    word_data = []
    for sentence in origin_data:
        for line in sentence:
            if not is_number(line[1].lower()):
                word_data.append(line[1].lower())
                

    word_data_counter = collections.Counter(word_data).most_common()

    if use_lower_bound:
        word_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_] + [item[0] for item in word_data_counter if item[1]>=freq_lower_bound]
    else:
        word_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_] + [item[0] for item in word_data_counter]


    word_to_idx = {word:idx for idx,word in enumerate(word_vocab)}

    idx_to_word = {idx:word for idx,word in enumerate(word_vocab)}


    if not quiet:
        print('\tword vocab size:{}'.format(len(word_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'word.vocab')

    word2idx_path = os.path.join(output_path,'word2idx.bin')

    idx2word_path = os.path.join(output_path,'idx2word.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(word_vocab))

    with open(word2idx_path,'wb') as f:
        pickle.dump(word_to_idx,f)

    with open(idx2word_path,'wb') as f:
        pickle.dump(idx_to_word,f)


def make_pos_vocab(file_name, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False):

    with open(file_name,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    pos_data = []
    for sentence in origin_data:
        for line in sentence:
            pos_data.append(line[5])
                

    pos_data_counter = collections.Counter(pos_data).most_common()

    if use_lower_bound:
        pos_vocab = [_PAD_,_UNK_,_ROOT_] + [item[0] for item in pos_data_counter if item[1]>=freq_lower_bound]
    else:
        pos_vocab = [_PAD_,_UNK_,_ROOT_] + [item[0] for item in pos_data_counter]


    pos_to_idx = {pos:idx for idx,pos in enumerate(pos_vocab)}

    idx_to_pos = {idx:pos for idx,pos in enumerate(pos_vocab)}


    if not quiet:
        print('\tpos tag vocab size:{}'.format(len(pos_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'pos.vocab')

    pos2idx_path = os.path.join(output_path,'pos2idx.bin')

    idx2pos_path = os.path.join(output_path,'idx2pos.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(pos_vocab))

    with open(pos2idx_path,'wb') as f:
        pickle.dump(pos_to_idx,f)

    with open(idx2pos_path,'wb') as f:
        pickle.dump(idx_to_pos,f)

def make_lemma_vocab(file_name, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False):

    with open(file_name,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    lemma_data = []
    for sentence in origin_data:
        for line in sentence:
            if not is_number(line[3].lower()):
                lemma_data.append(line[3].lower())
                
    lemma_data_counter = collections.Counter(lemma_data).most_common()

    if use_lower_bound:
        lemma_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_] + [item[0] for item in lemma_data_counter if item[1]>=freq_lower_bound]
    else:
        lemma_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_] + [item[0] for item in lemma_data_counter]


    lemma_to_idx = {lemma:idx for idx,lemma in enumerate(lemma_vocab)}

    idx_to_lemma = {idx:lemma for idx,lemma in enumerate(lemma_vocab)}


    if not quiet:
        print('\tlemma vocab size:{}'.format(len(lemma_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'lemma.vocab')

    lemma2idx_path = os.path.join(output_path,'lemma2idx.bin')

    idx2lemma_path = os.path.join(output_path,'idx2lemma.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(lemma_vocab))

    with open(lemma2idx_path,'wb') as f:
        pickle.dump(lemma_to_idx,f)

    with open(idx2lemma_path,'wb') as f:
        pickle.dump(idx_to_lemma,f)

def make_deprel_vocab(file_name, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False):

    with open(file_name,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    deprel_data = []
    for sentence in origin_data:
        for line in sentence:
            deprel_data.append(line[11])
                
    deprel_data_counter = collections.Counter(deprel_data).most_common()

    if use_lower_bound:
        deprel_vocab = [_PAD_,_UNK_] + [item[0] for item in deprel_data_counter if item[1]>=freq_lower_bound]
    else:
        deprel_vocab = [_PAD_,_UNK_] + [item[0] for item in deprel_data_counter]


    deprel_to_idx = {deprel:idx for idx,deprel in enumerate(deprel_vocab)}

    idx_to_deprel = {idx:deprel for idx,deprel in enumerate(deprel_vocab)}


    if not quiet:
        print('\tdeprel vocab size:{}'.format(len(deprel_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'deprel.vocab')

    deprel2idx_path = os.path.join(output_path,'deprel2idx.bin')

    idx2deprel_path = os.path.join(output_path,'idx2deprel.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(deprel_vocab))

    with open(deprel2idx_path,'wb') as f:
        pickle.dump(deprel_to_idx,f)

    with open(idx2deprel_path,'wb') as f:
        pickle.dump(idx_to_deprel,f)

def make_argument_vocab(train_file, dev_file, test_file, output_path, freq_lower_bound=0, quiet=False, use_lower_bound = False):

    argument_data = []

    with open(train_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    
    for sentence in origin_data:
        for line in sentence:
            for i in range(len(line)-14):
                argument_data.append(line[14+i])

    if dev_file is not None:
        with open(dev_file,'r') as f:
            data = f.readlines()

        origin_data = []
        sentence = []
        for i in range(len(data)):
            if len(data[i].strip())>0:
                sentence.append(data[i].strip().split('\t'))
            else:
                origin_data.append(sentence)
                sentence = []

        if len(sentence) > 0:
            origin_data.append(sentence)

        
        for sentence in origin_data:
            for line in sentence:
                for i in range(len(line)-14):
                    argument_data.append(line[14+i])

    if test_file is not None:
        with open(test_file,'r') as f:
            data = f.readlines()

        origin_data = []
        sentence = []
        for i in range(len(data)):
            if len(data[i].strip())>0:
                sentence.append(data[i].strip().split('\t'))
            else:
                origin_data.append(sentence)
                sentence = []

        if len(sentence) > 0:
            origin_data.append(sentence)

        for sentence in origin_data:
            for line in sentence:
                for i in range(len(line)-14):
                    argument_data.append(line[14+i])
                
    argument_data_counter = collections.Counter(argument_data).most_common()

    if use_lower_bound:
        argument_vocab = [_PAD_,_UNK_] + [item[0] for item in argument_data_counter if item[1]>=freq_lower_bound]
    else:
        argument_vocab = [_PAD_,_UNK_] + [item[0] for item in argument_data_counter]


    argument_to_idx = {argument:idx for idx,argument in enumerate(argument_vocab)}

    idx_to_argument = {idx:argument for idx,argument in enumerate(argument_vocab)}


    if not quiet:
        print('\targument vocab size:{}'.format(len(argument_vocab)))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'argument.vocab')

    argument2idx_path = os.path.join(output_path,'argument2idx.bin')

    idx2argument_path = os.path.join(output_path,'idx2argument.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(argument_vocab))

    with open(argument2idx_path,'wb') as f:
        pickle.dump(argument_to_idx,f)

    with open(idx2argument_path,'wb') as f:
        pickle.dump(idx_to_argument,f)

def count_sentence_predicate(sentence):
    count = 0
    for item in sentence:
        if item[12] == 'Y':
            assert item[12] == 'Y' and item[13] != '_'
            count += 1
    return count

def shrink_pretrained_embedding(train_file, dev_file, test_file, pretrained_file, pretrained_emb_size, output_path, quiet=False):
    word_set = set()
    with open(train_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())
    with open(dev_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())

    with open(test_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())

    pretrained_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_]
    pretrained_embedding = [
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size
                        ]

    with open(pretrained_file,'r') as f:
        for line in f.readlines():
            row = line.split(' ')
            word = row[0].lower()
            if word in word_set:
                pretrained_vocab.append(word)
                weight = [float(item) for item in row[1:]]
                assert(len(weight)==pretrained_emb_size)
                pretrained_embedding.append(weight)

    pretrained_embedding = np.array(pretrained_embedding,dtype=float)

    pretrained_to_idx = {word:idx for idx,word in enumerate(pretrained_vocab)}

    idx_to_pretrained = {idx:word for idx,word in enumerate(pretrained_vocab)}

    if not quiet:
        print('\tshrink pretrained vocab size:{}'.format(len(pretrained_vocab)))
        print('\tdataset sum:{} pretrained cover:{} coverage:{:.3}%'.format(len(word_set),len(pretrained_vocab),len(pretrained_vocab)/len(word_set)*100))

    if not quiet:
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'pretrain.vocab')

    pretrain2idx_path = os.path.join(output_path,'pretrain2idx.bin')

    idx2pretrain_path = os.path.join(output_path,'idx2pretrain.bin')

    pretrain_emb_path = os.path.join(output_path,'pretrain.emb.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(pretrained_vocab))

    with open(pretrain2idx_path,'wb') as f:
        pickle.dump(pretrained_to_idx,f)

    with open(idx2pretrain_path,'wb') as f:
        pickle.dump(idx_to_pretrained,f)

    with open(pretrain_emb_path,'wb') as f:
        pickle.dump(pretrained_embedding,f)


def flat_dataset(dataset_file, output_path):
    with open(dataset_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    with open(output_path, 'w') as f:
        for sidx in tqdm(range(len(origin_data))):
            sentence = origin_data[sidx]
            predicate_idx = 0            
            for i in range(len(sentence)):
                if sentence[i][12] == 'Y':
                    output_block = []
                    for j in range(len(sentence)):
                        ID = sentence[j][0] # ID
                        IS_PRED = 0
                        if i == j:
                            IS_PRED = 1

                        word = sentence[j][1].lower() # FORM
                        if is_number(word):
                            word = _NUM_
                        
                        lemma = sentence[j][3].lower() # PLEMMA
                        if is_number(lemma):
                            lemma = _NUM_

                        pos = sentence[j][5] # PPOS

                        deprel = sentence[j][11] # PDEPREL

                        tag = sentence[j][14+predicate_idx] # APRED

                        output_block.append([str(sidx), str(predicate_idx), str(len(sentence)), ID, str(IS_PRED), word, lemma, pos, deprel, tag])         
                    
                    for item in output_block:
                        f.write('\t'.join(item))
                        f.write('\n')
                    f.write('\n')
                    predicate_idx += 1


def stat_max_order(dataset_file):
    with open(dataset_file,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    max_order = 0

    for sidx in tqdm(range(len(origin_data))):
        sentence = origin_data[sidx]
        predicate_idx = 0

        for i in range(len(sentence)):
            if sentence[i][12] == 'Y':
                
                argument_set = set()
                for j in range(len(sentence)):
                    if sentence[j][14+predicate_idx] != '_':
                        argument_set.add(int(sentence[j][0]))
                
                cur_order = 1
                while True:
                    found_set = set()
                    son_data = []
                    order_idx = 0
                    while order_idx < cur_order:
                        son_order = [[] for _ in range(len(sentence)+1)]
                        for j in range(len(sentence)):
                            if len(son_data) == 0:
                                son_order[int(sentence[j][9])].append(int(sentence[j][0]))
                            else:
                                for k in range(len(son_data[-1])):
                                    if int(sentence[j][9]) in son_data[-1][k]:
                                        son_order[k].append(int(sentence[j][0]))
                                        break
                        son_data.append(son_order)
                        order_idx += 1
                    
                    current_node = int(sentence[i][0])
                    while True:
                        for item in son_data:
                            found_set.update(item[current_node])
                        if current_node != 0:
                            current_node = int(sentence[current_node-1][9])
                        else:
                            break
                    if len(argument_set - found_set) > 0:
                        cur_order += 1
                    else:
                        break
                if cur_order > max_order:
                    max_order = cur_order
                predicate_idx += 1

    print('max order:{}'.format(max_order))




def load_dataset_input(file_path):
    with open(file_path,'r') as f:
        data = f.readlines()

    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []

    if len(sentence) > 0:
        origin_data.append(sentence)

    return origin_data

def load_dump_data(path):
    return pickle.load(open(path,'rb'))

def load_deprel_vocab(path):
    with open(path,'r') as f:
        data = f.readlines()
    
    data = [item.strip() for item in data if len(item.strip())>0 and item.strip()!=_UNK_ and item.strip()!=_PAD_]

    return data

def output_predict(path, data):
    with open(path, 'w') as f:
        for sentence in data:
            for i in range(len(sentence[0])):
                line = [str(sentence[j][i]) for j in range(len(sentence))]
                f.write('\t'.join(line))
                f.write('\n')
            f.write('\n')