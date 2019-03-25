import os
from data_utils import *

if __name__ == '__main__':

    train_file = './data/train.txt'
    dev_file = './data/dev.txt'
    test_file = './data/test.txt'

    flat_dataset('./data/train.txt', './temp/train.flat.txt')
    flat_dataset('./data/dev.txt', './temp/dev.flat.txt')
    flat_dataset('./data/test.txt', './temp/test.flat.txt')

    # make word/pos/lemma/deprel/argument vocab
    print('\n-- making (word/lemma/pos/argument) vocab --')
    vocab_path = os.path.join(os.path.dirname(__file__),'temp')
    print('word:')
    make_word_vocab(train_file,vocab_path)
    print('pos:')
    make_pos_vocab(train_file,vocab_path)
    print('lemma:')
    make_lemma_vocab(train_file,vocab_path)
    print('deprel:')
    make_deprel_vocab(train_file,vocab_path)
    print('argument:')
    make_argument_vocab(train_file, dev_file, test_file, vocab_path)

    # shrink pretrained embeding
    print('\n-- shrink pretrained embeding --')
    pretrain_file = os.path.join(os.path.dirname(__file__),'data/glove.100d.txt')
    pretrained_emb_size = 100
    pretrain_path = os.path.join(os.path.dirname(__file__),'temp')
    shrink_pretrained_embedding(train_file,dev_file,test_file,pretrain_file,pretrained_emb_size, pretrain_path)