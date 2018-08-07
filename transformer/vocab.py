# coding=utf-8
import codecs
import logging
import os
from argparse import ArgumentParser
from collections import Counter

import yaml

from utils import AttrDict


def make_vocab(fpath, fname):
    """Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    for l in codecs.open(fpath, 'r', 'utf-8'):
        words = l.split()
        word2cnt.update(Counter(words))
    word2cnt.update({"<PAD>": 10000000000000,
                     "<UNK>": 1000000000000,
                     "<S>": 100000000000,
                     "</S>": 10000000000})
    with codecs.open(fname, 'w', 'utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{}\t{}\n".format(word, cnt))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))


def make_word(fpath, fname):
    """Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    for l in codecs.open(fpath, 'r', 'utf-8'):
        l = l.strip()
        words = l.split()
        words = words[1:]
        word2cnt.update(Counter(words))
    word2cnt.update({"<PAD>": 10000000000000,
                     "<UNK>": 1000000000000,
                     "<S>": 100000000000,
                     "</S>": 10000000000})
    with codecs.open(fname, 'w', 'utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{}\t{}\n".format(word, cnt))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))


def make_word_multilang(fpath, fname):
    """Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    for l in codecs.open(fpath, 'r', 'utf-8'):
        l = l.strip()
        words = l.split()
        words = words[1:]
        word2cnt.update(Counter(words))
    word2cnt.update({"<PAD>": 10000000000000,
                     "<UNK>": 1000000000000,
                     "<S>": 100000000000,
                     "</S>": 10000000000,
                     "<S_MA>": 1000000009,
                     "<S_EN>": 1000000008,
                     "<S_JA>": 1000000007,
                     "<S_AR>": 1000000006,
                     "<S_GE>": 1000000005,
                     "<S_SP>": 1000000004})
    with codecs.open(fname, 'w', 'utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{}\t{}\n".format(word, cnt))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))


def make_word_ma_en(fpath, fname):
    """Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    for l in codecs.open(fpath, 'r', 'utf-8'):
        l = l.strip()
        words = l.split()
        words = words[1:]
        word2cnt.update(Counter(words))
    word2cnt.update({"<PAD>": 10000000000000,
                     "<UNK>": 1000000000000,
                     "<S>": 100000000000,
                     "</S>": 10000000000,
                     "<S_MA>": 1000000009,
                     "<S_EN>": 1000000008
                     })
    with codecs.open(fname, 'w', 'utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{}\t{}\n".format(word, cnt))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))


def make_word_count():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/train/text'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/train/word_count.txt'
    make_word(text_path_in, word_path_out)


def make_word_count_thchs30():
    text_path_in = '/data/syzhou/work/data/kaldi-trunk/egs/data_process_fzy/s5/data/train_sp/text.char'
    word_path_out = '/data/syzhou/work/data/kaldi-trunk/egs/data_process_fzy/s5/data/train_sp/words_s2s_thchs30.txt'
    make_word(text_path_in, word_path_out)


def make_word_count_aishell2():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/AISHELL2/aishell2_s2s/data/train_dim80/text.char'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/AISHELL2/aishell2_s2s/words_s2s.txt'
    make_word(text_path_in, word_path_out)


def make_pinyin_count():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/pinyin/hkust.train.text.pinyin'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/pinyin/pinyin_s2s.txt'
    make_word(text_path_in, word_path_out)


def make_word_count_dim80():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/train_dim80/text'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/train_dim80/word_count.txt'
    make_word(text_path_in, word_path_out)


def make_word_count_en():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s/data_en/data/train/text'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s/data_en/data/train/words.txt'
    make_word(text_path_in, word_path_out)


def make_word_count_CALLHOME(lang):
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s/data_' + lang + '/data/train/text'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s/data_' + lang + '/data/train/words.txt'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_10000():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/train_dim80/bpe/bpe_10000/text.bpe_10000'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/train_dim80/bpe/bpe_10000/words_s2s.txt.bpe_10000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_5000():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/train_dim80/bpe/bpe_5000/text.bpe_5000'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/hkust_ci_phone/src_data/train_dim80/bpe/bpe_5000/words_s2s.txt.bpe_5000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_5000_spanish():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_sp/data/train_dim80/text.bpe_5000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_sp/words_s2s.txt.bpe_5000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_5000_english():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/data/train_dim80/text.bpe_5000'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/words_s2s.txt.bpe_5000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_3000_english():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/data/train_dim80/text.bpe_3000'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/words_s2s.txt.bpe_3000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_2000_english():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/data/train_dim80/text.bpe_2000'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/words_s2s.txt.bpe_2000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_100_english():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/data/train_dim80/text.bpe_100'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/words_s2s.txt.bpe_100'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_500_english():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/data/train_dim80/text.bpe_500'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/words_s2s.txt.bpe_500'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_50_english():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/data/train_dim80/text.bpe_50'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/words_s2s.txt.bpe_50'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_1000_english():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/data/train_dim80/text.bpe_1000'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/words_s2s.txt.bpe_1000'
    make_word(text_path_in, word_path_out)


def make_word_count_english():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/data/train_dim80/text'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_en/words_s2s.txt'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_500_spanish():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_sp/data/train_dim80/text.bpe_500'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_sp/words_s2s.txt.bpe_500'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_500_ar():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_ar/data/train_dim80/text.bpe_500'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_ar/words_s2s.txt.bpe_500'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_500_ge():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_ge/data/train_dim80/text.bpe_500'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_ge/words_s2s.txt.bpe_500'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_500_ja():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_ja/data/train_dim80/text.bpe_500'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_ja/words_s2s.txt.bpe_500'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_500_ma():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_ma/data/train_dim80/text.bpe_500'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/CALLHOME/s2s_new/data_ma/words_s2s.txt.bpe_500'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_5000_5lang():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/data/train_dim80/text.bpe_5000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/words_s2s.txt.bpe_5000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_3000_5lang():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/data/train_dim80/text.bpe_3000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/words_s2s.txt.bpe_3000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_7000_5lang():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/data/train_dim80/text.bpe_7000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/words_s2s.txt.bpe_7000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_9000_5lang():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/data/train_dim80/text.bpe_9000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/words_s2s.txt.bpe_9000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_1000_5lang():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/data/train_dim80/text.bpe_1000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/words_s2s.txt.bpe_1000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_5000_5lang_lid():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/data/train_dim80/text.bpe_5000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_5lang/words_s2s_lid.txt.bpe_5000'
    make_word_multilang(text_path_in, word_path_out)


def make_word_count_bpe_7000_6lang_lid():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/data/train_dim80/text.bpe_7000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/words_s2s_lid.txt.bpe_7000'
    make_word_multilang(text_path_in, word_path_out)


def make_word_count_bpe_3000_6lang_lid():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/data/train_dim80/text.bpe_3000'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/words_s2s_lid.txt.bpe_3000'
    make_word_multilang(text_path_in, word_path_out)


def make_word_count_bpe_12000_swbd_5lang_lid():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/data/train_dim80_swbd_callhm_5lang/text.bpe_12000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/words_s2s_lid.txt.swbd_5lang.bpe_12000'
    make_word_multilang(text_path_in, word_path_out)


def make_word_count_bpe_3000_swbd_5lang_lid():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/data/train_dim80_swbd_callhm_5lang/text.bpe_3000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/words_s2s_lid.txt.swbd_5lang.bpe_3000'
    make_word_multilang(text_path_in, word_path_out)


def make_word_count_bpe_6lang():
    for count in [1000, 3000, 5000, 7000, 9000]:
        text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/data/train_dim80/text.bpe_' + str(
            count)
        word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_6lang/words_s2s.txt.bpe_' + str(
            count)
        make_word(text_path_in, word_path_out)


def make_word_count_bpe_3000_swbd():
    text_path_in = '/data/syzhou/work/data/tensor2tensor/SWBD/data/train_dim80/text.bpe_3000'
    word_path_out = '/data/syzhou/work/data/tensor2tensor/SWBD/words_s2s.txt.bpe_3000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_6000_swbd():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/data/train_dim80/text.bpe_6000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/words_s2s.txt.bpe_6000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_9000_swbd():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/data/train_dim80/text.bpe_9000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/words_s2s.txt.bpe_9000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_12000_swbd():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/data/train_dim80/text.bpe_12000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/words_s2s.txt.bpe_12000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_15000_swbd():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/data/train_dim80/text.bpe_15000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/words_s2s.txt.bpe_15000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_18000_swbd():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/data/train_dim80/text.bpe_18000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/words_s2s.txt.bpe_18000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_21000_swbd():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/data/train_dim80/text.bpe_21000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/words_s2s.txt.bpe_21000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_30000_swbd():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/data/train_dim80/text.bpe_30000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/SWBD/words_s2s.txt.bpe_30000'
    make_word(text_path_in, word_path_out)


def make_word_count_bpe_3000_swbd_callhm_ma_lid():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_2lang/data/train_dim80_swbd_callhm_ma/text.bpe_3000.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_2lang/words_s2s_lid.txt.swbd_callhm_ma.bpe3k'
    make_word_ma_en(text_path_in, word_path_out)


def make_word_count_bpe_500_swbd_callhm_ma_lid():
    text_path_in = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_2lang/data/train_dim80_swbd_callhm_ma/text.bpe_500.sp'
    word_path_out = '/mnt/lustre/xushuang2/syzhou/tensor2tensor/CALLHOME/s2s_new/data_2lang/words_s2s_lid.txt.swbd_callhm_ma.bpe500'
    make_word_ma_en(text_path_in, word_path_out)


if __name__ == '__main__':
    # make_word_count()
    # make_pinyin_count()
    # make_word_count_dim80()
    # make_word_count_en()
    # make_word_count_CALLHOME('ma')
    # make_word_count_CALLHOME('ja')
    # make_word_count_CALLHOME('ar')
    # make_word_count_CALLHOME('ge')
    # make_word_count_CALLHOME('sp')
    # make_word_count_CALLHOME('6lang')
    # make_word_count_bpe_10000()
    # make_word_count_bpe_5000()
    # make_word_count_bpe_5000_spanish()
    # make_word_count_bpe_5000_english()
    # make_word_count_bpe_3000_english()
    # make_word_count_bpe_2000_english()
    # make_word_count_bpe_100_english()
    # make_word_count_bpe_500_english()
    # make_word_count_bpe_50_english()
    # make_word_count_bpe_1000_english()
    # make_word_count_english()
    # make_word_count_bpe_500_spanish()
    # make_word_count_bpe_500_ar()
    # make_word_count_bpe_500_ge()
    # make_word_count_bpe_500_ja()
    # make_word_count_bpe_5000_5lang()
    # make_word_count_bpe_3000_5lang()
    # make_word_count_bpe_7000_5lang()
    # make_word_count_bpe_9000_5lang()
    # make_word_count_bpe_1000_5lang()
    # make_word_count_bpe_5000_5lang()
    # make_word_count_bpe_5000_5lang_lid()
    # make_word_count_bpe_6lang()
    # make_word_count_bpe_7000_6lang_lid()
    # make_word_count_bpe_3000_6lang_lid()
    # make_word_count_bpe_3000_swbd()
    # make_word_count_bpe_6000_swbd()
    # make_word_count_bpe_9000_swbd()
    # make_word_count_bpe_12000_swbd()
    # make_word_count_bpe_15000_swbd()
    # make_word_count_bpe_18000_swbd()
    # make_word_count_bpe_21000_swbd()
    # make_word_count_bpe_30000_swbd()
    # make_word_count_bpe_12000_swbd_5lang_lid()
    # make_word_count_bpe_3000_swbd_5lang_lid()
    # make_word_count_bpe_3000_swbd_callhm_ma_lid()
    # make_word_count_bpe_500_swbd_callhm_ma_lid()
    # make_word_count_thchs30()
    make_word_count_aishell2()
    print ''

    # from ctypes import cdll
    #
    # cdll.LoadLibrary('/usr/local/cuda/lib64/libcudnn.so.6')
    # import os
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    #
    # parser = ArgumentParser()
    # parser.add_argument('-c', '--config', dest='config')
    # args = parser.parse_args()
    # # Read config
    # args.config = '/data/syzhou/work/data/tensor2tensor/transformer/config_template.yaml'
    # config = AttrDict(yaml.load(open(args.config)))
    # logging.basicConfig(level=logging.INFO)
    # if os.path.exists(config.src_vocab):
    #     logging.info('Source vocab already exists at {}'.format(config.src_vocab))
    # else:
    #     make_vocab(config.train.src_path, config.src_vocab)
    # if os.path.exists(config.dst_vocab):
    #     logging.info('Destination vocab already exists at {}'.format(config.dst_vocab))
    # else:
    #     make_vocab(config.train.dst_path, config.dst_vocab)
    # logging.info("Done")
