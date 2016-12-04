#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import word2vec
import numpy as np
import re
import nltk

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']', '{', '}', '?', '!', '-', u'–',
                      '+', '*', '--', '\'\'', '``']
punctuation = '?.!/;:()&+'


def replace_umlauts(text):
    """
    Replace German "Umlaute"
    """
    res = text[0]
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    return res


def load_diagnoses_and_labels(diagnoses, labels, data_subset):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    diag = list(open(diagnoses, "r").readlines())
    diag = [s.strip().lower() for s in diag]
    lbl = list(open(labels, "r").readlines())
    lbl = [s.strip().lower() for s in lbl]

    # Split by words
    for i in range(len(diag)):
        line = diag[i]
        diag[i] = ''.join([j if ord(j) < 128 else ' ' for j in line])

    diag = [sentence_detector.tokenize(sent.decode('utf-8')) for sent in diag]
    diag = [replace_umlauts(sent) for sent in diag]

    # Generate labels as one-hot-vectors
    for i in range(len(lbl)):
        lbl[i] = int(lbl[i])
    a = np.array(lbl)
    b = np.zeros((len(lbl), 5))         # 5 labels 0 to 4
    b[np.arange(len(lbl)), a] = 1
    one_hot = b

    # Shuffle data
    data_size = len(one_hot)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    # transform to np array
    diag = np.array(diag)
    one_hot = np.array(one_hot)
    diag = diag[shuffle_indices]
    one_hot = one_hot[shuffle_indices]

    # Create subset
    diag = diag[:data_subset]
    one_hot = one_hot[:data_subset]

    return [diag, one_hot]


def pad_diagnoses(sentences, padding_word=u'<PAD/>'):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        pad_list = [padding_word] * num_padding

        for j in range(len(pad_list)):
            sentence += pad_list[j]

        padded_sentences.append(sentence)
    return padded_sentences, sequence_length


def build_input_data(sentences, path):
    """
    Maps sentencs and labels to vectors based on Googles pre-trained word2vec
    Trained on more than a billion words
    """
    # create Google word2vec
    print('Load German word2vec pre-implemented ...')
    model = word2vec.Word2Vec.load_word2vec_format(path, binary=True)
    print('Model created successfully; now create create sentence representation...')

    tmp2 = []
    for sentence in sentences:
        tmp = []
        for word in sentence:
            # skip unknown words
            try:
                tmp.append(np.reshape(model[word], (300, 1), 1))

            except Exception as e:
                # append same sized 300 zeros for unknown words
                tmp.append(np.reshape(np.array([0] * 300), (300, 1), 1))
                # ToDo: generate txt with unknown words
                continue

        tmp2.append(tmp)
    x = np.array(tmp2)

    return x


def load_data(model, diagnoses, labels, data_subset):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """

    # ToDo: do some further text pre-processing: e.g. it doesnt like the "1." in "Beurteilung 1."
    # if so, the rest of the sentence is chopped off

    # Load and pre-process data
    diagnoses, one_hot = load_diagnoses_and_labels(diagnoses, labels, data_subset)
    diagnoses_padded, sequence_length = pad_diagnoses(diagnoses)
    x = build_input_data(diagnoses_padded, model)

    return [x, one_hot, sequence_length]
