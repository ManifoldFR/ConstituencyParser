import numpy as np
from preprocess import load_corpus
from preprocess import process_corpus, build_pcfg
from preprocess import train_language_model
import argparse
import nltk
from nltk import Nonterminal
import oov
import cyk
from typing import List
import random

random.seed(42)

# Python argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--input-sent", "-is")

args = parser.parse_args()

data = load_corpus()
data.pop()  # remove the last line

# STEP 2: SPLIT TRAIN/DEV/TEST
data_train, data_dev, data_test = np.split(
    data, [int(.8*len(data)), int(.9*len(data))])

print("Train data: {:d}".format(len(data_train)))
print("Dev data: {:d}".format(len(data_dev)))
print("Test data: {:d}".format(len(data_test)))


# STEP 3: BUILD LANGUAGE MODEL AND CFG
processed_trees, sentences = process_corpus(data_train)
language_model = train_language_model(sentences)

grammar, lexicon = build_pcfg(processed_trees)

nonterminal_symb = list(grammar._categories)  # LHSs of the grammar rules are all nonterminal symbols
terminal_symb = lexicon.pos()  # PoS ; the lexicon already has the list of PoS pre-computed
vocab = lexicon.tokens()  # stripped vocabulary

# filter productions between unary and binary
unary_idx = []
binary_idx = []
for i, prod in enumerate(grammar.productions()):
    if len(prod.rhs()) == 1:
        unary_idx.append(i)
    elif len(prod.rhs()) == 2:
        binary_idx.append(i)
    else:
        raise ValueError("The grammar is not CNF!")

oov_module = oov.OOVModule(terminal_symb, language_model, vocab)

cyk_module = cyk.CYKParser(grammar, oov_module, lexicon, unary_idx, binary_idx,
                           nonterminal_symb, terminal_symb)



_, sentence_dev = process_corpus(data_dev)
_, sentence_test = process_corpus(data_test)

if __name__ == "__main__":
    input_sentence = args.input_sent.split()
    
    cyk_module.cyk_parse(input_sentence)

