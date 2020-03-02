import numpy as np
from preprocess import load_corpus
from preprocess import process_corpus, get_productions
from preprocess import get_symbols, train_language_model
import argparse
import nltk
from nltk import Nonterminal
from oov import OOVModule
import cyk
from typing import List


parser = argparse.ArgumentParser()

def parse_sentence(sentence: str, oov: OOVModule):
    sentence = sentence.split()  # split on whitespace
    sentence_oov = oov.get_replacement_token()


data = load_corpus()
data.pop()  # remove the last line

# STEP 2: SPLIT TRAIN/DEV/TEST
data_train, data_dev, data_test = np.split(
    data, [int(.8*len(data)), int(.9*len(data))])

print("Data train {:d}".format(len(data_train)))
print("Data dev {:d}".format(len(data_dev)))
print("Data test {:d}".format(len(data_test)))


# STEP 3: BUILD LANGUAGE MODEL AND CFG
processed_trees, sentences = process_corpus(data_train)
language_model = train_language_model(sentences)


treebank_rules = get_productions(processed_trees)
S = Nonterminal('+SENT')
grammar = nltk.induce_pcfg(S, treebank_rules)

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

nonterminal_symb, terminal_symb = get_symbols(grammar.productions())
oov_module = OOVModule(terminal_symb, language_model)
cyk_module = cyk.CYKParser(grammar, oov_module, unary_idx, binary_idx, nonterminal_symb)

# cyk_module.cyk_parse("Pourquoi ce thème ?".split())
cyk_module.cyk_parse("Gutenberg est mort .".split())
