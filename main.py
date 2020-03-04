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
from sklearn.model_selection import train_test_split

random.seed(42)

# Python argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--input-sent", "-is")

args = parser.parse_args()

data = load_corpus()
data.pop()  # remove the last line

# STEP 2: SPLIT TRAIN/DEV/TEST
# indices = np.random.permutation(len(data))
# print(indices)
# data = data[indices]  # shuffle everyone, yay !
np.random.shuffle(data)

data_train, data_dev, data_test = np.split(
    data, [int(.8*len(data)), int(.9*len(data))])

print("Train data: {:d}".format(len(data_train)))
print("Dev data: {:d}".format(len(data_dev)))
print("Test data: {:d}".format(len(data_test)))


# STEP 3: BUILD LANGUAGE MODEL AND CFG
processed_trees, sentences = process_corpus(data_train)
language_model = train_language_model(sentences)

grammar, lexicon = build_pcfg(processed_trees)
# grammar.chomsky_normal_form()
print("Built PCFG. {:s} | {:s}".format(repr(grammar), str(lexicon)))

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
    import PYEVALB.scorer as evalscorer
    import PYEVALB.parser as evalparser
    
    
    def evaluate_predict(sentence, target_parse, cyk_module: cyk.CYKParser, scorer: evalscorer.Scorer) -> evalscorer.Result:
        predicted_string = cyk_module.cyk_parse(sentence)
        if predicted_string is not None:
            pred_tree = evalparser.create_from_bracket_string(predicted_string)
            gold_tree = evalparser.create_from_bracket_string(target_parse)
            result = scorer.score_trees(gold_tree, pred_tree)
        else:
            result = None
        return result

    
    
    input_sentence = args.input_sent    
    if input_sentence is not None:
        input_sentence = input_sentence.split()
        cyk_module.cyk_parse(input_sentence)
    
    import random
    
    # parse like 10 sentences from train set to check
    for _ in range(10):
        ins_trees = data_dev
        ins_sents = sentence_dev
        
        idx = random.randint(0, len(ins_trees))
        
        print()
        print("Parsing train sentence #%d" % idx)

        scorer = evalscorer.Scorer()
        # Perform CYK prediction
        sent_ = ins_sents[idx]
        target_ = ins_trees[idx][2:-1]
        res_ = evaluate_predict(sent_, target_, cyk_module, scorer)
        print(res_)
    
    # summary_ = evalscorer.summary.summary([res_])
    # print(summary_)
