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


# Python argument parser  |  parameters from "run.sh" are passed onto this
parser = argparse.ArgumentParser()
parser.add_argument("--input-sent", "-is", required=False)
parser.add_argument("--seed", help="Set a random seed (default: %(default)d)", default=42)
group1 = parser.add_argument_group("evaluation")
group1.add_argument("--dataset", choices=["train", "dev", "test"],
                    help="Specify a dataset to evaluate (default: %(default)s)")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# STEP 1: LOAD DATA, SPLIT TRAIN/DEV/TEST
data = load_corpus()
data.pop()  # remove the last line

np.random.shuffle(data)  # shuffle all the sentences once | it is important the seed be fixed here

# Split test data off before shuffling again to prevent contamination
data_train, data_test = np.split(
    data, [int(.9*len(data))])

# Shuffle train data
np.random.shuffle(data_train)

# Resplit between train and dev
data_train, data_dev = np.split(data_train, [int(8/9*len(data_train))])

# STEP 3: BUILD LANGUAGE MODEL AND CFG
processed_trees, sentences_train = process_corpus(data_train, "Train")
language_model = train_language_model(sentences_train)

_, sentences_dev = process_corpus(data_dev, "Dev")
_, sentences_test = process_corpus(data_test, "Test")

DATASET_MAP = {
    "train": (data_train, sentences_train),
    "dev": (data_dev, sentences_dev),
    "test": (data_test, sentences_test)
}

grammar, lexicon = build_pcfg(processed_trees)
grammar.chomsky_normal_form()
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
            # in that case the sentence errored out
            result = evalscorer.Result()
            result.state = 2
        return result

    
    
    input_sentence = args.input_sent    
    if input_sentence is not None:
        input_sentence = input_sentence.split()
        cyk_module.cyk_parse(input_sentence)
    
    #### DATASET EVALUATION
    
    dataset_choice = args.dataset
    if dataset_choice is not None:
        import random
        ins_trees, ins_sents = DATASET_MAP[dataset_choice]
        
        results_ = []
        
        SMOKE_TEST = True  # smoke detector 
        num_sents = len(ins_trees) if not SMOKE_TEST else 4
        for _i in range(num_sents):
            # idx = random.randint(0, len(ins_trees))
            idx = _i
            
            print("\nParsing %s set sentence #%d / %d" % (dataset_choice, idx, num_sents))

            scorer = evalscorer.Scorer()
            sent_ = ins_sents[idx]
            target_ = ins_trees[idx][2:-1]
            
            # Perform CYK prediction
            res_ = evaluate_predict(sent_, target_, cyk_module, scorer)
            print(res_)
            results_.append(res_)
        
        summary_ = evalscorer.summary.summary(results_)
        print(summary_)
