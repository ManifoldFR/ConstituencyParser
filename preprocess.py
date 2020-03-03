import pickle
import numpy as np
import nltk
from nltk import Tree, Nonterminal, ngrams
import re
from typing import List
from nltk.lm.preprocessing import padded_everygram_pipeline


# EMBEDDINGS

## For polyglot embeddings
Token_ID = {"<UNK>": 0, "<S>": 1, "</S>": 2, "<PAD>": 3}
ID_Token = {v: k for k, v in Token_ID.items()}  # reverse map
# Digit strings will be normalized by replacing them with #
DIGITS = re.compile("[0-9]", re.UNICODE)


def load_embeddings():
    with open('polyglot-fr.pkl', 'rb') as f:
        words, embeddings = pickle.load(f, encoding='latin1')
    print("Embeddings shape is {}".format(embeddings.shape))
    return words, embeddings


# CORPUS AND GRAMMAR DEFINITION

def load_corpus():
    with open("sequoia-corpus+fct.mrg_strict") as f:
        data = f.read().split('\n')
    return data


def _clean_line(line):
    """Strip functional labels from non-terminal names."""
    expr = "\(\w+(-\w+)"
    prog = re.compile(expr)
    match = prog.findall(line)
    line_clean = line.copy()
    for s in match:
        line_clean = line_clean.replace(s, "")
    return line_clean


def _process_line(line):
    """Pre-process the line and extract the tree."""
    line = _clean_line(line)
    t: Tree = Tree.fromstring(line)
    t.chomsky_normal_form(horzMarkov=2)
    t.collapse_unary(collapsePOS=False, collapseRoot=False)  # we use this for counting
    sentence = t.leaves()  # list of tokens!
    # import ipdb; ipdb.set_trace()
    return t, sentence


def process_corpus(lines):
    """Process the treebank corpus.
    
    Returns
    -------
    The list of parsed trees (type `nltk.tree.Tree`), and sentences to use for training
    a language model.
    """
    res = []
    sentences = []
    for line in lines:
        proc_line, sentence = _process_line(line)
        res.append(proc_line)
        sentences.append(sentence)
    return res, sentences


def get_productions(data):
    """Extract the productions (rewriting rules) from a list of parsed trees.
    
    Parameters
    ----------
    data
        List of NLTK trees (extracted from the annotated corpus).
    """
    rules = set()
    for it in data:
        for prod in it.productions():
            rules.add(prod)
            # import ipdb; ipdb.set_trace()
    return rules

def build_pcfg(data):
    """Get a Probabilistic Context-Free Grammar (PCFGs) from the list of parsed trees.
    
    Parameters
    ----------
    data
        List of NLTK trees extracted from the corpus.
    """
    # dictionary of counts of (lhs, rhs) occurences
    from collections import defaultdict
    counts = defaultdict(dict)
    total_counts = defaultdict(int)
    
    rules = set()
    for it in data:
        for prod in it.productions():
            rules.add(prod)
            lhs = prod.lhs().symbol()
            if prod.is_lexical():
                # for lexical nodes we have a token
                rhs = prod.rhs()[0].symbol().lower()
            else:
                rhs = tuple(l.symbol() for l in prod.rhs())
            if rhs not in counts[lhs]:
                counts[lhs][rhs] = 0
            else:
                counts[lhs][rhs] += 1
                total_counts[rhs] += 1
    
    
    # STEP 2: renormalize probabilities
    ## loop over rhs

    return rules


def get_symbols(productions: List[nltk.ProbabilisticProduction]):
    """Extract the symbols from a rule.
    
    Returns Python sets for easy updates.
    """
    
    def _get_symbols(rule: nltk.ProbabilisticProduction):
        non_terminal = [rule.lhs().symbol()]
        if rule.is_lexical():
            # rhs contains terminal token
            rhs_is_word = True
            terminal = set(map(str.lower, rule.rhs()))
        else:
            terminal = set()  # no terminal nodes
            rhs_is_word = False
            for item in rule.rhs():
                non_terminal.append(item.symbol())
        non_terminal = set(non_terminal)
        return rhs_is_word, non_terminal, terminal

    nonterm_symb = set()
    term_symb = set()
    for rule in productions:
        is_word, nt, t = _get_symbols(rule)
        nonterm_symb.update(nt)
        term_symb.update(t)
    return nonterm_symb, term_symb
    


def train_language_model(sentences, method="laplace"):
    """Extract unigrams and bigrams and train a language model using NLTK helpers.
    The returned language model has helper methods to compute unigram and bigram scores.
    
    Returns
    -------
    model
        A language model. We use the `nltk.lm.Laplace` class to use smoothed probability scores.
    """
    sentences = [[s.lower() for s in sent] for sent in sentences]
    train_grams, vocab = padded_everygram_pipeline(2, sentences)
    from nltk import lm
    if method == "wittenbell":
        model = lm.WittenBellInterpolated(2)
    elif method == "laplace":
        model = lm.Laplace(2)
    else:
        raise ValueError("Unknown language model type")
    model.fit(train_grams, vocab)
    return model
