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
    """Pre-process the line and extract the tree and sentence (token list)."""
    line = _clean_line(line)
    t: Tree = Tree.fromstring(line, remove_empty_top_bracketing=True)  # remove bracketing because nltk gets the wrong start symbol!
    t.chomsky_normal_form(horzMarkov=2)
    t.collapse_unary(collapsePOS=True, collapseRoot=False)
    sentence = t.leaves()  # tokenized sentence, good for making language models with :)
    return t, sentence


def process_corpus(lines, tqdm_desc=None):
    """Process the treebank corpus.
    
    Returns
    -------
    The list of parsed trees (type `nltk.tree.Tree`), and sentences to use for training
    a language model.
    """
    parsed_trees = []
    sentences = []
    import tqdm
    for line in tqdm.tqdm(lines, desc=tqdm_desc):
        t, sentence = _process_line(line)
        parsed_trees.append(t)
        sentences.append(sentence)
    return parsed_trees, sentences


def build_pcfg(data: List[Tree]):
    """Get productions from the list of parsed trees, build a PCFG and probabilistic lexicon.
    
    Parameters
    ----------
    data
        List of NLTK trees (extracted from the annotated corpus).
    """
    rules: List[nltk.Production] = []
    lexicon_rules: List[nltk.Production] = []
    # Iterate over the trees, separate lexical rules from others
    for t in data:
        for prod in t.productions():
            if prod.is_lexical():
                lexicon_rules.append(prod)
            else:
                rules.append(prod)

    S = Nonterminal('SENT')
    grammar = nltk.induce_pcfg(S, rules)  # using NLTK's convenience function that counts the rules and LHSs
    lexicon = ProbabilisticLexicon(lexicon_rules)  # our own data structure
    return grammar, lexicon


class ProbabilisticLexicon(object):
    """Record of rules of the form `PoS -> token` with assigned probabilities."""
    
    def __init__(self, lexical_rules: List[nltk.Production]):
        super().__init__()
        self._raw_rules = lexical_rules.copy()  # all the rules, with duplicates
        self._tokens = set()
        self._pos = set()

        self._compute_frequencies()

    def _compute_frequencies(self):
        """Code inspired by NLTK's `nltk.grammar.induce_pcfg` for counting production frequencies.
        
        Also normalizes the input tokens to lower case."""
        from collections import defaultdict
        token_count = defaultdict(int)
        prod_count = defaultdict(int)
        
        _tokens = []
        _pos = []
        
        for idx, rule in enumerate(self._raw_rules):
            token = rule._rhs[0].lower()
            pos_ = rule._lhs
            rule._rhs = (token,)  # modify the token to lowercase
            # it is necessary to use (pos, token) as keys
            # because for hashcode reasons
            prod_count[pos_, token] += 1
            _tokens.append(token)
            _pos.append(pos_)
            token_count[token] += 1
        
        ## IMPORTANT: do not use Python set
        ## because the hash function is randomly and uncontrollably 
        ## seeded, which changes the token and PoS ordering
        
        self._tokens = list(np.unique(_tokens))
        self._pos = list(np.unique(_pos))
        
        # use ProbabilisticProduction to represent the triple (PoS (lhs), token (rhs), probability)
        _proba_prods = set()
        for rule in self._raw_rules:
            rhs_ = rule._rhs
            pos_ = rule._lhs
            token = rhs_[0]
            _proba_prods.add(
                nltk.ProbabilisticProduction(
                    rule._lhs, rhs_, prob=prod_count[pos_, token] / token_count[token])
            )
        self._proba_prods = list(_proba_prods)
        
        # compute map from tokens to distribution of PoS
        self._token_pos_map = defaultdict(dict)
        for prod in self._proba_prods:
            token = prod._rhs[0].lower()
            lhs_ = prod._lhs
            self._token_pos_map[token][lhs_] = prod.prob()
    
    def get_pos_distribution(self, token) -> dict:
        return self._token_pos_map[token.lower()]
    
    def rules(self) -> List[nltk.ProbabilisticProduction]:
        return self._proba_prods
    
    def tokens(self) -> List[str]:
        """Return the token vocabulary."""
        return list(self._tokens)

    def pos(self) -> List[str]:
        """Return the list of parts-of-speech (PoS)"""
        return list(self._pos)

    def __repr__(self):
        return "<Lexicon with {:d} tokens and {:d} PoS>".format(
            len(self._tokens), len(self._pos))


def train_language_model(sentences, method="wittenbell"):
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
    else:
        model = lm.Laplace(2)
    model.fit(train_grams, vocab)
    return model
