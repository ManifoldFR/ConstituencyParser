import numpy as np
from typing import List
import nltk
import oov
import math
from preprocess import ProbabilisticLexicon


class CYKParser(object):
    """Parser using a probabilistic variant of the Cocke-Younger-Kasami
    algorithm, to return the most probable parse tree.
    See: https://en.wikipedia.org/wiki/CYK_algorithm
    
    Attributes
    ----------
    grammar: ntlk.grammar.PCFG
        Probabilistic context-free grammar(PCFG) in Chomsky normal form.
    """
    
    def __init__(self, grammar, oov_module: oov.OOVModule, lexicon: ProbabilisticLexicon,
                 unary_idx, binary_idx, nonterminal_symb, terminal_symb, debug=False):
        """
        
        Parameters
        ----------
        grammar
            Grammar (PCFG) object.
        unary_idx
            Indices of the unary rules of the grammar.
        binary_idx
            Indices of the binary rules of the grammar.
        """
        super().__init__()
        self.grammar: nltk.PCFG = grammar
        self.lexicon = lexicon
        self.oov_module = oov_module
        self.debug: bool = debug
        
        nonterminal_symb = list(nonterminal_symb)
        
        start_symb = self.grammar.start()
        start_idx = nonterminal_symb.index(start_symb)
        nonterminal_symb.pop(start_idx)
        nonterminal_symb.insert(0, start_symb)
        
        self.all_symbols: list = nonterminal_symb + terminal_symb
        self.terminal_symb = terminal_symb
        # reverse map for the symbols
        self.symb_inverse_map = {
            sm: i for i, sm in enumerate(self.all_symbols)
        }
        self.unary_rules = unary_idx
        self.binary_rules = binary_idx


    def cyk_parse(self, sentence: List[str]):
        """
        
        Parameters
        ----------
        sentence : List[str]
            Tokenized sentence `s_1,...,s_n` (each entry is a token string).
        """
        import tqdm
        
        print("Decoding \"{:s}\"".format(' '.join(sentence)), "({:d} words)".format(len(sentence)), end=' ')
        # auxiliary sentence
        sent_aux = self.oov_module.get_replacement_tokens(sentence)
        print("Auxiliary sentence: \"{:s}\"".format(' '.join(sent_aux)))
        
        n = len(sent_aux)
        r = len(self.all_symbols)
        
        # the value matrix we build with dynamic programming prob_matrix[i, j, A]
        # records the score of the parse (sub)tree with root A and
        # lexical nodes (s_i,...,s_j)
        value = np.zeros((n, n, r))
        value[:] = -np.inf

        # backtrack_[i, j, A] records the idx-tuple of symbols (p, B, C) such that rule A -> B C
        # at split point p in the parse tree gave the best score
        backtrack = np.zeros((n, n, r, 3), dtype=int)
        
        # STEP 1: FOR EVERY TOKEN x FIND PoS A WITH RULE A -> x IN LEXICON
        for j, token in enumerate(sent_aux):
            # iterate over PoS -> x rules
            for prod in self.lexicon.rules():
                # import ipdb; ipdb.set_trace()
                if prod.rhs()[0].lower() == token:
                    # The rhs symbol is s! Record this
                    lhs_ = prod.lhs()
                    v = self.symb_inverse_map[lhs_]
                    # print(j, s, '|', v, '|', prod)
                    # The probability of the PoS is conditioned on the token
                    value[j, j, v] = prod.logprob()
        
        # STEP 2: ITERATE ON THE TRIANGLE AND LOOK AT ALL SUB-SENTENCES xi,...,xj
        # LOOK AT BINARY PRODUCTIONS AND ASSIGN SPLIT PROBABILITIES
        # LOOK AT ALL SPLITTING POINTS IN xi,...,xj
        #TODO SOME SENTENCES CAN'T BE PARSED WTH !! FIX
        for span in tqdm.trange(1, n):  # substring length - 1 i.e. number of tokens to look forward
            if self.debug:
                print("SUBSTRING LENGTH %d" % (span+1))
            for i in range(n-span):
                j = i + span  # actual last index | slice [i:j+1]
                # iterate over split indices | saying we cut at pos means between tokens sent[pos] and sent[pos+1]
                for pos in range(i, j):
                    if self.debug:
                        print("i=%d,p=%d,j=%d\t" % (i,pos,j), ' '.join(sent_aux[i:pos+1]), "<C>", ' '.join(sent_aux[pos+1:j+1]))
                    for rule_idx in self.binary_rules:
                        prod: nltk.ProbabilisticProduction = self.grammar.productions()[rule_idx]
                        lhs_ = prod._lhs
                        rhs_1 = prod._rhs[0]
                        rhs_2 = prod._rhs[1]
                        a = self.symb_inverse_map[lhs_]
                        b = self.symb_inverse_map[rhs_1]
                        c = self.symb_inverse_map[rhs_2]
                        prob_split = prod.logprob() + value[i, pos, b] + value[pos+1, j, c]
                        # Record the value of the best possible parse subtree
                        if value[i, j, a] < prob_split:
                            # print(prod, "|", i, pos, j, "|", a, b, c)
                            value[i, j, a] = prob_split
                            backtrack[i, j, a] = [pos, b, c]
        # the best split for sentence[0..n] is given by
        # the backtracking array at indices [0, n, s0] where s0 is the index of the start symbol
        start_idx = 0
        
        def _decoder_func(i, j, symbol_idx, debug=False):
            result = ""
            
            if debug:
                print("DEBUG: i=%d, j=%d, symbol=%s" % (i,j,self.all_symbols[symbol_idx]), end=' ')
            # Check this invariant for sanity
            assert i <= j
            
            if i == j:
                result += "({:s} {:s})".format(
                    self.all_symbols[symbol_idx].symbol(),
                    sentence[i])
            else:
                pos, b, c = backtrack[i, j, symbol_idx]
                if debug:
                    print("| pos=%d, b=%d, c=%d" % (pos, b, c))
                # Perform recursive calls!
                lhs_ = _decoder_func(i, pos, b)
                rhs_ = _decoder_func(pos+1, j, c)
                result += "({:s} {:s} {:s})".format(
                    self.all_symbols[symbol_idx].symbol(),
                    lhs_, rhs_)
            return result
        
        try:
            decoded_tree_bracketed_ = _decoder_func(0, n-1, start_idx)
            print(decoded_tree_bracketed_)
        except AssertionError:
            decoded_tree_bracketed_ = None
            print("Parsing failed.")
        return decoded_tree_bracketed_
    
