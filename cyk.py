import numpy as np
from typing import List
import nltk
import oov


class CYKParser(object):
    """Parser using a probabilistic variant of the Cocke-Younger-Kasami
    algorithm, to return the most probable parse tree.
    See: https://en.wikipedia.org/wiki/CYK_algorithm
    
    Attributes
    ----------
    grammar: ntlk.grammar.PCFG
        Probabilistic context-free grammar(PCFG) in Chomsky normal form.
    """
    
    def __init__(self, grammar, oov_module: oov.OOVModule, unary_idx, binary_idx, nonterminal_symb):
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
        self.oov_module = oov_module
        nonterminal_symb = list(nonterminal_symb)
        start_idx = nonterminal_symb.index("+SENT")
        nonterminal_symb.pop(start_idx)
        nonterminal_symb.insert(0, "+SENT")
        self.nonterminal_symbols: list = nonterminal_symb
        self.nonterminal_inverse_map = {
            nt: i for i, nt in enumerate(nonterminal_symb)
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
        sent_aux = self.oov_module.get_replacement_tokens(sentence)
        
        print(sent_aux)
        
        n = len(sent_aux)
        r = len(self.nonterminal_symbols)
        
        # the value matrix we build with dynamic programming prob_matrix[i, j, A]
        # records the score of the parse (sub)tree with root A and
        # lexical nodes (s_i,...,s_j)
        value = np.zeros((n, n, r))
        # backtrack_[i, j, A] records the idx-tuple of symbols (p, B, C) such that rule A -> B C
        # at split point p in the parse tree gave the best score
        backtrack = np.zeros((n, n, r, 3), dtype=int)
        
        # STEP 1: FOR EVERY TOKEN x FIND NONTERMINALS A WITH RULE A -> x
        for j, s in enumerate(sent_aux):
            for rule_idx in self.unary_rules:
                prod: nltk.ProbabilisticProduction = self.grammar.productions()[rule_idx]
                if prod.is_lexical() and prod.rhs()[0].lower() == s:
                    # The rhs symbol is s! Record this
                    lhs_ = prod.lhs().symbol()
                    v = self.nonterminal_inverse_map[lhs_]
                    print(prod, "|", j)
                    # print(j, s, '|', v, '|', prod)
                    value[j, j, v] = prod.logprob()
        # import ipdb; ipdb.set_trace()
        # STEP 2: ITERATE ON THE TRIANGLE AND LOOK AT ALL SUB-SENTENCES xi,...,xj
        # LOOK AT BINARY PRODUCTIONS AND ASSIGN SPLIT PROBABILITIES
        # LOOK AT ALL SPLITTING POINTS IN xi,...,xj
        for i in range(n):
            for j in range(i+1, n):
                for pos in range(i, j):  # split indices
                    for rule_idx in self.binary_rules:
                        prod: nltk.ProbabilisticProduction = self.grammar.productions()[rule_idx]
                        a = self.nonterminal_inverse_map[prod.lhs().symbol()]
                        b = self.nonterminal_inverse_map[prod.rhs()[0].symbol()]
                        c = self.nonterminal_inverse_map[prod.rhs()[1].symbol()]
                        prob_split = prod.logprob() + value[i, pos, b] + value[pos+1, j, c]
                        if value[i, j, a] < prob_split:
                            print(prod, "|", i, pos, j, "|", a, b, c)
                            value[i, j, a] = prob_split
                            backtrack[i, j, a] = [pos, b, c]
        
        # return value, backtrack_

        # the best split for sentence[0..n] is given by
        # the backtracking array at indices [0, n, s0] where s0 is the index of the start symbol
        # the start symbol here is '+SENT'
        start_idx = self.nonterminal_inverse_map['+SENT']  # this should be 0
        
        def _decoder_func(i, j, nt_idx):
            result = ""
            
            if i == j:
                result += "({:s} {:s})".format(
                    self.nonterminal_symbols[nt_idx],
                    sentence[i])
            else:
                pos, b, c = backtrack[i, j, nt_idx]
                # perform recursive call
                lhs_ = _decoder_func(i, pos, b)
                rhs_ = _decoder_func(pos+1, j, c)
                result += "({:s} {:s} {:s})".format(
                    self.nonterminal_symbols[nt_idx],
                    lhs_, rhs_)
            return result
        print("\nDecoding...")
        decoded_tree_bracketed_ = _decoder_func(0, n-1, start_idx)
        print(decoded_tree_bracketed_)
        return decoded_tree_bracketed_
    