import numpy as np
from preprocess import load_embeddings, DIGITS
from sklearn.neighbors import NearestNeighbors
from typing import List
from nltk.lm.api import LanguageModel


def levenshtein(s1: str, s2: str):
    """Compute the Levenshtein Edit distance between two strings.
    
    Parameters
    ----------
    s1 : str
    s2 : str
    """
    s1 = s1.lower()
    s2 = s2.lower()
    m = np.zeros((len(s1)+1, len(s2)+1))
    m[:, 0] = np.arange(len(s1)+1)
    m[0, :] = np.arange(len(s2)+1)
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                m[i, j] = min(m[i-1, j]+1, m[i, j-1]+1, m[i-1, j-1])
            else:
                m[i, j] = min(m[i-1, j]+1, m[i, j-1]+1, m[i-1, j-1]+1)
    return m[len(s1), len(s2)]



class OOVModule(object):
    """Utility class to handle proposing PoS tags for out-of-vocabulary (OOV)
    words. Combines formal proposals (based on spelling) with semantic proposals
    (based on embeddings).
    """

    def __init__(self, corpus_terminals, language_model, n_spell_neighbors=6, n_embed_neighbors=6):
        super().__init__()
        # Get embeddings
        words, embeddings = load_embeddings()
        self.words: List[str] = words
        self.embed_word_map = {w: i for i, w in enumerate(words)}
        self.embeddings = embeddings  # we don't actually use this for the KNN
        # Corpus terminals
        self.n_spell_neighbors = n_spell_neighbors
        self.n_embed_neighbors = n_embed_neighbors
        self.corpus: List[str] = corpus_terminals
        self.language_model: LanguageModel = language_model
        
        # Pre-compute a Euclidean Nearest-Neighbor search tree in (normalized) embedding space.
        ## Euclidean distance on normalized vector == cosine distance
        self.embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embed_knn: NearestNeighbors = NearestNeighbors(
            n_neighbors=self.n_embed_neighbors).fit(self.embeddings_normalized)
        

    def find_closest_spelling(self, word: str):
        """Fix a possible spelling error by finding the closest neighbors of the
        token wrt the Levenshtein distance.
        
        Parameters
        ----------
        word : str
        """
        # TODO this is a major bottleneck
        # TODO be smarter, maybe generate the proposals at edit distance < 2 instead of looping over corpus
        num_neighbors = self.n_spell_neighbors
        all_score = []
        all_neigh = []
        for other in self.corpus:
            all_neigh.append(other)
            all_score.append(levenshtein(word, other))
        idx = np.argpartition(all_score, num_neighbors)[:num_neighbors]
        score = np.asarray(all_score)[idx]
        idx_sort = np.argsort(score)  # sort the much smaller array
        score = score[idx_sort]
        neigh = np.asarray(all_neigh)[idx][idx_sort].tolist()
        return score, neigh

    def word2embed_idx(self, word: str):
        """Recover the index of the word in the embedding corpus if present, if not
        return None."""
        try:
            # First try naively to use word as a key
            return self.embed_word_map[word]
        except KeyError:
            # If it fails, use the backoff strategy
            return self._embed_backoff(word)

    def _embed_backoff(self, word: str):
        """Backoff strategy if the word is not found in the embedding corpus.
        Keeping things simple: change around the case and normalize the digits.
        
        Inspired by the Polyglot example notebook on KNN: https://nbviewer.jupyter.org/gist/aboSamoor/6046170"""
        # STEP 1: NORMALIZE THE DIGITS
        word = DIGITS.sub("#", word)

        # STEP 2: NORMALIZE THE CASE
        proposals = [word.lower(), word.upper(), word.title()]
        ## filter out the "None" and assemble list
        proposed_idx = sorted(
            filter(None, [self.embed_word_map.get(w, None) for w in proposals]))
        
        if len(proposed_idx) > 0:
            return proposed_idx[0]
        else:
            # everything failed, give up
            return None

    def find_closest_embedding(self, word: str) -> list:
        word_idx_embed = self.word2embed_idx(word)
        if word_idx_embed is None:
            return []
        else:
            w_embed_ = self.embeddings_normalized[word_idx_embed]
            dists_, e_neigh_idx = self.embed_knn.kneighbors([w_embed_])
            e_neigh_idx = e_neigh_idx.ravel()
            e_neigh = np.asarray(self.words)[e_neigh_idx]
            neigh = [el for el in e_neigh if el in self.corpus]
            return neigh
    
    def _make_proposals(self, word: str) -> list:
        """Make proposals for an OOV word."""
        score_, spelling_neighs_ = self.find_closest_spelling(word)
        embed_neighs_ = self.find_closest_embedding(word)
        neighs = spelling_neighs_ + embed_neighs_
        if word not in neighs:
            neighs.append(word)
        return neighs
    
    def get_replacement_tokens(self, sentence: List[str]):
        """Obtain a sequence of tokens with OOV tokens properly replaced with their
        closest neighbors in the corpus (wrt the language model scores).
        
        We use a greedy strategy to obtain the maximum total score for the sentence.
        """
        
        res_seq = sentence.copy()  # result sequence we modify in-place
        total_score = 0.
        
        for i, token in enumerate(sentence):
            if i > 0:
                context = []
            else:
                context = [res_seq[i-1]]
            
            if token.lower() in self.corpus:
                total_score += self.language_model.logscore(token, context=context)
            else:
                proposals = self._make_proposals(token)
                proposal_scores = np.array([
                    total_score + self.language_model.logscore(prop, context=context)
                    for prop in proposals
                ])
                # print({prop: sc for prop, sc in zip(proposals, proposal_scores)})
                best_idx = np.argmax(proposal_scores)
                res_seq[i] = proposals[best_idx]
                total_score = proposal_scores[best_idx]
        res_seq = list(map(str.lower, res_seq))
        return res_seq
