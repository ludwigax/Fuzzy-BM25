import re
import math
import numpy as np
from scipy.sparse import coo_matrix
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Optional, Dict, List, Union, Tuple, Mapping, Any
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import string
from collections import defaultdict

from .fuzzy_map import FuzzyMap


_tokenizer = TreebankWordTokenizer()
_stop_words = set(stopwords.words('english'))
_punctuation = set(string.punctuation)


def _clean_token(token: str) -> str:
    if not token or token in _punctuation or not any(c.isalpha() for c in token):
        return None
    if re.fullmatch(r"\d+", token):
        return None
    if token.lower() in _stop_words:
        return None
    return token.lower()


def _treebank_tokenize(text: str) -> List[str]:
    tokens = _tokenizer.tokenize(text)
    return [token for token in map(_clean_token, tokens) if token is not None]


@dataclass
class BMArray:
    """
    Term Frequency: Dict[term, List[frequency]] # term frequencies of each document
    Inverse Document Frequency: Dict[term, coefficient] # idf of each term
    Term Document Count: Dict[term, frequency] # number of documents containing the term
    """
    tf: Optional[coo_matrix]
    idf: Optional[float]
    tdc: Optional[int]

class BM25:
    def __init__(self, corpus: List[Union[List, str]], tokenize_func: Optional[callable]=None):
        self.doc_nums = len(corpus)
        self.doc_lens = []
        self.avg_doc_len = 0
        self.avg_idf = 0
        self.tarray: FuzzyMap[BMArray] = FuzzyMap()
        self.tokenize_func = tokenize_func or _treebank_tokenize

        if len(corpus) < 1:
            raise ValueError("BM25 cannot be instantiated with an empty corpus")
        if isinstance(corpus[0], str):
            corpus = self.tokenize_corpus(corpus)

        self.initialize(corpus)

    def initialize(self, corpus):
        total_words = 0
        term_doc_freq = defaultdict(lambda: defaultdict(int))
        
        for i, doc in tqdm(enumerate(corpus), desc="Initializing BM25", total=len(corpus)):
            total_words += len(doc)
            self.doc_lens.append(len(doc))
            for w in doc:
                term_doc_freq[w][i] += 1
        
        for w, doc_freqs in tqdm(term_doc_freq.items(), desc="Building COO matrices", total=len(term_doc_freq)):
            doc_indices = list(doc_freqs.keys())
            frequencies = list(doc_freqs.values())

            row_indices = [0] * len(doc_indices)
            tf_matrix = coo_matrix(
                (frequencies, (row_indices, doc_indices)), 
                shape=(1, self.doc_nums), 
                dtype=int
            )
            tdc = len(doc_indices)
            
            self.tarray[w] = BMArray(
                tf=tf_matrix,
                idf=None,
                tdc=tdc,
            )
        
        self.initialize_idf()
        self.avg_doc_len = total_words / self.doc_nums
        
        self.tarray.add_index('all')
    
    def initialize_idf(self):
        raise NotImplementedError()
    
    def tokenize_corpus(self, corpus):
        """Tokenize corpus with optional multiprocessing support"""
        try:
            with Pool(cpu_count()) as pool:
                tokenized_corpus = pool.map(self.tokenize_func, corpus)
            return tokenized_corpus
        except (OSError, RuntimeError, AttributeError):
            return [self.tokenize_func(doc) for doc in tqdm(corpus, desc="Tokenizing corpus")]

    def _score_func(self, q_freq: np.ndarray, idf: float, doc_lens: np.ndarray) -> np.ndarray:
        """Calculate scores for a single term across all documents
        
        Args:
            q_freq: Term frequency array for the query term
            idf: Inverse document frequency for the term
            doc_lens: Document lengths array
            
        Returns:
            Score array for all documents
        """
        raise NotImplementedError()
    
    def get_scores(self, query: List[str]):
        """Get BM25 scores for a query with exact matching"""
        query = list(filter(None, [_clean_token(q) for q in query]))
        scores = np.zeros(self.doc_nums)
        doc_lens = np.array(self.doc_lens)
        
        for q in query:
            if q not in self.tarray:
                continue
            bmarray = self.tarray[q]
            q_freq = bmarray.tf.toarray()[0]
            idf = bmarray.idf or 0
            term_scores = self._score_func(q_freq, idf, doc_lens)
            scores += term_scores
        return scores
    
    def fuzzy_get_scores(self, query: List[str]):
        """Get BM25 scores for a query with fuzzy matching support"""
        query = list(filter(None, [_clean_token(q) for q in query]))
        scores = np.zeros(self.doc_nums)
        doc_lens = np.array(self.doc_lens)
        
        for q in query:
            fuzzy_results = self.tarray.fuzzy_search(q, return_scores=True)
            if not fuzzy_results:
                continue
            top_matches = fuzzy_results[:3]
            total_fuzzy_weight = sum(result.score for result in top_matches)
            if total_fuzzy_weight == 0:
                continue
            for result in top_matches:
                bmarray = result.value
                if bmarray is None:
                    continue
                q_freq = bmarray.tf.toarray()[0]
                idf = bmarray.idf or 0

                fuzzy_weight = result.score / total_fuzzy_weight
                term_scores = self._score_func(q_freq, idf, doc_lens)
                scores += fuzzy_weight * term_scores
        return scores

    def get_batch_scores(self, query_list: List[List]):
        """Get batch scores with exact matching"""
        if isinstance(query_list[0], str):
            raise ValueError("get_batch_scores requires a list of queries, not a single query")

        scores = np.zeros((len(query_list), self.doc_nums))
        for i, query in enumerate(query_list):
            scores[i] = self.get_scores(query)
        return scores

    def fuzzy_get_batch_scores(self, query_list: List[List]):
        """Get batch scores with fuzzy matching"""
        if isinstance(query_list[0], str):
            raise ValueError("fuzzy_get_batch_scores requires a list of queries, not a single query")

        scores = np.zeros((len(query_list), self.doc_nums))
        for i, query in enumerate(query_list):
            scores[i] = self.fuzzy_get_scores(query)
        return scores

    def get_topk(self, query, n=5, fuzzy=False) -> Tuple[List, List]:
        """Get top-k results with exact matching"""
        if fuzzy:
            scores = self.fuzzy_get_scores(query)
        else:
            scores = self.get_scores(query)
        top_k = np.argsort(scores)[::-1][:n]
        return list(zip(top_k, scores[top_k]))


class BM25Common(BM25): 
    """BM25Common (BM25Okapi) is a variant of BM25 that uses a different term frequency normalization"""
    def __init__(
        self,
        corpus: List[Union[List, str]], 
        tokenize_func: Optional[callable]=None, 
        k1=1.5, 
        b=0.75, 
        epsilon=0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenize_func)

    def initialize_idf(self):
        idf_sum = 0
        negative_idfs = []
        for w, bmarray in tqdm(self.tarray.items(), desc="Calculating IDF", total=len(self.tarray)):
            idf = self.calc_idf(w, init=True)
            bmarray.idf = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(w)
        self.avg_idf = idf_sum / len(self.tarray)
        eps = self.epsilon * self.avg_idf
        for w in negative_idfs:
            self.tarray[w].idf = eps

    def calc_idf(self, w: Union[str, List[str]]=None, init=False):
        if isinstance(w, list):
            freq = np.count_nonzero(
                np.sum([self.tarray[_w].tf.toarray()[0] for _w in w if _w in self.tarray], axis=0) > 0
            )
        else:
            freq = self.tarray[w].tdc
        idf = math.log(self.doc_nums - freq + 0.5) - math.log(freq + 0.5)
        if idf < 0 and not init:
            idf = self.epsilon * self.avg_idf
        return idf

    def _score_func(self, q_freq: np.ndarray, idf: float, doc_lens: np.ndarray) -> np.ndarray:
        """BM25 (Okapi) scoring function"""
        return idf * q_freq * (self.k1 + 1) / \
               (q_freq + self.k1 * (1 - self.b + self.b * doc_lens / self.avg_doc_len))


class BM25L(BM25):
    """BM25L is a variant of BM25 that uses log-normalization for term frequency"""
    def __init__(
        self,
        corpus: List[Union[List, str]], 
        tokenize_func: Optional[callable]=None, 
        k1=1.5, 
        b=0.75, 
        delta=1,
    ):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenize_func)

    def initialize_idf(self):
        for w, bmarray in tqdm(self.tarray.items(), desc="Calculating IDF", total=len(self.tarray)):
            idf = math.log(self.doc_nums + 1) - math.log(bmarray.tdc + 0.5)
            bmarray.idf = idf

    def _score_func(self, q_freq: np.ndarray, idf: float, doc_lens: np.ndarray) -> np.ndarray:
        """BM25L scoring function"""
        ctd = q_freq / (1 - self.b + self.b * doc_lens / self.avg_doc_len)
        return idf * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
               (self.k1 + ctd + self.delta)

class BM25Plus(BM25):
    """BM25+ is a variant of BM25 that uses a different term frequency normalization"""
    def __init__(
        self,
        corpus: List[Union[List, str]], 
        word_tokenize: Optional[callable]=None, 
        k1=1.5, 
        b=0.75, 
        delta=1,
    ):
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, word_tokenize)

    def initialize_idf(self):
        for w, bmarray in tqdm(self.tarray.items(), desc="Calculating IDF", total=len(self.tarray)):
            idf = math.log(self.doc_nums + 1) - math.log(bmarray.tdc)
            bmarray.idf = idf

    def _score_func(self, q_freq: np.ndarray, idf: float, doc_lens: np.ndarray) -> np.ndarray:
        """BM25+ scoring function"""
        return idf * (self.delta + (q_freq * (self.k1 + 1)) / \
               (self.k1 * (1 - self.b + self.b * doc_lens / self.avg_doc_len) + q_freq))
