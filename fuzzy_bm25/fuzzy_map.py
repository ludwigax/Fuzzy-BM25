import math
import weakref
import random
from collections import defaultdict, deque
from typing import Dict, Iterable, Set, List, Tuple, Optional, Callable, Any, Union, Generic, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod
import bisect

V = TypeVar('V')


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


@dataclass
class SearchResult(Generic[V]):
    """Search result containing key, value and similarity score"""
    key: str
    value: V
    score: float = 0.0


class FuzzyIndex(ABC):
    """Abstract base class for fuzzy indices"""
    
    def __init__(self, store: Dict[str, Any]):
        self.store = store
    
    @abstractmethod
    def add(self, key: str) -> None:
        """Add a key to the index"""
        pass
    
    @abstractmethod
    def remove(self, key: str) -> None:
        """Remove a key from the index"""
        pass
    
    @abstractmethod
    def find(self, query: str, **kwargs) -> Iterable[str]:
        """Find similar keys in the index"""
        pass


class BKTreeIndex(FuzzyIndex):
    """BK-tree index for Levenshtein distance search"""
    
    def __init__(self, store: Dict[str, Any], distance_fn: Optional[Callable] = None, balanced: bool = True):
        super().__init__(store)
        self.distance_fn = distance_fn or _levenshtein_distance
        self.nodes: Dict[str, Dict[int, str]] = {}
        self.root: Optional[str] = None
        self.balanced = balanced
        self._pending_keys = []
    
    def add(self, key: str) -> None:
        """Add a key to the BK-tree"""
        if self.balanced:
            self._pending_keys.append(key)
        else:
            self._add_single(key)
    
    def _add_single(self, key: str) -> None:
        """Add a single key to the BK-tree"""
        if self.root is None:
            self.root = key
            self.nodes[key] = {}
            return
        
        current = self.root
        while True:
            distance = self.distance_fn(key, current)
            children = self.nodes[current]
            
            if distance in children:
                current = children[distance]
            else:
                children[distance] = key
                self.nodes[key] = {}
                break

    def _build_balanced_tree(self):
        """Build a balanced BK-tree from pending keys"""
        if not self._pending_keys:
            return
        
        keys = self._pending_keys.copy()
        random.shuffle(keys)
        
        keys.sort(key=len)
        root_idx = len(keys) // 2
        root_key = keys[root_idx]
        
        self.nodes.clear()
        self.root = None
        self._pending_keys.clear()
        
        self._add_single(root_key)
        
        for key in keys:
            if key != root_key:
                self._add_single(key)
    
    def remove(self, key: str) -> None:
        """Remove a key from the BK-tree (simplified implementation)"""
        if key in self.nodes:
            del self.nodes[key]
    
    def find(self, query: str, max_distance: int = 2) -> Iterable[str]:
        """Find keys within max_distance of query"""
        if self.balanced and self._pending_keys:
            self._build_balanced_tree()
            
        if self.root is None:
            return
        
        stack = [self.root]
        while stack:
            current = stack.pop()
            if current not in self.nodes:
                continue
            
            distance = self.distance_fn(query, current)
            if distance <= max_distance:
                yield current
            
            lower_bound = distance - max_distance
            upper_bound = distance + max_distance
            
            for child_distance, child_key in self.nodes[current].items():
                if lower_bound <= child_distance <= upper_bound:
                    stack.append(child_key)


class TrieIndex(FuzzyIndex):
    """Trie index for prefix matching and editable distance search"""
    
    class TrieNode:
        def __init__(self):
            self.children: Dict[str, 'TrieIndex.TrieNode'] = {}
            self.is_end: bool = False
            self.key: Optional[str] = None
    
    def __init__(self, store: Dict[str, Any]):
        super().__init__(store)
        self.root = self.TrieNode()
    
    def add(self, key: str) -> None:
        """Add a key to the trie"""
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.is_end = True
        node.key = key
    
    def remove(self, key: str) -> None:
        """Remove a key from the trie"""
        def _remove_helper(node: 'TrieIndex.TrieNode', key: str, depth: int) -> bool:
            if depth == len(key):
                if not node.is_end:
                    return False
                node.is_end = False
                node.key = None
                return len(node.children) == 0
            
            char = key[depth]
            if char not in node.children:
                return False
            
            should_delete_child = _remove_helper(node.children[char], key, depth + 1)
            
            if should_delete_child:
                del node.children[char]
            
            return not node.is_end and len(node.children) == 0
        
        _remove_helper(self.root, key, 0)
    
    def find(self, query: str, max_edits: int = 2, prefix_match: bool = False) -> Iterable[str]:
        """Find keys with fuzzy matching"""
        if prefix_match:
            yield from self._prefix_search(query)
        else:
            yield from self._fuzzy_search(query, max_edits)
    
    def _prefix_search(self, prefix: str) -> Iterable[str]:
        """Find all keys with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return
            node = node.children[char]
        
        def _collect_keys(node: 'TrieIndex.TrieNode'):
            if node.is_end and node.key:
                yield node.key
            for child in node.children.values():
                yield from _collect_keys(child)
        
        yield from _collect_keys(node)
    
    def _fuzzy_search(self, query: str, max_edits: int) -> Iterable[str]:
        r"""
        Ludwig's note:
        I found that there may be risks here. For example, when the prefix matches,
        if there is a short word but the longer word is not in the index,
        the search might return the shorter word even though the subsequent 
        deletion edit distance is much greater than max_edits.
        """
        """Find keys within edit distance"""
        def _search_recursive(node: 'TrieIndex.TrieNode', query_idx: int, edits_left: int):
            if node.is_end and node.key:
                yield node.key
            
            if edits_left == 0:
                remaining = query[query_idx:]
                current = node
                for char in remaining:
                    if char not in current.children:
                        return
                    current = current.children[char]
                if current.is_end and current.key:
                    yield current.key
                return
            
            if query_idx < len(query):
                char = query[query_idx]
                if char in node.children:
                    yield from _search_recursive(node.children[char], query_idx + 1, edits_left)
            
            for child_char, child_node in node.children.items():
                # Substitution
                if query_idx < len(query):
                    yield from _search_recursive(child_node, query_idx + 1, edits_left - 1)
                # Insertion
                yield from _search_recursive(child_node, query_idx, edits_left - 1)
            
            # Deletion
            if query_idx < len(query):
                yield from _search_recursive(node, query_idx + 1, edits_left - 1)
        
        seen = set()
        for key in _search_recursive(self.root, 0, max_edits):
            if key not in seen:
                seen.add(key)
                yield key


class NGramIndex(FuzzyIndex):
    """N-gram inverted index for Jaccard similarity search"""
    
    def __init__(self, store: Dict[str, Any], n: int = 3):
        super().__init__(store)
        self.n = n
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.key_ngrams: Dict[str, Set[str]] = {}
    
    def _get_ngrams(self, text: str) -> Set[str]:
        """Extract n-grams from text"""
        if len(text) < self.n:
            return {text}
        
        ngrams = set()
        for i in range(len(text) - self.n + 1):
            ngrams.add(text[i:i + self.n])
        return ngrams
    
    def add(self, key: str) -> None:
        """Add a key to the n-gram index"""
        ngrams = self._get_ngrams(key.lower())
        self.key_ngrams[key] = ngrams
        
        for ngram in ngrams:
            self.inverted_index[ngram].add(key)
    
    def remove(self, key: str) -> None:
        """Remove a key from the n-gram index"""
        if key in self.key_ngrams:
            ngrams = self.key_ngrams[key]
            for ngram in ngrams:
                self.inverted_index[ngram].discard(key)
                if not self.inverted_index[ngram]:
                    del self.inverted_index[ngram]
            del self.key_ngrams[key]
    
    def find(self, query: str, min_similarity: float = 0.3, top_k: int = 10) -> Iterable[str]:
        """Find keys with Jaccard similarity above threshold"""
        query_ngrams = self._get_ngrams(query.lower())
        if not query_ngrams:
            return
        
        candidates: Dict[str, float] = defaultdict(float)
        
        for ngram in query_ngrams:
            if ngram in self.inverted_index:
                for key in self.inverted_index[ngram]:
                    candidates[key] += 1
        
        similarities = []
        for key, intersection_count in candidates.items():
            if key in self.key_ngrams:
                key_ngrams = self.key_ngrams[key]
                union_size = len(query_ngrams) + len(key_ngrams) - intersection_count
                jaccard_sim = intersection_count / union_size if union_size > 0 else 0
                
                if jaccard_sim >= min_similarity:
                    similarities.append((jaccard_sim, key))
        
        similarities.sort(reverse=True)
        for _, key in similarities[:top_k]:
            yield key


class FuzzyMap(Generic[V]):
    """A dictionary-like data structure with fuzzy search capabilities
    """
    
    # Predefined index configurations
    INDEX_PRESETS = {
        'similarity': ['ngram'],
        'edit_distance': ['bk', 'trie'],
        'all': ['bk', 'trie', 'ngram']
    }
    
    def __init__(self) -> None:
        self._store: Dict[str, V] = {}
        self._indices: Dict[str, FuzzyIndex] = {}
        self._auto_method_cache: Dict[str, str] = {}
    
    def __setitem__(self, key: str, value: V) -> None:
        """Set an item in the map"""
        is_new_key = key not in self._store
        self._store[key] = value
        
        if is_new_key:
            for index in self._indices.values():
                index.add(key)
    
    def __getitem__(self, key: str) -> V:
        """Get an item from the map"""
        return self._store[key]
    
    def __delitem__(self, key: str) -> None:
        """Delete an item from the map"""
        if key in self._store:
            del self._store[key]
            for index in self._indices.values():
                index.remove(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in the map"""
        return key in self._store
    
    def __len__(self) -> int:
        """Return the number of items in the map"""
        return len(self._store)
    
    def keys(self):
        """Return keys iterator"""
        return self._store.keys()
    
    def values(self):
        """Return values iterator"""
        return self._store.values()
    
    def items(self):
        """Return items iterator"""
        return self._store.items()
    
    def get(self, key: str, default: Optional[V] = None) -> Optional[V]:
        """Get an item with default value"""
        return self._store.get(key, default)
    
    def add_index(self, indices: Union[str, List[str], Dict[str, Dict]], **global_kwargs) -> 'FuzzyMap':
        """Add fuzzy indices to the map (chainable)
        
        Args:
            indices: Can be:
                - str: Single index type ('bk', 'trie', 'ngram') or preset ('all', 'fast', etc.)
                - List[str]: Multiple index types ['bk', 'trie']
                - Dict[str, Dict]: Custom configuration {'my_bk': {'type': 'bk', 'balanced': True}}
            **global_kwargs: Global arguments applied to all indices
            
        Returns:
            Self for method chaining
        """
        if isinstance(indices, str) and indices in self.INDEX_PRESETS:
            indices = self.INDEX_PRESETS[indices]
        
        if isinstance(indices, str):
            indices = [indices]
        
        if isinstance(indices, list):
            for idx_type in indices:
                self._add_single_index(idx_type, idx_type, **global_kwargs)
        
        elif isinstance(indices, dict):
            for name, config in indices.items():
                idx_type = config.pop('type', name)
                merged_kwargs = {**global_kwargs, **config}
                self._add_single_index(name, idx_type, **merged_kwargs)
        
        return self
    
    def _add_single_index(self, name: str, index_type: str, **kwargs) -> None:
        """Add a single index to the map"""
        if index_type == 'bk':
            index = BKTreeIndex(self._store, **kwargs)
        elif index_type == 'trie':
            index = TrieIndex(self._store, **kwargs)
        elif index_type == 'ngram':
            index = NGramIndex(self._store, **kwargs)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        for key in self._store:
            index.add(key)
        self._indices[name] = index
    
    def remove_index(self, name: str) -> None:
        """Remove an index from the map"""
        if name in self._indices:
            del self._indices[name]
    
    def _auto_select_method(self, query: str) -> str:
        """Automatically select the best search method based on query characteristics"""
        if query in self._auto_method_cache:
            return self._auto_method_cache[query]
        
        available_methods = list(self._indices.keys())
        if not available_methods:
            raise ValueError("No indices available. Please add indices first.")
        
        query_len = len(query)
        
        if query_len <= 3 and 'ngram' in available_methods:
            method = 'ngram'
        elif query_len <= 8 and 'bk' in available_methods:
            method = 'bk'
        elif 'trie' in available_methods:
            method = 'trie'
        else:
            method = available_methods[0]
        
        self._auto_method_cache[query] = method
        return method
    
    def fuzzy_search(self, query: str, method: Optional[str] = None, 
                    return_scores: bool = False, auto: bool = True, **kwargs) -> Union[List[str], List[SearchResult[V]]]:
        """Perform fuzzy search using specified or auto-selected method
        
        Args:
            query: Query string
            method: Search method ('bk', 'trie', 'ngram') or None for auto-selection
            return_scores: Whether to return similarity scores
            auto: Whether to enable auto method selection when method is None
            **kwargs: Additional arguments for the search method
            
        Returns:
            List of matching keys or SearchResult objects
        """
        if method is None and auto:
            method = self._auto_select_method(query)
        elif method is None:
            method = list(self._indices.keys())[0] if self._indices else None
        
        if method not in self._indices:
            raise ValueError(f"Index '{method}' not found. Available indices: {list(self._indices.keys())}")
        
        index = self._indices[method]
        matching_keys = list(index.find(query, **kwargs))
        
        if not return_scores:
            return matching_keys
        
        results = []
        for key in matching_keys:
            if max(len(query), len(key)) == 0:
                score = 1.0
            else:
                max_len = max(len(query), len(key))
                if method == 'bk':
                    distance = index.distance_fn(query, key)
                    score = 1.0 - (distance / max_len)
                elif method == 'ngram':
                    query_ngrams = index._get_ngrams(query.lower())
                    key_ngrams = index.key_ngrams.get(key, set())
                    intersection = len(query_ngrams & key_ngrams)
                    union = len(query_ngrams | key_ngrams)
                    score = intersection / union if union > 0 else 0.0
                else:
                    distance = _levenshtein_distance(query, key)
                    score = 1.0 - (distance / max_len)
            
            results.append(SearchResult(key=key, value=self._store[key], score=score))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def fuzzy_get(self, query: str, method: Optional[str] = None, default: Optional[V] = None, **kwargs) -> Optional[V]:
        """Get the best fuzzy match for a query
        
        Args:
            query: Query string
            method: Search method to use (None for auto-selection)
            default: Default value if no match found
            **kwargs: Additional arguments for the search method
            
        Returns:
            Value of the best matching key or default
        """
        results = self.fuzzy_search(query, method, return_scores=True, **kwargs)
        if results:
            return results[0].value
        return default
    
    def get_index_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all indices"""
        info = {}
        for name, index in self._indices.items():
            info[name] = {
                'type': type(index).__name__,
                'size': len(self._store)
            }
            if isinstance(index, NGramIndex):
                info[name]['n'] = index.n
                info[name]['ngram_count'] = len(index.inverted_index)
        return info