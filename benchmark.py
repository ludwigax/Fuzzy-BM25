#!/usr/bin/env python3
"""
Benchmark script for fuzzy_bm25 performance evaluation
Comprehensive evaluation of exact and fuzzy matching capabilities
"""

import time
import random
import string
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import json
from pathlib import Path

from fuzzy_bm25 import BM25Common, FuzzyMap

# Generate synthetic document corpus
def generate_web_documents(n_docs: int = 10000) -> List[str]:
    """Generate synthetic web documents for performance testing"""
    topics = [
        "machine learning artificial intelligence neural networks deep learning",
        "web search information retrieval search engines indexing ranking",
        "natural language processing text mining sentiment analysis",
        "computer vision image recognition object detection classification",
        "database systems sql nosql data warehousing big data analytics",
        "software engineering programming languages frameworks development",
        "cybersecurity encryption authentication network security protocols",
        "cloud computing distributed systems microservices containerization",
        "mobile applications android ios app development user interface",
        "e-commerce online shopping digital marketing social media platforms"
    ]
    
    docs = []
    for i in range(n_docs):
        # Select random topics and mix them
        selected_topics = random.sample(topics, random.randint(1, 3))
        
        # Generate document with some noise
        doc_words = []
        for topic in selected_topics:
            words = topic.split()
            # Add some random words and typos
            for word in words:
                if random.random() < 0.9:  # 90% chance to include word
                    if random.random() < 0.05:  # 5% chance of typo
                        word = introduce_typo(word)
                    doc_words.append(word)
                    
        # Add some random common words
        common_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
        doc_words.extend(random.sample(common_words, random.randint(2, 5)))
        
        # Shuffle and join
        random.shuffle(doc_words)
        docs.append(" ".join(doc_words))
    
    return docs

def introduce_typo(word: str) -> str:
    """Introduce random typos to simulate real-world queries"""
    if len(word) < 3:
        return word
    
    typo_type = random.choice(["substitute", "delete", "insert", "transpose"])
    
    if typo_type == "substitute":
        pos = random.randint(0, len(word) - 1)
        new_char = random.choice(string.ascii_lowercase)
        return word[:pos] + new_char + word[pos+1:]
    elif typo_type == "delete":
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos+1:]
    elif typo_type == "insert":
        pos = random.randint(0, len(word))
        new_char = random.choice(string.ascii_lowercase)
        return word[:pos] + new_char + word[pos:]
    elif typo_type == "transpose" and len(word) > 1:
        pos = random.randint(0, len(word) - 2)
        return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
    
    return word

def generate_queries_with_typos(n_queries: int = 100) -> List[Tuple[List[str], List[str]]]:
    """Generate query pairs: (clean_query, typo_query)"""
    base_queries = [
        ["machine", "learning"],
        ["deep", "neural", "networks"],
        ["search", "engine", "optimization"],
        ["natural", "language", "processing"],
        ["computer", "vision", "recognition"],
        ["database", "management", "systems"],
        ["software", "development", "programming"],
        ["network", "security", "encryption"],
        ["cloud", "computing", "services"],
        ["mobile", "application", "development"],
        ["web", "development", "frameworks"],
        ["data", "science", "analytics"],
        ["artificial", "intelligence", "algorithms"],
        ["information", "retrieval", "systems"],
        ["user", "interface", "design"]
    ]
    
    queries = []
    for _ in range(n_queries):
        base_query = random.choice(base_queries)
        # Create typo version
        typo_query = [introduce_typo(word) if random.random() < 0.3 else word for word in base_query]
        queries.append((base_query, typo_query))
    
    return queries

def benchmark_bm25_performance():
    """Comprehensive benchmark of BM25 with and without fuzzy matching"""
    print("Performance Evaluation: fuzzy_bm25")
    print("=" * 50)
    
    # Generate test data
    print("Generating synthetic document corpus...")
    corpus_sizes = [1000, 5000, 10000, 20000]
    query_counts = [50, 100, 200, 500]
    
    results = {
        "corpus_sizes": corpus_sizes,
        "exact_times": [],
        "fuzzy_times": [],
        "exact_recall": [],
        "fuzzy_recall": [],
        "query_performance": {
            "query_counts": query_counts,
            "exact_qps": [],
            "fuzzy_qps": []
        }
    }
    
    # Test different corpus sizes
    for corpus_size in corpus_sizes:
        print(f"\nTesting corpus size: {corpus_size} documents")
        
        # Generate corpus and queries
        docs = generate_web_documents(corpus_size)
        queries = generate_queries_with_typos(100)
        
        # Initialize BM25
        print("  Building BM25 index...")
        bm25 = BM25Common(docs)
        
        # Test exact matching
        print("  Evaluating exact matching...")
        exact_start = time.time()
        exact_results = []
        for clean_query, _ in queries:
            scores = bm25.get_scores(clean_query)
            top_docs = np.argsort(scores)[::-1][:10]
            exact_results.append(len([i for i in top_docs if scores[i] > 0]))
        exact_time = time.time() - exact_start
        
        # Test fuzzy matching
        print("  Evaluating fuzzy matching...")
        fuzzy_start = time.time()
        fuzzy_results = []
        for _, typo_query in queries:
            scores = bm25.fuzzy_get_scores(typo_query)
            top_docs = np.argsort(scores)[::-1][:10]
            fuzzy_results.append(len([i for i in top_docs if scores[i] > 0]))
        fuzzy_time = time.time() - fuzzy_start
        
        # Record results
        results["exact_times"].append(exact_time)
        results["fuzzy_times"].append(fuzzy_time)
        results["exact_recall"].append(np.mean(exact_results))
        results["fuzzy_recall"].append(np.mean(fuzzy_results))
        
        print(f"    Exact matching: {exact_time:.3f}s, Recall@10: {np.mean(exact_results):.2f}")
        print(f"    Fuzzy matching: {fuzzy_time:.3f}s, Recall@10: {np.mean(fuzzy_results):.2f}")
    
    # Test query throughput
    print(f"\nEvaluating query throughput (10k documents)...")
    docs = generate_web_documents(10000)
    bm25 = BM25Common(docs)
    
    for query_count in query_counts:
        queries = generate_queries_with_typos(query_count)
        
        # Exact matching QPS
        start_time = time.time()
        for clean_query, _ in queries:
            bm25.get_scores(clean_query)
        exact_qps = query_count / (time.time() - start_time)
        
        # Fuzzy matching QPS  
        start_time = time.time()
        for _, typo_query in queries:
            bm25.fuzzy_get_scores(typo_query)
        fuzzy_qps = query_count / (time.time() - start_time)
        
        results["query_performance"]["exact_qps"].append(exact_qps)
        results["query_performance"]["fuzzy_qps"].append(fuzzy_qps)
        
        print(f"    {query_count} queries: Exact {exact_qps:.1f} QPS, Fuzzy {fuzzy_qps:.1f} QPS")
    
    return results

def create_performance_plots(results):
    """Create performance visualization plots"""
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('fuzzy_bm25 Performance Benchmark', fontsize=16, fontweight='bold')
    
    # Plot 1: Processing Time vs Corpus Size
    ax1.plot(results["corpus_sizes"], results["exact_times"], 'o-', label='Exact Matching', linewidth=2, markersize=8)
    ax1.plot(results["corpus_sizes"], results["fuzzy_times"], 's-', label='Fuzzy Matching', linewidth=2, markersize=8)
    ax1.set_xlabel('Corpus Size (documents)')
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Processing Time vs Corpus Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recall Performance
    ax2.bar(np.array(results["corpus_sizes"]) - 400, results["exact_recall"], 800, 
            label='Exact (Clean Queries)', alpha=0.7, color='skyblue')
    ax2.bar(np.array(results["corpus_sizes"]) + 400, results["fuzzy_recall"], 800, 
            label='Fuzzy (Typo Queries)', alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Corpus Size (documents)')
    ax2.set_ylabel('Average Recall@10')
    ax2.set_title('Recall Performance: Exact vs Fuzzy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Query Throughput
    ax3.plot(results["query_performance"]["query_counts"], 
             results["query_performance"]["exact_qps"], 'o-', 
             label='Exact QPS', linewidth=2, markersize=8)
    ax3.plot(results["query_performance"]["query_counts"], 
             results["query_performance"]["fuzzy_qps"], 's-', 
             label='Fuzzy QPS', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Queries')
    ax3.set_ylabel('Queries Per Second (QPS)')
    ax3.set_title('Query Throughput Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Fuzzy Advantage
    fuzzy_advantage = [f/e for f, e in zip(results["fuzzy_recall"], results["exact_recall"])]
    ax4.bar(results["corpus_sizes"], fuzzy_advantage, color='gold', alpha=0.8)
    ax4.axhline(y=1.0, color='red', linestyle='--', label='Baseline (1.0x)')
    ax4.set_xlabel('Corpus Size (documents)')
    ax4.set_ylabel('Fuzzy Recall / Exact Recall')
    ax4.set_title('Fuzzy Matching Advantage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("Performance plot saved as 'benchmark_results.png'")
    
    # Save detailed results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved as 'benchmark_results.json'")
    
    return fig

def main():
    """Run the complete benchmark suite"""
    print("fuzzy_bm25 Performance Benchmark Suite")
    print("=" * 50)
    
    # Run benchmarks
    results = benchmark_bm25_performance()
    
    # Create plots
    print(f"\nGenerating performance visualizations...")
    create_performance_plots(results)
    
    # Summary
    print(f"\nBENCHMARK SUMMARY")
    print("=" * 30)
    max_corpus = max(results["corpus_sizes"])
    max_idx = results["corpus_sizes"].index(max_corpus)
    
    print(f"Maximum tested corpus size: {max_corpus:,} documents")
    print(f"Exact matching: {results['exact_times'][max_idx]:.2f}s ({100/results['exact_times'][max_idx]:.1f} QPS)")
    print(f"Fuzzy matching: {results['fuzzy_times'][max_idx]:.2f}s ({100/results['fuzzy_times'][max_idx]:.1f} QPS)")
    print(f"Fuzzy recall improvement: {results['fuzzy_recall'][max_idx]/results['exact_recall'][max_idx]:.2f}x")
    print(f"Peak throughput: {max(results['query_performance']['fuzzy_qps']):.1f} fuzzy QPS")
    
    print(f"\nBenchmark completed. Results saved as 'benchmark_results.png'")

if __name__ == "__main__":
    main() 