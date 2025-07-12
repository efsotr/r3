import re
import itertools
import logging
import random
import os
import time
import argparse
import json
from collections import defaultdict, Counter
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
import faiss

from datasets import load_dataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.preprocessing as preprocessing

from openai import OpenAI
import openai

OPENAI_RETRIES = 3
RANDOM_SEED = 42

OPENAI_CLIENT = OpenAI()

DICT_SIZE = {
    "tulu": 3500,
    "feedbackcollection": 2500,
    "ultrafeedback": 2000,
    "skyworks": 2000,
    "math-step-pairwise": 3000,
    "ace-code-pairwise": 3000,
    "glue": 400,
    "super-glue": 400,
    "summeval": 1000,
    "evouna": 200,
}

SOURCE_NAME = {
    "tulu": "rubricreward/llm-metric-tulu-new",
    "feedbackcollection": "rubricreward/llm-metric-feedbackcollection-new",
    "ultrafeedback": "rubricreward/llm-metric-ultrafeedback-new",
    "skyworks": "rubricreward/llm-metric-skyworks-new",
    "math-step-pairwise": "rubricreward/llm-metric-math-step-pairwise-new",
    "ace-code-pairwise": "rubricreward/llm-metric-ace-code-pairwise-new",
    "glue": "rubricreward/llm-metric-glue",
    "super-glue": "rubricreward/llm-metric-super-glue",
    "summeval": "rubricreward/llm-metric-summeval-new",
    "evouna": "rubricreward/llm-metric-evouna",
}

K_CANDIDATES = [3, 4, 5, 7, 10]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def hierarchical_cluster_fit_threshold(
    embeddings,
    thresholds,
    min_size_per_cluster=3,
    linkage="average"  # "ward" works poorly with cosine distance
):
    best_score = -1
    best_labels = None
    best_threshold = None

    for threshold in tqdm(thresholds):
        model = AgglomerativeClustering(
            n_clusters=None,
            linkage=linkage,
            metric="cosine",  # critical for normalized embeddings!
            distance_threshold=threshold
        )
        labels = model.fit_predict(embeddings)

        # Skip if any cluster is too small or only one cluster exists
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or np.any(counts < min_size_per_cluster):
            logging.info(f"Skipping threshold {threshold} due to cluster being too small")
            continue

        score = silhouette_score(embeddings, labels, metric="cosine")
        logging.info(f"Threshold {threshold} has score of {score}")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_threshold = threshold

    # Fallback: If no threshold works, use a fixed k
    if best_threshold is None:
        feasible_k = max(2, len(embeddings) // min_size_per_cluster)
        model = AgglomerativeClustering(n_clusters=feasible_k, linkage=linkage, metric="cosine")
        best_labels = model.fit_predict(embeddings)
        best_threshold = "Fallback (k={})".format(feasible_k)

    return best_labels, best_threshold

def cluster_fit_k(embeddings, k_candidates, sample_size=20000):
    faiss.normalize_L2(embeddings)  # Critical for cosine similarity

    # Test k candidates
    scores = []

    for k in k_candidates:
        # FAISS K-means (GPU-accelerated)
        kmeans = faiss.Kmeans(
            d=embeddings.shape[1],
            k=k,
            niter=300,
        )
        kmeans.train(embeddings)
        logging.info(f"k={k}: Finished training")
        # Get cluster assignments and compute silhouette score
        _, labels = kmeans.index.search(embeddings, 1)
        labels = labels.flatten()
        score = silhouette_score(embeddings, labels, metric='cosine', sample_size=sample_size)
        scores.append(score)
        logging.info(f"k={k}: Silhouette Score = {score:.3f}")
    
    best_k = k_candidates[np.argmax(scores)]
    kmeans = faiss.Kmeans(
        d=embeddings.shape[1],
        k=best_k,
        niter=300,
    )
    kmeans.train(embeddings)
    return best_k, kmeans

def mmr_select(embeddings, selected, candidates, lambda_param=0.5, top_k=1):
    """Selects diverse points using Maximal Marginal Relevance (MMR)."""
    selected_set = embeddings[selected] if selected else np.zeros((1, embeddings.shape[1]))
    remaining = embeddings[candidates]
    sim_to_selected = cosine_similarity(remaining, selected_set).max(axis=1) if selected else np.zeros(len(candidates))
    sim_to_query = cosine_similarity(remaining, np.mean(selected_set, axis=0, keepdims=True)).flatten()
    
    mmr_scores = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
    mmr_indices = np.argsort(-mmr_scores)[:top_k]
    return [candidates[i] for i in mmr_indices]


def proportional_cluster_sample_with_diversity(kmeans, embeddings, total_samples, diversity_ratio=0.75):
    """Sample from clusters: 25% closest to centroid, 75% diverse using MMR."""
    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.flatten()
    unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
    logging.info(f"Unique clusters and cluster counts: {unique_clusters}; {cluster_counts}")
    n_clusters = len(unique_clusters)

    assert total_samples >= n_clusters, f"Need at least {n_clusters} samples (1/cluster)"

    proportions = cluster_counts / cluster_counts.sum()
    allocations = np.maximum(1, (proportions * (total_samples - n_clusters)).astype(int))
    allocations = allocations.cumsum()

    while allocations[-1] < total_samples:
        allocations[-1] += 1

    sampled_indices = []

    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_embs = embeddings[cluster_indices]
        centroid = kmeans.centroids[cluster]

        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        sorted_idx = np.argsort(dists)

        n_samples = allocations[i] - (0 if i == 0 else allocations[i - 1])
        n_close = max(1, int(n_samples * (1 - diversity_ratio)))
        n_diverse = n_samples - n_close

        # Local indices of the closest samples
        selected_close_local = sorted_idx[:n_close].tolist()

        # Local indices of the remaining candidates
        remaining_local = sorted_idx[n_close:].tolist()

        # MMR-based selection on local indices
        selected_diverse_local = []
        for _ in tqdm(range(n_diverse)):
            if not remaining_local:
                break
            selected = mmr_select(cluster_embs, selected_close_local + selected_diverse_local, remaining_local, lambda_param=0.7)
            selected_diverse_local.append(selected[0])
            remaining_local.remove(selected[0])

        # Map back to global indices
        final_indices = cluster_indices[selected_close_local + selected_diverse_local]
        sampled_indices.append(final_indices.tolist())

    return sampled_indices

def fast_cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def mmr_select_faiss(embeddings, selected, candidates, lambda_param=0.5, top_k=1):
    """Fast MMR selection using cosine similarity with FAISS candidates."""
    if not selected:
        selected_embs = np.mean(embeddings[candidates], axis=0, keepdims=True)
        sim_to_query = fast_cosine_similarity(embeddings[candidates], selected_embs).flatten()
        return [candidates[np.argmax(sim_to_query)]]

    selected_embs = embeddings[selected]
    candidate_embs = embeddings[candidates]

    sim_to_query = fast_cosine_similarity(candidate_embs, np.mean(selected_embs, axis=0, keepdims=True)).flatten()
    sim_to_selected = fast_cosine_similarity(candidate_embs, selected_embs).max(axis=1)

    mmr_scores = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
    top_indices = np.argsort(-mmr_scores)[:top_k]
    return [candidates[i] for i in top_indices]


def proportional_cluster_sample_with_diversity_faiss(kmeans, embeddings, total_samples, diversity_ratio=0.75, max_mmr_candidates=-1):
    """Fast cluster sampling with centroid-faiss and approximate MMR."""
    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.flatten()
    unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_clusters)

    assert total_samples >= n_clusters, f"Need at least {n_clusters} samples (1 per cluster)."

    proportions = cluster_counts / cluster_counts.sum()
    allocations = np.maximum(1, (proportions * (total_samples - n_clusters)).astype(int)).cumsum()
    while allocations[-1] < total_samples:
        allocations[-1] += 1

    sampled_indices = []

    for i, cluster in enumerate(unique_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_embs = embeddings[cluster_indices].astype(np.float32)
        centroid = kmeans.centroids[cluster].astype(np.float32).reshape(1, -1)

        # Use FAISS to get sorted indices by L2 distance to centroid
        index = faiss.IndexFlatL2(cluster_embs.shape[1])
        index.add(cluster_embs)
        _, I = index.search(centroid, len(cluster_embs))
        sorted_idx = I[0]

        n_samples = allocations[i] - (0 if i == 0 else allocations[i - 1])
        n_close = max(1, int(n_samples * (1 - diversity_ratio)))
        n_diverse = n_samples - n_close

        selected_close_local = sorted_idx[:n_close].tolist()
        remaining_local = sorted_idx[n_close:].tolist()

        # Subsample candidates to speed up MMR
        if max_mmr_candidates > 0 and len(remaining_local) > max_mmr_candidates:
            remaining_local = np.random.choice(remaining_local, size=max_mmr_candidates, replace=False).tolist()

        selected_diverse_local = []
        for _ in tqdm(range(n_diverse)):
            if not remaining_local:
                break
            selected = mmr_select_faiss(cluster_embs, selected_close_local + selected_diverse_local, remaining_local, lambda_param=0.7)
            selected_diverse_local.append(selected[0])
            remaining_local.remove(selected[0])

        final_indices = cluster_indices[selected_close_local + selected_diverse_local]
        sampled_indices.append(final_indices.tolist())

    return sampled_indices

def print_some_examples_each_group(sampled_indices, dataset):
    for i in range(len(sampled_indices)):
        logging.info(dataset[sampled_indices[i][0]])
        
def get_task_input(prompt):
    # Step 1: Find the content between '### TASK' and '### RESPONSE'
    task_match = re.search(r'### TASK\s+(.*?)\s+### RESPONSE', prompt, re.DOTALL)

    if task_match:
        content = task_match.group(1)
        final_content = re.sub(r'### INPUT\n', 'Input: ', content)
        return final_content.strip() # Remove any leading/trailing whitespace
    return None

def request_openai_completion(message):
    for attempt in range(OPENAI_RETRIES):
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4.1",
                messages=message,
                max_tokens=8192,
            )

            return response.choices[0].message.content
        except openai.OpenAIError as e:
            if "rate" in str(e).lower():
                logging.warning("Hit rate limit; retrying...")
                time.sleep(61)
            else:
                logging.exception("Error calling OpenAI API:")
                time.sleep(1)
    logging.exception(f"Could not resolve error after {OPENAI_RETRIES} attempts")
    return None

def define_clusters(prompt_array, kmeans, embeddings_array, k_cluster, num_shots):
    # Get cluster assignments and distances
    distances, cluster_assignments = kmeans.index.search(embeddings_array, 1)
    cluster_assignments = cluster_assignments.flatten()
    clusters = defaultdict(list)
    for local_idx, (cluster_id, distance) in enumerate(zip(cluster_assignments, distances)):
        clusters[cluster_id].append((prompt_array[local_idx], distance[0]))
        
    # Get representative examples (closest to centroid)
    cluster_examples = {}
    for cluster_id in clusters:
        # Sort by distance ascending (smallest first)
        sorted_items = sorted(clusters[cluster_id], key=lambda x: x[1])
        cluster_examples[cluster_id] = [get_task_input(item[0]) for item in sorted_items[:num_shots]] # Get few-shot examples

    # Build cluster examples string
    clusters_str = "\n\n".join(
        f"## Cluster {cid} Examples\n" + "\n".join(f"- {ex}" for ex in ex_list)
        for cid, ex_list in sorted(cluster_examples.items())
    )

    messages_cluster = [
        {"role": "user", "content": f"""
        Analyze the following {k_cluster} clusters of items and suggest a distinct category name for each without any explanation.
        The context behind such clustering is based on the topic.

        Rules:
        1. Use clear, specific names, ABOUT SPECIFIC DOMAIN TOPIC. AVOID GENERIC CLASSIFICATION that are not insightful.
        2. Maintain parallel structure
        3. No overlapping categories

        Output: Comma-separated names (one per cluster) without any explanations and maintain order starting from cluster 0 first.
        Example output format: Technology, Sports Apparel, Healthcare Devices
        
        ###### START OF CLUSTER OF ITEMS ######
        
        {clusters_str}
        
        ###### END OF CLUSTER OF ITEMS ######

        Your categories:"""}
    ]

    clustered_categories = request_openai_completion(messages_cluster)
    categories = [cat.strip() for cat in clustered_categories.split(",")]
    logging.info(f"Categories are: {set(categories)}")

    # Based on the K-means result, assign based on the cluster id
    final_assignment = []
    for i, cluster_id in enumerate(cluster_assignments):
        final_assignment.append(categories[cluster_id])
    
    return final_assignment

def compute_stratified_sample_sizes(category_values: List[str], total_samples: int, min_per_category: int = 10) -> Dict[str, int]:
    category_counts = Counter(category_values)
    total_size = len(category_values)

    sample_sizes_per_category = {
        cat: max(min_per_category, int((count / total_size) * total_samples))
        for cat, count in category_counts.items()
    }

    # Adjust downward if over budget
    current_total = sum(sample_sizes_per_category.values())
    if current_total > total_samples:
        for cat in sorted(sample_sizes_per_category, key=lambda s: sample_sizes_per_category[s], reverse=True):
            if current_total <= total_samples:
                break
            if sample_sizes_per_category[cat] > min_per_category:
                sample_sizes_per_category[cat] -= 1
                current_total -= 1

    # Adjust upward if under budget
    elif current_total < total_samples:
        remaining = total_samples - current_total
        for cat in sorted(sample_sizes_per_category, key=lambda s: -sample_sizes_per_category[s])[:remaining]:
            sample_sizes_per_category[cat] += 1

    assert sum(sample_sizes_per_category.values()) == total_samples
    return sample_sizes_per_category
        
def stratified_by_aspect_and_source(aspect_list, source_list, total_samples, min_per_category=10):
    unique_aspects = sorted(set(aspect_list))
    per_aspect_budget = total_samples // len(unique_aspects)

    # Build aspect-source category mapping
    category_values = [f"{aspect}-{source}" for aspect, source in zip(aspect_list, source_list)]
    category_counts = Counter(category_values)

    aspect_to_sources = defaultdict(set)
    for cat in category_counts:
        aspect, src = cat.split('-')
        aspect_to_sources[aspect].add(src)

    sample_sizes_per_category = {}
    for aspect in unique_aspects:
        subcategories = [f"{aspect}-{src}" for src in aspect_to_sources[aspect]]
        sub_counts = {cat: category_counts[cat] for cat in subcategories}
        sub_total = sum(sub_counts.values())

        # Proportional allocation within aspect
        sub_alloc = {
            cat: max(min_per_category, int((count / sub_total) * per_aspect_budget))
            for cat, count in sub_counts.items()
        }

        current_total = sum(sub_alloc.values())
        if current_total > per_aspect_budget:
            for cat in sorted(sub_alloc, key=lambda s: sub_alloc[s], reverse=True):
                if current_total <= per_aspect_budget:
                    break
                if sub_alloc[cat] > min_per_category:
                    sub_alloc[cat] -= 1
                    current_total -= 1
        elif current_total < per_aspect_budget:
            remaining = per_aspect_budget - current_total
            for cat in sorted(sub_alloc, key=lambda s: -sub_alloc[s])[:remaining]:
                sub_alloc[cat] += 1

        sample_sizes_per_category.update(sub_alloc)

    assert sum(sample_sizes_per_category.values()) == total_samples
    return sample_sizes_per_category

        
def calculate_samples_per_category(dataset, dataset_name):
    if dataset_name in ["math-step-pairwise", "ace-code-pairwise"]:
        sample_sizes_per_category = {}
        category_values = ["train"] * len(dataset)
        sample_sizes_per_category['train'] = DICT_SIZE[dataset_name]
    elif dataset_name == "evouna":
        sample_sizes_per_category = {}
        sample_sizes_per_category['train'] = DICT_SIZE[dataset_name]

        category_values = [""] * len(dataset)
        for i in range(0, len(dataset), 2):
            selected_for_train = random.choice([i, i + 1])
            if i == selected_for_train:
                selected_for_no_train = i + 1
            else:
                selected_for_no_train = i

            category_values[selected_for_train] = "train"
            category_values[selected_for_no_train] = "no_train"
    elif dataset_name in ["ultrafeedback", "summeval"]:
        aspect_list = dataset['aspect']
        source_list = dataset['source'] if dataset_name == "ultrafeedback" else dataset['article_id']
        sample_sizes_per_category = stratified_by_aspect_and_source(
            aspect_list, source_list, DICT_SIZE[dataset_name]
        )
    elif dataset_name == "skyworks":
        category_values = dataset['task_category']
        countable = [c for c in category_values if c != "safety"]
        sample_sizes_per_category = compute_stratified_sample_sizes(
            countable, DICT_SIZE[dataset_name]
        )
        sample_sizes_per_category["safety"] = DICT_SIZE[dataset_name]  # if safety has its own budget
    elif dataset_name == "feedbackcollection":
        category_values = dataset['score']
        sample_sizes_per_category = compute_stratified_sample_sizes(
            category_values, DICT_SIZE[dataset_name]
        )
    elif dataset_name in ["glue", "super-glue"]:
        category_values = dataset['task']
        sample_sizes_per_category = compute_stratified_sample_sizes(
            category_values, DICT_SIZE[dataset_name]
        )
        
    return category_values, sample_sizes_per_category

def run_sampling():
    parser = argparse.ArgumentParser(description='General workflow for sampling')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=SOURCE_NAME.keys(),
                        help=f"Dataset name.")
    parser.add_argument('--embedding_path', '-d', type=str,
                        help="Embedding path.")
    parser.add_argument('--indices_folder_path', type=str,
                        help="Embedding path.")
    args = parser.parse_args()
    set_seed()

    dataset = load_dataset(SOURCE_NAME[args.dataset_name])['train']
    dataset_embeddings = np.load(args.embedding_path).astype('float32')
    dataset_embeddings = preprocessing.normalize(dataset_embeddings, norm='l2', axis=1)

    # Step 1: Get all categories and compute stratified sample sizes
    category_values, sample_sizes_per_category = calculate_samples_per_category(dataset, args.dataset_name)

    # Step 2: Sample from each category using the corresponding embeddings
    all_sampled_indices, local_sampled_indices_list, cluster_assignment_list = [], [], [] 
    for category_type, n_samples in sample_sizes_per_category.items():
        logging.info(f"Sampling from category: {category_type} ({n_samples} samples)")

        # Get dataset indices where category == category_type
        category_indices = [i for i, src in enumerate(category_values) if src == category_type]

        # Subset prompts and embeddings
        category_prompts = [dataset[i]['prompt'] for i in category_indices]
        category_embeddings = dataset_embeddings[category_indices]

        # Fit k-means clustering on this subset
        best_k, kmeans = cluster_fit_k(category_embeddings, K_CANDIDATES)

        # Define clusters (can be omitted if not needed for sampling logic)
        cluster_assign = define_clusters(category_prompts, kmeans, category_embeddings, best_k, 5)
        
        cluster_assignment_list.append(cluster_assign)

        # Sample indices relative to this category subset
        sampled_local_indices = proportional_cluster_sample_with_diversity_faiss(
            kmeans,
            category_embeddings,
            total_samples=n_samples,
            diversity_ratio=0.75
        )
        
        local_sampled_indices_list.append(sampled_local_indices)

        # Map back to global indices
        flat_sampled_indices = list(itertools.chain.from_iterable(sampled_local_indices))
        sampled_global_indices = [category_indices[i] for i in flat_sampled_indices]
        all_sampled_indices.append(sampled_global_indices)

    with open(os.path.join(args.indices_folder_path, f"{args.dataset_name}_local_indices_per_cluster.json"), 'w') as f:
        json.dump(local_sampled_indices_list, f)

    with open(os.path.join(args.indices_folder_path, f"{args.dataset_name}_indices.json"), 'w') as f:
        json.dump(all_sampled_indices, f)
        
    with open(os.path.join(args.indices_folder_path, f"{args.dataset_name}_cluster_assignment.json"), 'w') as f:
        json.dump(cluster_assignment_list, f)

if __name__ == '__main__':
    run_sampling()
