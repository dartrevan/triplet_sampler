from sklearn.neighbors import KDTree
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm
from mesh_hierarchy import MeSHGraph
import os

from typing import Optional, Any, List


def load_model(path):
    checkpoint_files = os.listdir(path)
    if 'pytorch_model.bin' in checkpoint_files:
        word_embedding_model = models.BERT(path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return SentenceTransformer(path)


def get_hierarchy_aware_negatives(hierarchy: Any, label: str):
    parents = hierarchy.get_parents(label)
    negatives = []
    for parent in parents:
        children = hierarchy.get_children(parent)
        children = [child for child in children if child !=label]
        negatives += children
    return negatives


def find_last_occurence(ordered_labels, label):
    for i in range(len(labels) - 1, 0, -1):
        if ordered_labels[i][0] == label: return i
    return 0


def get_negative_examples(label: str, negatives_count: int, hierarchy: Optional[Any] = None,
                          hierarchy_aware: bool = True, hard: bool = True, ordered_labels: Optional[List[str]] = None):

    if hierarchy_aware:
        subsample = get_hierarchy_aware_negatives(hierarchy, label)
        ordered_subsample = [(concept_id, concept_name) for concept_id, concept_name in ordered_labels
                             if concept_id in subsample and concept_id != label]
    else:
        last_idx = find_last_occurence(ordered_labels, label)
        ordered_subsample = ordered_labels[last_idx + 1:]

    if not hard:
        ordered_subsample = np.random.permutation(ordered_subsample).tolist()
    negatives = [concept_name for concept_id, concept_name in ordered_subsample[:negatives_count]]
    return negatives


def get_positive_examples(label: str, positives_count: int, hierarchy: Optional[Any] = None,
                          parents_count: int = 0, hard: bool = True, ordered_labels: Optional[List[str]] = None):
    positives = [concept_name for concept_id, concept_name in ordered_labels if concept_id == label]
    if not hard:
        positives = np.random.permutation(positives).tolist()
    positives = positives[:positives_count]
    if parents_count > 0:
        parent_labels = hierarchy.get_parents(label)
        parents = [concept_name for concept_id, concept_name in ordered_labels if concept_id in parent_labels]
        if not hard:
            parents = np.random.permutation(parents).tolist()
    return positives + parents


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--texts')
    parser.add_argument('--labels')
    parser.add_argument('--vocab')
    parser.add_argument('--hierarchy')
    parser.add_argument('--negatives_count_per_example', type=int, default=5)
    parser.add_argument('--positive_count_per_example', type=int, default=10)
    parser.add_argument('--parents_sample_count', type=int, default=1)
    parser.add_argument('--save_to')
    parser.add_argument('--path_to_bert_model')
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--hierarchy_aware', action='store_true')
    args = parser.parse_args()

    texts = pd.read_csv(args.texts, names=['entity_text'], encoding='utf-8', dtype='str')
    texts = texts.fillna('NOTEXT')
    labels = pd.read_csv(args.labels, names=['label'], encoding='utf-8', dtype='str')
    vocab = pd.read_csv(args.vocab, names=['label', 'text'], sep='\t', encoding='utf-8', dtype='str')
    hierarchy = MeSHGraph(args.hierarchy)

    data = pd.concat([texts, labels], axis=1)

    # load model
    if args.normal:
        model = load_model(args.path_to_bert_model)

    prev_label = None
    prev_text = None
    tree = KDTree(np.vstack(vocab.embeddings))
    with open(args.save_to, 'w', encoding='utf-8') as output_stream:
        for row_idx, row in tqdm(data.iterrows(), total=data.shape[0]):
            entity_text = row['entity_text']
            label = row['label']
            entity_embedding = row['embedding']
            order = tree.query(entity_text, k=vocab.shape[0])
            ordered_labels = vocab.sort_values('order')[['label', 'text']].values[order].tolist()
            positive_examples = get_positive_examples(label, args.positives_count_per_example, hierarchy,
                                                      args.parents_sample_count, args.hard, ordered_labels)
            negative_examples = get_negative_examples(label, args.negatives_count_per_example, hierarchy,
                                                      args.hierarchy_aware, args.hard, ordered_labels)
            for positive_example in positive_examples:
                for negative_example in negative_examples:
                    output_stream.write(f'{entity_text}\t{positive_example}\t{negative_example}\n')

