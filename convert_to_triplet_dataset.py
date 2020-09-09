from sklearn.neighbors import KDTree
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
import scipy
from tqdm import tqdm
from mesh_hierarchy import MeSHGraph

model = None

def random_sample_probs(text, label, vocab):
  sampling_vocab_size = vocab.shape[0] - vocab[vocab.label == label].shape[0]
  probabilities = np.array([1.0/sampling_vocab_size]*vocab.shape[0])
  probabilities[vocab.label == label] = 0.0
  return probabilities


def embedding_based_sampling(text_embedding, label, vocab):
  distances = scipy.spatial.distance.cdist([text_embedding], vocab.embeddings.tolist(), "cosine")[0]
  mean = np.mean(distances)
  std = np.std(distances)
  left_bound = np.min(distances)
  right_bound = np.max(distances)
  probabilities = truncnorm.pdf(distances, left_bound, right_bound, mean, std)
  probabilities[distances == 0.0] = 0.0
  probabilities[vocab.label == label] = 0.0
  return probabilities/np.sum(probabilities)


def hard_sampling(text_embedding, label, vocab, tree):
  top_dist, top_idx = tree.query([text_embedding], k=900)
  probabilities = np.zeros(vocab.shape[0])
  top_idx = [tidx for tidx in top_idx[0] if vocab.iloc[tidx].label!=label][:15]
  top_idx =  np.random.choice(top_idx, 15, replace=False)
  return vocab.iloc[top_idx].text.tolist()


def sample_negative_examples(vocab, probs, n):
  return np.random.choice(vocab.text, n, p=probs, replace=False)


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
  parser.add_argument('--normal', action='store_true')
  args = parser.parse_args()

  texts = pd.read_csv(args.texts, names=['entity_text'], encoding='utf-8', dtype='str')
  texts = texts.fillna('NOTEXT')
  labels = pd.read_csv(args.labels, names=['label'], encoding='utf-8', dtype='str')
  vocab = pd.read_csv(args.vocab, names=['label', 'text'], sep='\t', encoding='utf-8', dtype='str')
  hierarchy = MeSHGraph(args.hierarchy)

  if args.normal:
    word_embedding_model = models.BERT(args.path_to_bert_model)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                 pooling_mode_mean_tokens=True,
                                 pooling_mode_cls_token=False,
                                 pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    vocab['embeddings'] = model.encode(vocab.text.str.lower().tolist(),batch_size=128, show_progress_bar=True)
  data = pd.concat([texts, labels], axis=1)
  data = pd.merge(data, vocab, left_on='label', right_on='label')
  data = data.groupby(['entity_text', 'label']).head(args.positive_count_per_example).reset_index(drop=True)
  prev_label = None
  prev_text  = None
  #tree = KDTree(np.vstack(vocab.embeddings))
  with open(args.save_to, 'w', encoding='utf-8') as output_stream:
    for row_idx, row in tqdm(data.iterrows(), total=data.shape[0]): #text, label in tqdm(zip(texts.text.values.tolist(), labels.label.values.tolist()), total=1700*40):
      text = row['entity_text']
      label = row['label']
      positive_example = row['text']
      if args.normal and (prev_text != text or prev_label != label):
        text_embedding =  model.encode([text.lower()])[0]
        negative_examples = hard_sampling(text_embedding, label, vocab, tree)
      elif prev_text != text or prev_label != label:
        sampling_probabilities = random_sample_probs(text, label, vocab)
      negative_examples = sample_negative_examples(vocab, sampling_probabilities, args.negatives_count_per_example)
      for negative_example in negative_examples:
        # negative_example = sample_negative_examples(vocab, sampling_probabilities)
        output_stream.write('{}\t{}\t{}\n'.format(text, positive_example, negative_example))
      parent_labels = hierarchy.get_parents(label)
      parent_pos_examples = vocab[vocab.label.isin(parent_labels)]['text'].tolist()
      for _ in range(min(args.parents_sample_count, len(parent_pos_examples))): #parent_pos_examples[:args.parents_sample_count]:
        parent_pos_example = np.random.choice(parent_pos_examples)
        snegative_example = np.random.choice(negative_examples)
        output_stream.write('{}\t{}\t{}\n'.format(positive_example, parent_pos_example, snegative_example))
      prev_label = label
      prev_text = text
