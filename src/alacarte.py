# https://github.com/yagays/alacarte_embedding/blob/master/src/alacarte.py
# はエラーになったので、geminiに修正してもらった。↓
# 2025-Dec時点でcolabで実行できることを確認した。
!pip install gensim pyflann-py3 mecab-python3 unidic-lite unidic
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from gensim.models import KeyedVectors
from pyflann import FLANN
import MeCab
import unidic
from collections import Counter
from tqdm import tqdm
import sys
import re
import gensim

def tokenize(text):
    # Use unidic.DICDIR to specify the dictionary path
    wakati = MeCab.Tagger(f"-d {unidic.DICDIR} -O wakati")
    wakati.parse("")
    return wakati.parse(text).strip().split(" ")

def window_without_center(seq, n=1):
    start = 0
    seq_len = len(seq)

    while True:
        center = start + n
        end = center + n + 1

        window_index_list = range(start, end)
        yield seq[center], [seq[i] for i in window_index_list if i != center]

        start += 1
        if end >= seq_len:
            break


def ngram(words, n):
    return [t for t in list(zip(*(words[i:] for i in range(n))))]


class ALaCarteEmbedding:
    def __init__(self, word2vec, tokenize, target_word_list=[], ngram=[1], window_size=1, min_count=1):
        self.w2v = word2vec
        self.embedding_dim = self.w2v.wv.vector_size
        self.vocab = set(self.w2v.wv.key_to_index.keys())
        self.target_word_list = set(target_word_list)
        for word in self.target_word_list:
            self.vocab.add(word)
        self.tokenize = tokenize
        self.ngram = ngram
        self.window_size = window_size
        self.min_count = min_count

        self.c2v = {}
        self.target_counts = Counter()
        self.alacarte = {}
        self.flann = FLANN()

    def _get_embedding_vec(self, token):
        if type(token) == str:
            # for unigram
            if token in self.w2v.wv.key_to_index:
                return self.w2v.wv[token]
            else:
                return np.zeros(self.embedding_dim)
        else:
            # for ngram
            vec = np.zeros(self.embedding_dim)
            for t in token:
                if t in self.w2v.wv.key_to_index:
                    vec += self.w2v.wv[t]
            return vec

    def _make_context_vectors(self, tokens, n):
        if n > 1:
            token_list = ngram(tokens, n)
        else:
            token_list = tokens

        for target_token, context in window_without_center(token_list, self.window_size):
            context_vector = np.zeros(self.embedding_dim)
            if self.target_word_list and target_token not in self.vocab:
                # target_word_list is specified and each target token is not in the vocabulary
                continue

            for token in context:
                context_vector += self._get_embedding_vec(token)

            if target_token in self.c2v:
                self.c2v[target_token] += context_vector
            else:
                self.c2v[target_token] = context_vector
            self.vocab.add(target_token)
            self.target_counts[target_token] += 1

    def build(self, sentences):
        # compute each word’s context embedding
        for sentence_obj in tqdm(sentences):
            # Check if sentence_obj is already a list of tokens (e.g., from Text8Corpus)
            if isinstance(sentence_obj, list) and all(isinstance(token, str) for token in sentence_obj):
                tokens = sentence_obj
            else:
                # Assume it's a raw text string and tokenize it
                tokens = self.tokenize(sentence_obj)

            if len(tokens) > self.window_size * 2 + 1:
                for n in self.ngram:
                    self._make_context_vectors(tokens, n)

        # remove low frequency token
        for word, freq in self.target_counts.items():
            if freq < self.min_count and word in self.vocab:
                self.vocab.remove(word)

        # compute context-to-feature transform
        X_all = np.array([v / self.target_counts[k] for k, v in self.c2v.items() if k in self.vocab])

        X = np.array([v / self.target_counts[k] for k, v in self.c2v.items() if k in self.w2v.wv.key_to_index])
        y = np.array([self.w2v.wv[k] for k, v in self.c2v.items() if k in self.w2v.wv.key_to_index])
        self.A = LinearRegression(fit_intercept=False).fit(X, y).coef_.astype(np.float32)  # emb x emb

        # set a la carte embedding
        self.alacarte = normalize(X_all.dot(self.A.T))
        self.alacarte_vocab = [v for v in self.c2v.keys() if v in self.vocab]

        # make index for similaarity search
        self.flann.build_index(self.alacarte)

    def most_similar(self, word, topn=1):
        # Handle tuple input for n-grams if word is a tuple
        if isinstance(word, tuple):
            word_key = word # keep it as a tuple for lookup
        else:
            word_key = word # assume it's a string

        if word_key not in self.alacarte_vocab:
            # Handle out-of-vocabulary words for most_similar if necessary, or raise an error
            print(f"Warning: '{word}' not in Alacarte Embedding vocabulary. Cannot find similar words.")
            return []

        word_vec = self.alacarte[self.alacarte_vocab.index(word_key)]
        result, dists = self.flann.nn_index(word_vec, num_neighbors=topn)

        if topn != 1:
            result = result[0]
            dists = dists[0]

        output = []
        for i, index in enumerate(result.tolist()):
            # Join back n-grams for output if they were tuples
            text = "".join(self.alacarte_vocab[index]) if isinstance(self.alacarte_vocab[index], tuple) else self.alacarte_vocab[index]
            sim = dists[i]
            output.append((text, sim))
        return output

    def save(self, path):
        with open(path, "w") as f:
            f.write(f"{len(self.alacarte_vocab)} {self.embedding_dim}\n")
            for arr, word_key in zip(self.alacarte, self.alacarte_vocab):
                # Convert tuple n-grams to string for saving
                word_str = "".join(word_key) if isinstance(word_key, tuple) else word_key
                f.write(" ".join([word_str] + [str(np.round(s, 6)) for s in arr.tolist()]) + "\n")

%cd /content/drive/MyDrive/stm/chapter3

from gensim.models import word2vec

text = word2vec.Text8Corpus ("data/ja.text8.txt")
w2v = word2vec.Word2Vec (text, vector_size=100, min_count=10, window=10)

alc = ALaCarteEmbedding(word2vec=w2v,
                        tokenize=tokenize,
                        min_count=10,
                        ngram=[1, 2])
alc.build(text)

# 類似する単語を推定
print("Most similar words to '信濃':")
print(alc.most_similar("信濃"))
print("\nMost similar words to '信濃' (top 10):")
print(alc.most_similar("信濃", topn=10))

# 学習したA La Carte Embeddingをsave()で保存します。
alc.save("model/alacarte_embedding.txt")
print("\nALaCarte Embedding model saved to model/alacarte_embedding.txt")
