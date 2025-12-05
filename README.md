# A La Carte Embedding
Python implementation of A La Carte Embedding

以下のように実行します。
＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

text = word2vec.Text8Corpus ("ja.text8.txt")   # このファイルは **https://github.com/Hironsan/ja.text8 ** で取得しました。
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
