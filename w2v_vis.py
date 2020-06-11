from gensim.models import KeyedVectors
from gensim.models import Word2Vec
# Load gensim word2vec
w2v_path = 'word2vec.model'
# w2v = KeyedVectors.load_word2vec_format(w2v_path)
w2v = KeyedVectors.load(w2v_path)

import io

# Vector file, `\t` seperated the vectors and `\n` seperate the words
"""
0.1\t0.2\t0.5\t0.9
0.2\t0.1\t5.0\t0.2
0.4\t0.1\t7.0\t0.8
"""
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')

# Meta data file, `\n` seperated word
"""
token1
token2
token3
"""
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

# Write meta file and vector file
for index in range(len(w2v.wv.index2word)):
    word = w2v.wv.index2word[index]
    vec = w2v.wv.vectors[index]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

# Then we can visuale using the `http://projector.tensorflow.org/` to visualize those two files.

# 1. Open the Embedding Projector.
# 2. Click on "Load data".
# 3. Upload the two files we created above: vecs.tsv and meta.tsv.