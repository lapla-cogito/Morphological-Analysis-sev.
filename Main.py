import gensim
from gensim import corpora
from pprint import pprint
from collections import defaultdict
from janome.tokenizer import Tokenizer

#test
s = Tokenizer()
for token in s.tokenize("I am John"):
    print(token)
