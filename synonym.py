import nltk
from nltk import word_tokenize, pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('punkt')

import gensim
import torch
from transformers import AlbertForMaskedLM, AlbertTokenizerFast
import random
import itertools
import json

def load_glove(glove_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
    return model

def perplexity(tokenizer, model, sentence, max_length=256):
    enc = tokenizer(sentence, return_tensors='pt', max_length=max_length, truncation=True)
    input_ids = enc.input_ids
    target_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    return torch.exp(outputs[0])

def synonym_perturbs(sentence, wordvecs, topk=5):
    words = [w.strip(string.punctuation) for w in sentence.split(" ")]
    tags = pos_tag(words)
    syns = {}
    for i, t in enumerate(tags):
        word, pos = t
        # verbs, adjectives, or adverbs
        do_syns = pos == "VB" or pos == "JJ" or pos == "NN"
        if do_syns: 
            w = word.lower().strip(string.punctuation)
            syns[i] = [s for s, _ in wordvecs.most_similar(w, topn=topk)]
            syns[i].append(w)
    
    # generate sentences from synonym combinations
    idxs = list(syns.keys())
    sentences = []
    for p in itertools.product(*syns.values()):
        perturb_words = words[:]
        for i, j in enumerate(idxs):
            perturb_words[j] = p[i]
        sentences.append(" ".join(perturb_words))
    return sentences

if __name__ == "__main__":
    glove_dim = 100
    glove_path = os.path.join(f"glove.6B.{glove_dim}d.txt")
    glove = load_glove(glove_path)

