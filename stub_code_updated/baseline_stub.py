#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''

import nltk, re
import operator
from nltk.tree import Tree

# Read the file from disk
def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()
    return text


# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

# get a list of tree objects from the .par input file
def get_parse_trees(trees):
    return [Tree.fromstring(t) for t in trees.split('\n') if t.strip()]

# recursively find noun phrase in a tree
def find_noun_phrase(tree):
    if not isinstance(tree,Tree): return
    if tree.label() == 'NP':
        print(tree)
    else:
        for i in range(len(tree)):
            find_noun_phrase(tree[i])

def get_bow(tagged_tokens, stopwords):
    # returns a lowercase list of tokens with stopwords removed
    return set([t[0].lower() for t in tagged_tokens
               if t[0].lower() not in stopwords])

def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i+1:]


# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
# returns the sentence that has the most overlap
# with the question
def baseline(qbow, sentences, parse_trees, stopwords):
    # Collect all the candidate answers
    answers = []
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)

        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)

        answers.append((overlap, sent))

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = (answers[0])[1]
    return best_answer


if __name__ == '__main__':
    text_file = "../hw6_dataset/fables-01.sch"
    parse_file = "../hw6_dataset/fables-01.sch.par"
    stopwords = set(nltk.corpus.stopwords.words("english"))
    text = read_file(text_file)
    parse_text = read_file(parse_file)
    question = "Where was the crow sitting?"
    qbow = get_bow(get_sentences(question)[0], stopwords)
    sentences = get_sentences(text)
    parse_trees = get_parse_trees(parse_text)
    for p in parse_trees: 
        p.chomsky_normal_form()
        print(p)
    exit()
    for p in parse_trees:
        find_noun_phrase(p)
    answer = baseline(qbow, sentences, parse_trees, stopwords)

    print(" ".join(t[0] for t in answer))
 
