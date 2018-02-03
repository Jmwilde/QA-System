'''
Created on May 14, 2014
@author: Reid Swanson

Modified on May 21, 2015
'''

import re, sys, nltk
from nltk.stem.wordnet import WordNetLemmatizer

# Our simple grammar from class (and the book)
GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

LOC_PP = set(["in", "on", "at"])

def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()
    
    return text

def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences

def pp_filter(subtree):
    return subtree.label() == "PP"

def is_location(prep):
    return prep[0] in LOC_PP

def find_locations(tree):
    # Starting at the root of the tree
    # Traverse each node and get the subtree underneath it
    # Filter out any subtrees who's label is not a PP
    # Then check to see if the first child (it must be a preposition) is in
    # our set of locative markers
    # If it is then add it to our list of candidate locations
    
    # How do we modify this to return only the NP: add [1] to subtree!
    # How can we make this function more robust?
    # Make sure the crow/subj is to the left
    locations = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_location(subtree[0]):
            locations.append(subtree)
    
    return locations

def find_candidates(sentences, chunker):
    candidates = []
    for sent in crow_sentences:
        tree = chunker.parse(sent)
        # print(tree)
        locations = find_locations(tree)
        candidates.extend(locations)
        
    return candidates

def find_sentences(patterns, sentences):
    # Get the raw text of each sentence to make it easier to search using regexes
    raw_sentences = [" ".join([token[0] for token in sent]) for sent in sentences]
    
    result = []
    for sent, raw_sent in zip(sentences, raw_sentences):
        for pattern in patterns:
            if not re.search(pattern, raw_sent):
                matches = False
            else:
                matches = True
        if matches:
            result.append(sent)
            
    return result


def np_filter(x):
    return x.label() == 'NP'


# Gets all words in a tree or subtree
def tree_words(t):
    return [word for word, tag in t.leaves()]


def find_noun_phrases(sentence):
    # r'NounPhrase: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'

    # Our tools
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()

    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(sentence))
    tree = chunker.parse(tagged_tokens)
    for subtree in tree.subtrees(filter=np_filter): # Generate all subtrees
        print("Noun Phrase:", subtree)
        for word in tree_words(subtree):
            print(word, end=' ')
        print()
        # for word, tag in subtree.leaves():
        #     print(word, end=' ')
        # print()

if __name__ == '__main__':
    
    # filename = "fables-01.story"
    # text = read_file(filename)
    
    # # Apply the standard NLP pipeline we've seen before
    # sentences = get_sentences(text)
    
    # # Assume we're given the keywords for now
    # # What is happening
    # verb = "sitting"
    # # Who is doing it
    # subj = "crow"
    # # Where is it happening (what we want to know)
    # loc = None
    
    # # Might be useful to stem the words in case there isn't an extact
    # # string match
    # subj_stem = lmtzr.lemmatize(subj, "n")
    # verb_stem = lmtzr.lemmatize(verb, "v")
    
    # # Find the sentences that have all of our keywords in them
    # # How could we make this better?
    # crow_sentences = find_sentences([subj_stem, verb_stem], sentences)
    
    # # Extract the candidate locations from these sentences
    # locations = find_candidates(crow_sentences, chunker)
    
    # # Print them out
    # for loc in locations:
    #     print(loc)
    #     print(" ".join([token[0] for token in loc.leaves()]))

    sentence = "Throughout the day, many birds drink out of it and bathe in it."
    result = find_noun_phrases(sentence)

    # Lemmatize question root word with proper wordnet tag
    # lem_qroot = lmtzr.lemmatize(qword, wn_tag)
    # print(result)



