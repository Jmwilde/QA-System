'''
Question Answering System
Henry, John, Ben, Ryan

Usage: python3 QA.py <process_stories file>

Example: python3 QA.py hw7_dataset/process_stories.txt
'''

'''
Table of Contents:
- GLOBAL TOOLS
- RANDOM FUNCTIONS
- NORMALIZE
- POS_TAG/LEMMATIZE
- GET QUESTION, SENTENCES, ETC
- DEP GRAPH FUNCTIONS
- PARSE TREE FUNCTIONS
- GET BASE ANSWER
- WORD2VEC
- WHERE QUESTION
- WHO QUESTION
- WHY QUESTION
- MAIN
'''

import sys, nltk, operator, string, re, os, argparse, gensim
import numpy as np
import sys, nltk, operator, string, re
import argparse
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.tree import Tree
from nltk.parse import DependencyGraph

##################################################################################
# GLOBAL TOOLS -------------------------------------------------------------------
##################################################################################

stopwords = set(nltk.corpus.stopwords.words("english"))

GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """
chunker = nltk.RegexpParser(GRAMMAR)
lmtzr = WordNetLemmatizer()
stemmer =  LancasterStemmer()

##################################################################################
# RANDOM FUNCTIONS ---------------------------------------------------------------
##################################################################################

# Read the file from disk
def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()
    return text

# creates a list of dictionaries
# arg type: (list, list, list)
# return type: list
def make_dictionary_list(questions, filtered_questions, IDs, story_types, difficulty):
    l = []
    for i, j, k, y, z in zip(questions, filtered_questions, IDs, story_types, difficulty):
        dic = {}
        dic['question'] = i
        dic['filtered_question'] = j
        dic['ID'] = k
        dic['story_type'] = y
        dic['difficulty'] = z
        l.append(dic)
    return l

# writes answers to the output file
# arg type: (string, string)
# return type: NONE
def write_output(fh, answer, qID):
    fh.write('QuestionID: ' + qID)
    fh.write('\n')
    fh.write('Answer: ' + answer)  
    fh.write('\n\n') 

def write_debug_output(fh, question, answer, qID, story_type, difficulty):
    fh.write('QuestionID: ' + qID)
    fh.write('\n')
    fh.write('Difficulty: ' + difficulty)
    fh.write('\n')
    fh.write('story_type: ' + story_type)
    fh.write('\n')
    fh.write('Question: ' + question)
    fh.write('\n')
    fh.write('Answer: ' + answer)  
    fh.write('\n\n')

# gets a list of questions, IDs, and story types
# arg type: string
# return type: (list, list, list)
def get_Q_I_T_D(qtext):
    questions = get_questions(qtext)
    IDs = split(get_IDs(qtext))
    story_types = get_type(qtext)
    difficulty = get_difficulty(qtext)
    return questions, IDs, story_types, difficulty

# Creates list of 2 element lists
# Each element: [questions file, data prefix up to the '.']
def process_file_names(stories_file):
    files = []
    dataset = "hw8_dataset/"
    with open(stories_file) as infile:
        for story in infile:
            story = story.strip()
            data_story = dataset + story + "."
            fnames = [data_story + "questions", data_story]
            files.append(fnames)
    return files

# returns a tuple with the first element being the keyword indicating question type
# returns: (keyword, yesOrNoQuestion)
def question_type(question):
    # Group who and whom together
    qtoken = re.search(r'\b([wW]ho|[wW]hom)\b', question.lower())
    if qtoken and len(qtoken.group(1)) > 0:
        return ("who", False)

    # Non-binary questions
    # Example: (When) could blah blah, (Where) did blah, (What) should blah
    qtoken = re.search(r'\b([wW]hose|[wW]hat|[wW]here|[wW]hen|[wW]hy|[wW]hat|[hH]ow)\b', question.lower())
    if qtoken and len(qtoken.group(1)) > 0:
        return (qtoken.group(1).lstrip(), False)

    # Binary (Yes-No) questions
    # (Was) the password correct?
    qtoken = re.search(r'\b([iI]s|[wW]as|[wW]ould|[cC]ould|[sS]hould|[cC]an)\b', question.lower())
    if qtoken and len(qtoken.group(1)) > 0:
        return (qtoken.group(1).lstrip(), True)

    # If nothing found, default to boolean keyword search
    # Example: The sky's color happens to be blue?
    return ("is", True)


def specialized_answer(question):
    # Categories the question based on the question word and
    # attempts to perform a special case function if it applies,
    # Otherwise, default to the general case
    question_word, yesno_question = question_type(question)
    # TODO: Populate this as we create special cases depending on question type
    #if question_word == "who":
    #    return who(question)
    return None

##################################################################################
# NORMALIZE ----------------------------------------------------------------------
##################################################################################

# normalizes a sentence
# arg type: (string, list) 
# return type: list
def normalize(sentence):
    stopwords = set(nltk.corpus.stopwords.words("english"))
    normalized_sentence = tokenize(sentence)
    normalized_sentence = remove_stopwords(normalized_sentence, stopwords)
    normalized_sentence = remove_punct(normalized_sentence)
    return normalized_sentence

# tokenizes a sentnece
# arg type: string
# return type: list
def tokenize(l):
    return nltk.word_tokenize(l)

# removes stopwords from a tokenized sentnece
# arg type: (list, list)
# return type: list
def remove_stopwords(l, stopwords):
    filtered_words = [t.lower() for t in l if t.lower() not in stopwords]
    return filtered_words

# removes punctuation from a tokenized sentnece
# arg type: list
# return type: list
def remove_punct(l):
    for i in l:
        if i[0] in string.punctuation:
            l.remove(i)
    return l

##################################################################################
# POS_TAG/LEMMATIZE ---------------------------------------------------------------
##################################################################################

# lemmatizes a tokenized sentnece
# arg type: list
# return type: list
def lemmatize_list(l):
    lemmatized_list = pos(l)
    lemmatized_list = [lemmatize_tuple(lemma) for lemma in lemmatized_list]
    return lemmatized_list

# pos_tags a tokenized sentnece
# arg type: list
# return type: list
def pos(l):
    return nltk.pos_tag(l)

# returns the wordnet tag of a pos_tagged tuple
# arg type: string
# return type: string
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# lemmatizes a pos_tagged tuple
# arg type: tuple
# return type: string
def lemmatize_tuple(tup):
    lemma = WordNetLemmatizer()
    tag = get_wordnet_pos(tup[1])
    if (tag != ''):
        if (tup[0] == 'felt'):
            lemmatized_word = 'feel'
        else:
            lemmatized_word = lemma.lemmatize(tup[0], tag)
    else:
        lemmatized_word = tup[0]
    return lemmatized_word

##################################################################################
# GET QUESTION, SENTENCES, ETC ---------------------------------------------------
##################################################################################

# tokenize a text into a list of sentences
# arg type: string
# return type: list
def get_sent(text):
    return nltk.sent_tokenize(text)

# split prefixes from questions
# arg type: list
# return type: list
def split(l):
    j = []
    for i in l:
        j.append(i.split(': ')[1])
    return j

# returns a list of questions from a question file
# arg type: string
# return type: list
def get_questions(text):
    matches = re.findall(r'Question:\s*(?P<ques>.*)\n', text)
    return [match.strip() for match in matches]

# returns a list of IDs from a question file
# arg type: string
# return type: list
def get_IDs(text):
    return re.findall(r'QuestionID: [0-9A-Za-z\s,;-]+(?=\n)', text)

# returns a list of story types from a question file
# arg type: string
# return type: list
def get_type(text):
    matches = re.findall(r'Type:\s*(?P<type>.*)\n*', text)
    return [match.strip() for match in matches]

def get_difficulty(text):
    matches = re.findall(r'Difficulty:\s*(?P<Difficulty>.*)\n*', text)
    return [match.strip() for match in matches]

# normalizes and lemmatizes the list of questions
# arg type: (list, list)
# return type: list
def get_filtered_questions(questions):
    return [lemmatize_list(normalize(question)) for question in questions]

# normalizes and lemmatizes the list of sentences
# arg type: (list, list)
# return type: list
def get_filtered_sentences(sentences):
    return [lemmatize_list(normalize(sentence)) for sentence in sentences]

##################################################################################
# DEP GRAPH FUNCTIONS ------------------------------------------------------------
##################################################################################

# Read the dependency parses from a file
def read_dep_parses(depfile):
    fh = open(depfile, 'r')

    # dict to store the results
    id_to_graphs = {}
    graphs = []
    # default to the filename if no question id found
    question_ids = []

    # Read the lines containing the first parse.
    dep, question_ids = read_dep(fh, question_ids)

    # While there are more lines:
    # 1) create the DependencyGraph
    # 2) add it to our list
    # 3) try again until we're done
    while dep is not None:
        graph = DependencyGraph(dep)
        graphs.append(graph)

        dep, question_ids = read_dep(fh, question_ids)
    fh.close()

    # Finally, build a dictionary out of the given question ids to dependency graphs
    for i in range(len(question_ids)):
        # print("ID:", question_ids[i], "Graph:", i)
        # print(graphs[i].tree().pretty_print())
        id_to_graphs[question_ids[i]] = graphs[i]

    # id_to_graphs is a dictionary that maps
    return (graphs, id_to_graphs)

# returns a tuple, first element is the dependency graph's lines
# second element is a list of the question ids
def read_dep(fh, id_list):
    dep_lines = []
    for line in fh:
        line = line.strip()
        match = re.match(r"QuestionId:\s*(?P<id>.*)", line)
        if len(line) == 0:
            # End of a question
            return (update_inconsistent_tags("\n".join(dep_lines)), id_list)
        elif match:
            id_list.append(match.group('id'))
            continue
        dep_lines.append(line)
    if len(dep_lines) > 0:
        return (update_inconsistent_tags("\n".join(dep_lines)), id_list)
    else:
        return (None, id_list)

# find the node with similar word
def find_node(word, graph):
    best_node = None
    best_score = 0
    for node in graph.nodes.values():
        # check for exact match first
        if 'word' in node and node["word"] and node['word'].lower() == word.lower():
            return node
        # return matching lemma if not exact
        if 'lemma' in node and node["lemma"] and node['lemma'].lower() == word.lower():
            return node
        # Otherwise return the node with the best relationship with the key
        score = symilarity(word, node["word"])
        if score <= best_score:
            best_node = node
            best_score = score
    return best_node

def node_from_address(index, graph):
    for node in graph.nodes.values():
        if 'address' in node and node['address'] and node['address'] == index:
            return node
    return None

# Finds first word with this relation in the graph
def find_node_word(graph, relation):
    for node in graph.nodes.values():
        if node['rel'] == relation:
            return node['word']
    return None

def get_deps_of_type(node, graph, dep_tag):
    subjs = []
    if dep_tag in node['deps'].keys():
        index_list = node['deps'][dep_tag]
        for index in index_list:
            # Convert the node index to the node itself
            subj = node_from_address(index, graph)
            # And add it to the list we're building
            if subj:
                subjs.append(subj)
    return subjs

def get_graph_words(graph):
    words = []
    for node in graph.nodes.values():
        if 'word' in node and node['word'] is not None:
            words.append(node['word'])
    return words

def graph_overlap(qbow, sgraphs):
    # Collect all the candidate answers
    answers = []
    for sgraph in sgraphs:
        words = get_graph_words(sgraph)
        sgraph_words = set([word.lower() for word in words if word.lower() not in stopwords])
        overlap = len(qbow & sgraph_words)
        answers.append((overlap, sgraph))
        
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)
    best_answer = (answers[0])[1]    
    return best_answer

def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results += get_dependents(dep, graph)
    return results

def find_root(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'ROOT':
            return node
    return None

# Note: the dependency tags return by Stanford Parser are slightly different than
# what NLTK expects. We tried to change all of them, but in case we missed any, this
# method should correct them for you.
def update_inconsistent_tags(old):
    return old.replace("root", "ROOT")

##################################################################################
# PARSE TREE FUNCTIONS -----------------------------------------------------------
##################################################################################

# read constituency parse file
# arg type: file
# return type: Tree
def read_con_parses(parfile):
    fh = open(parfile, 'r')
    lines = fh.readlines()
    fh.close()
    return [Tree.fromstring(line) for line in lines]

# See if our pattern matches the current root of the tree
# arg type: (string, Tree)
# return type: string
def matches(pattern, root):
    # Base cases to exit our recursion
    # If both nodes are null we've matched everything so far
    if root is None and pattern is None: 
        return root
        
    # We've matched everything in the pattern we're supposed to (we can ignore the extra
    # nodes in the main tree for now)
    elif pattern is None:                
        return root
        
    # We still have something in our pattern, but there's nothing to match in the tree
    elif root is None:                   
        return None

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    # Otherwise they labels need to match
    elif plabel == rlabel:
        # If there is a match we need to check that all the children match
        # Minor bug (what happens if the pattern has more children than the tree)
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild) 
            if match is None:
                return None 
        return root
    return None

# pattern matcher
# arg type: (string, Tree)
# return type: ?
def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None

##################################################################################
# GET BASE ANSWER ----------------------------------------------------------------
##################################################################################

# choose sentence from the text which contains the most words from the question
# arg type: (list, list, list)
# return type: string
def get_answer_sentence(filtered_sentences, sentences, question):
    best = 0
    answer = ''
    for sentence_filtered, sentence_raw in zip(filtered_sentences, sentences):
        count = 0
        for word in set(sentence_filtered):
            if word in question:
                count += 1
        if (count > best):
            best = count
            answer = sentence_raw
    return answer    

# choose sentence from the text which contains the most words from the question
# arg type: (list, list, list)
# return type: string
def w2v_get_answer_sentence(filtered_sentences, sentences, question):
    best = float('inf')
    answer = ''
    for sentence_filtered, sentence_raw in zip(filtered_sentences, sentences):
        angle = sentence_angle(sentence_filtered,question)
        print(angle)
        if (angle < best):
            best = angle
            answer = sentence_raw
    return answer

##################################################################################
# WORD2VEC -----------------------------------------------------------------------
##################################################################################

# word2vec model loading
w2v = os.path.join("GoogleNews-vectors-negative300.bin")
w2vecmodel=gensim.models.KeyedVectors.load_word2vec_format(w2v, binary=True)
# returns an average of the word vectors in a sentence

def sent2vec(words):
    res = 0
    if hasattr(w2vecmodel,'vector_size'):
        res = np.zeros(w2vecmodel.vector_size)
    count = 0
    for word in words:
        if word in w2vecmodel:
            count += 1
            res += w2vecmodel[word]
    if count != 0:
        res /= count
    return res

def maxima_match(keyword, words):
    # Returns the closest matching word vector to the keyword in
    # Setup
    if keyword not in w2vecmodel:
        return sent2vec(words) # Default if keyword isn't in the model

    # Looks through the list of tokens for the closest matching
    # to keyword and returns it.
    best_score = 0
    closest_match = None
    for word in words:
        if word in w2vecmodel:
            score = relation_vector(keyword, word)
            if score > best_score:
                best_score = score
                closest_match = word
    print ("Best score: " + str(best_score))
    return closest_match


def relation_vector(wordA, wordB):
    if not wordA or not wordB or wordA not in w2vecmodel or wordB not in w2vecmodel:
        return 0;
    return np.linalg.norm(w2vecmodel[wordA] - w2vecmodel[wordB])

'''
def sent2vec(words):
    res = np.zeros(w2vecmodel.vector_size)
    for word in words:
        word = word2v(word)
        for i in range(w2vecmodel.vector_size):
            if abs(word[i]) > abs(res[i]):
                res[i] = word[i]
    return res
'''

# returns the vector representation of a word
def word2v(word):
    res = np.zeros(w2vecmodel.vector_size)
    if word in w2vecmodel:
        res += w2vecmodel[word]
    return res

# get the angle between 2 words
def word_angle(w1,w2):
    # convert to words to vectors
    w1,w2 = word2v(w1),word2v(w2)
    return angle_between(w1,w2)

# get the angle between 2 sentences
def sentence_angle(s1,s2):
    # convert to words to vectors
    s1,s2 = sent2vec(s1),sent2vec(s2)
    return angle_between(s1,s2)

# NOTE: 2 functions below taken from:
# https://stackoverflow.com/questions/2827393/
# angles-between-two-n-dimensional-vectors-in-python

# gets the normalization (unit-length form) of a vector
def unit_vector(vector):
    if np.count_nonzero(vector) == 0:
        return vector
    return vector / np.linalg.norm(vector)

# returns the angle between 2 np vectors
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.dot(v1_u, v2_u))

##################################################################################
# WHERE QUESTION -----------------------------------------------------------------
##################################################################################

# iterate through parse trees to find a prepositional phrase near a question word beggining with a locative verb
# arg type: (trees, list)
# return type: string
def where(trees, filtered_question, question, difficulty):
    where = None
    
    fq = nltk.pos_tag(filtered_question)
    locative_verbs = ['onto', 'in', 'along', 'on', 'under', 'near', 'at', 'was in', 'into', 'upon', 'over']


    for tree in trees:
        if (difficulty == 'Hard'):
            S = ((" ".join(tree.leaves())).lower()).split(' ')
            for x in filtered_question:
                syn_x = str(wn.synsets(x)[0])
                syn_x = syn_x.strip('Synset')
                syn_x = syn_x.strip('()')
                syn_x = syn_x.strip("''")
                synset_x = wn.synset(syn_x)
                for y in S:
                    if (len(wn.synsets(y)) != 0):
                        syn_y = str(wn.synsets(y)[0])
                        syn_y = syn_y.strip('Synset')
                        syn_y = syn_y.strip('()')
                        syn_y = syn_y.strip("''")
                        synset_y = wn.synset(syn_y)
                        similarity = synset_x.path_similarity(synset_y)
                        if (similarity != None and similarity >= .5 and similarity < 1):
                            filtered_question.remove(x)
                            filtered_question.append(y)

        S = ((" ".join(tree.leaves())).lower()).split(' ')
        lemmatized_tree = lemmatize_list(S)

        if (len(filtered_question) == 1):
            for subtree in tree.subtrees():
                tree_list = ((" ".join(subtree.leaves())).lower()).split(' ')
                if (filtered_question[0] in tree_list):
                    if (tree_list.index(filtered_question[0]) != len(tree_list)-1):
                        next_item = tree_list[tree_list.index(filtered_question[0])+1]
                        if (next_item == 'was'):
                            pp_pattern = nltk.ParentedTree.fromstring("(PP)")
                            pp_subtree = pattern_matcher(pp_pattern, subtree)
                            PP = (" ".join(pp_subtree.leaves())).lower()
                            if (PP.split(' ')[0] in locative_verbs):
                                where = PP
        else:
            for word in filtered_question:
                if (word in lemmatized_tree):
                    for subtree in tree.subtrees():
                        if len((" ".join(subtree.leaves())).split(' ')) > 1:
                            PP = (" ".join(subtree.leaves())).lower()
                            if (PP.split(' '))[0] in locative_verbs:
                                loc_verb = PP.split(' ')[0]
                                if (distance(lemmatized_tree, word, loc_verb) < 3):
                                    where = PP
    return where

# return distance between two list elements
# arg type: (list, list_element, list_element)
#return type: int
def distance(l, word_one, word_two):
    return l.index(word_two) - l.index(word_one) 

##################################################################################
# WHO QUESTION -------------------------------------------------------------------
##################################################################################

def who(qgraph, sgraphs, base_answer):
    who = None

    # Key relations in the question.
    # Given a question graph and an story sentence graph, we look for
    # the these relations in the qgraph and try to find these words
    # in the sgraph. First one to find a non-None answer wins.
    # Preference given to the root word of the question.
    relations = ['ROOT', 'nsubj', 'dobj']
    keywords = []

    for rel in relations:
        keyword = find_node_word(qgraph, rel)
        if not keyword:
            continue
        keyword, tag = nltk.pos_tag(nltk.word_tokenize(keyword))[0]

        # Process the keyword
        lem_keyword = lmtzr.lemmatize(keyword, get_wordnet_pos(tag))
        stem_keyword = stemmer.stem(lem_keyword)
        keywords.append(lem_keyword)
        qwords = get_graph_words(qgraph)
        filtered_qwords = set([word.lower() for
                               word in qwords if word.lower() not in stopwords])

        # Find the most overlap sgraph
        sgraph = graph_overlap(filtered_qwords, sgraphs)
        # sgraph.tree().pretty_print()

        # Look for the node in story graph that has the keyword
        for node in sgraph.nodes.values():
            current_word = node['word']
            if not current_word:
                continue
            word, tag = nltk.pos_tag(nltk.word_tokenize(current_word))[0]
            lem_word = lmtzr.lemmatize(word, get_wordnet_pos(tag))
            stem_word = stemmer.stem(word)

            # Compare the lemmatized versions
            if lem_word == lem_keyword:
            # if stem_word == stem_keyword:
                while node['word'] is not None:
                    for rel, address in node['deps'].items():
                        if rel == 'nsubj' or rel == 'nmod':
                            # Note address is a list of addresses.
                            # Let's assume there's only one nsubj/nmod.
                            answer_node = sgraph.nodes[address[0]]
                            deps = get_dependents(answer_node, sgraph)
                            deps.append(answer_node)
                            deps = sorted(deps, key=operator.itemgetter("address"))
                            who = " ".join(dep["word"] for dep in deps)
                            return who

                    # Go up a level in the graph
                    next_node = sgraph.nodes[node['head']]

                    # Avoid infinite loops caused by cycles in the graph
                    if next_node['address'] == node['address']:
                        break
                    node = next_node

    # Fallback regex noun phrase approach.
    # Needs more work, as it's only good some of the time.
    # Greatest flaw is that it ends with incorrect answers instead of None.
    if who is None:
        base_tokens = nltk.pos_tag(nltk.word_tokenize(base_answer))
        split_word = None

        # Find the word in base answer to split sentence on
        for kw in keywords:
            for word, tag in base_tokens:
                stem_word = stemmer.stem(word)
                lem_word = lmtzr.lemmatize(word, get_wordnet_pos(tag))
                if lem_word == kw:
                    split_word = word
                    break
        if not split_word:
            return None

        # Split on the qroot word
        sentence = base_answer.split(split_word, maxsplit=1)[0]

        # Get the last NP of the sentence
        nps = get_noun_phrases(sentence)
        noun_phrase = nps[-1:]
        if nps:
            who = " ".join(noun_phrase[0])

    return who


# One possible approach to answering 'who is the story about?' questions.
# Find all living things mentioned in the first 3 words of each sentence.
# Surprisingly somewhat accurate.
def about(story_sentences):
    things = []
    for sent in story_sentences:
        words = nltk.pos_tag(nltk.word_tokenize(sent))
        start_words = words[:3]  # Get first 3 words
        for word, tag in start_words:
            if tag.startswith('NN') and living_thing(word):
                    things.append(word.lower())
    answers = list(set(things))
    num_answers = len(answers)

    # Inserting the proper determiners and conjunctions like 'a' and 'and'
    a_or_an = 'a'
    vowels = 'a', 'e', 'i', 'o', 'u'
    if num_answers:
        if num_answers > 2:
            for i, a in enumerate(answers[:-1]):
                a_or_an = 'a'
                if a.startswith(vowels):
                    a_or_an = 'an'
                answers[i] = "{} {},".format(a_or_an, a)
            p = " ".join(answers[:-1])
            answer = p + " and a {}".format(answers[-1])
            return answer
        elif num_answers == 2:
            if answers[1].startswith(vowels):
                    a_or_an = 'an'
            answers.insert(1,'and {}'.format(a_or_an)) 
            answer = "a " + " ".join(answers)
            return answer.strip()
        else:
            answer = "a " + answers[0]
            return answer.strip()
    else:
        return None


# Use wordnet to check if a word is a living thing
def living_thing(word):
    things = ['living_thing.n.01', 'person.n.01', 'animal.n.01']
    syns = wn.synsets(word)
    if syns:
        syn = wn.synsets(word)[0] # First synset
        for synset in syn.hypernym_paths()[0]:  # First path
            if synset.name() in things:
                return True
    return False


def np_filter(x):
    return x.label() == 'NP'


# Gets all words in a tree or subtree
def tree_words(t):
    return [word for word, tag in t.leaves()]


# Gets a list of noun phrases from a sentence
# Each NP is a sublist of the returned list
def get_noun_phrases(sentence):
    noun_phrases = []
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(sentence))
    tree = chunker.parse(tagged_tokens)
    # print(tree)
    for subtree in tree.subtrees(filter=np_filter):
        np_words = [word for word in tree_words(subtree)]
        noun_phrases.append(np_words)
    return noun_phrases

##################################################################################
# WHAT QUESTION ------------------------------------------------------------------
##################################################################################

def what(question, question_dep_graph, story_dep_trees, story_parse_trees, filtered_question):
    # Try to analyze the meaning of the question's action
    return_answer = smart_what(question, question_dep_graph, story_dep_trees)
    #print("Smart answer: " + str(return_answer))
    if not return_answer:
        return_answer = shallow_what(story_parse_trees, question, question_dep_graph)
    return return_answer

def shallow_what(parse_trees, question, question_dep_graph):
    root = find_root(question_dep_graph)['word']
    keywords = question.split(root)

    tree = best_candidate_tree(parse_trees, root, keywords)
    np_pattern = nltk.ParentedTree.fromstring("(NP)")
    vp_pattern = nltk.ParentedTree.fromstring("(VP)")

    np_subtrees = thorough_pattern_matcher(np_pattern, tree)
    vp_subtrees = thorough_pattern_matcher(vp_pattern, tree)

    all_phrases = []
    for np_subtree in np_subtrees:
        NP = (" ".join(np_subtree.leaves())).lower()
        all_phrases.append(NP)
    for vp_subtree in vp_subtrees:
        VP = (" ".join(vp_subtree.leaves())).lower()
        all_phrases.append(VP)

    # Choose the best np or vp from the candidate sentence
    main_lemma = lemmatize_list([root])[0]
    extra_words = lemmatize_list(keywords)
    best_phrase = None
    highest_score = -999999;
    for phrase in all_phrases:
        phrase_words = lemmatize_list(tokenize(phrase))
        phrase_score = 50

        for word in phrase_words:
            # If the phrase has the main word in it, it's good
            if word == main_lemma:
                phrase_score += 9999
            # Might be bigger than necessary if parts of the question are in it
            if word in extra_words:
                phrase_score -= 10
            # Also take into account the length of the sentence for tiebreakers
            phrase_score -= len(phrase_words)
        # and after calculating the score keep track of the best
        if phrase_score > highest_score:
            highest_score = phrase_score
            best_phrase = phrase
    # then return that best
    return best_phrase

def best_candidate_tree(parse_trees, main_key,  list_of_keys):
    candidate_trees = []
    for current_tree in parse_trees:
        S = ((" ".join(current_tree.leaves())).lower()).split(' ')
        lemmatized_tree = lemmatize_list(S)
        entire_sentence = (" ".join(lemmatized_tree)).lower()
        if main_key in entire_sentence:
            candidate_trees.append(current_tree)

    # Now match as many secondary keywords as possible
    if len(candidate_trees) == 0:
        return None

    best_candidate = candidate_trees[0]
    best_match = 0
    for cand in candidate_trees:
        candidate_sentence = (" ".join(cand.leaves())).lower()
        current_matches = 0
        for keyword in list_of_keys:
            # Count the number of matching keywords in the sentence
            current_matches += sent_symilarity(keyword, nltk.sent_tokenize(candidate_sentence))
        # Check to see if a new record was met
        if current_matches > best_match:
            best_candidate = cand
            best_match = current_matches
    return best_candidate

def sent_symilarity(key, sent_list):
    total_score = 0
    for token in sent_list:
        total_score += symilarity(key, token)
    total_score = total_score/len(sent_list)
    return total_score

def symilarity(wordX, wordY):
    similarity_score = 0
    synset_x = None
    synset_y = None
    if wordX and len(wn.synsets(wordX)) != 0:
        syn_x = str(wn.synsets(wordX)[0])
        syn_x = syn_x.strip('Synset')
        syn_x = syn_x.strip('()')
        syn_x = syn_x.strip("''")
        synset_x = wn.synset(syn_x)
    if wordY and len(wn.synsets(wordY)) != 0:
        syn_y = str(wn.synsets(wordY)[0])
        syn_y = syn_y.strip('Synset')
        syn_y = syn_y.strip('()')
        syn_y = syn_y.strip("''")
        synset_y = wn.synset(syn_y)
    if synset_y and synset_x:
        similarity_score = synset_x.path_similarity(synset_y)
        if similarity_score:
            return similarity_score
    return 0

def smart_what(question, question_dep_graph, story_dep_trees):
    # STEP 1: Get the subject of the question- What
    wh_word_node = find_node("what", question_dep_graph)
    if not wh_word_node:
        # If still nothing found, return None and fail
        return None

    # Step 2: Get the action of the subject
    # We could get here just by taking the root, but sometimes that might not be the case
    wh_root_node = node_from_address(wh_word_node['head'], question_dep_graph )
    if not wh_root_node:
        wh_root_node = find_root(question_dep_graph)
    # print("Root: " + str(wh_root_node['word']))

    # Fix question type
    new_wh_root_node = None
    if (wh_root_node['word'] == 'did' or wh_root_node['word'] == 'does'):
        new_wh_root_node = node_from_address(wh_root_node['head'], question_dep_graph)
    if (wh_root_node['word'] == 'is' or wh_root_node['word'] == 'was'):
        new_wh_root_node = node_from_address(wh_root_node['head'], question_dep_graph)
    if new_wh_root_node:
        wh_root_node = new_wh_root_node


    subjects = []
    # Step 3: Match the action in the question with the same word in the story
    for story_dep in story_dep_trees:
        # Plan A- Look for exact word
        action_in_story_node = find_node(wh_root_node['word'], story_dep)
        # Plan B- Look for matching lemma
        if not action_in_story_node:
            action_in_story_node = find_node(wh_root_node['lemma'], story_dep)
        if action_in_story_node:
            # get the direct objects of the action in question
            dobjs = get_deps_of_type(action_in_story_node, story_dep, 'dobj')
            if len(dobjs) > 0:
                subjects += dobjs
                subjects = purge_subject_list(subjects, question)
                #print("Dobjs: ")
                #print_nodes(dobjs)
            nsubjs = get_deps_of_type(action_in_story_node, story_dep, 'nsubj')
            if len(nsubjs) > 0 and len(subjects) == 0: # dobj > nsubj
                subjects += nsubjs
                subjects = purge_subject_list(subjects, question)
                #print("nsubs: ")
                #print_nodes(nsubjs)
            nmods = get_deps_of_type(action_in_story_node, story_dep, 'amod')
            if len(nmods) > 0:
                subjects += nmods
                subjects = purge_subject_list(subjects, question)
                #print("nmods: ")
                #print_nodes(nmods)
            nsubjpasses = get_deps_of_type(action_in_story_node, story_dep, 'nsubjpass')
            if len(nsubjpasses) > 0:
                subjects += nsubjpasses
                subjects = purge_subject_list(subjects, question)
                #print("nsubjpass: ")
                #print_nodes(nsubjpasses)
            if len(subjects) == 0:
                ccomps = get_deps_of_type(action_in_story_node, story_dep, 'ccomp')
                if len(ccomps) > 0:
                    subjects += ccomps
                    subjects = purge_subject_list(subjects, question)
                    #print("Ccomp: ")
                    #print_nodes(ccomps)

    subjects = purge_subject_list(subjects, question)



    if len(subjects) > 0:
        return " ".join([subj_node['word'] for subj_node in subjects] )
    else:
        # print ("ErrorDEP: " + str(action_in_story_node))
        return None

def thorough_pattern_matcher(pattern, tree):
    # doesn't just return the topmost instance of the pattern, it returns all instances
    all_patterns = []
    if tree:
        for subtree in tree.subtrees():
            node = matches(pattern, subtree)
            if node is not None:
                all_patterns.append(node)
    return all_patterns

def print_nodes (list_of_nodes):
    for node in list_of_nodes:
        print (str(node['word']) + ', ')

def purge_subject_list(subjects, question):
    for tok in subjects:
        # remove tokens that are found in the question
        if tok['word'].lower() in question.lower():
            subjects.remove(tok)
    return subjects
##################################################################################
# WHY QUESTION -------------------------------------------------------------------
##################################################################################

def why(sentence):
    intention_words = ['because', 'in order', 'for it', 'to']
    for word in intention_words:
        if word in sentence:
            why = word + sentence.split(word, maxsplit=1)[1]
            return why
    return None

##################################################################################
# MAIN ---------------------------------------------------------------------------
##################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 8')
    parser.add_argument('infile', help='The file that specifies which stories to use.')
    args = parser.parse_args()
    stories_file = args.infile
    question_files = process_file_names(stories_file)

    outfile = open('wilde_sherman_kwok_ball_answers.txt', 'w', encoding='utf-8')
    debug_file = open('debug_my_answers.txt', 'w', encoding='utf-8')

    for file in question_files:
        
        # get all questions, IDs, and story types as lists
        questions, IDs, story_types, difficulty = get_Q_I_T_D(read_file(file[0]))
        questions = [question.lower() for question in questions]
        filtered_questions = get_filtered_questions(questions)

        # make dic of all necessary info
        dictionary_list = make_dictionary_list(questions, filtered_questions, IDs, story_types, difficulty)

        # get the dict of qgraphs indexed by qID
        qgraphs = read_dep_parses(file[1] + 'questions.dep')[1]

        for dictionary in dictionary_list:

            # get the question, ID, and story type
            question = dictionary['question']
            filtered_question = dictionary['filtered_question']
            qID = dictionary['ID']
            story_types = (dictionary['story_type'].lower()).split("|")
            difficulty = dictionary['difficulty']
            qgraph = qgraphs[qID]

            # pick story type
            if (len(story_types) == 2):
                story_type = 'sch'
            else:
                story_type = story_types[0].strip()
            
            # get list of sgraphs associated with the current story type
            sgraphs = read_dep_parses(file[1] + story_type + '.dep')[0]

            # get sentences and nltk pipeline-filtered sentences
            sentences = get_sent(read_file(file[1] + story_type))
            filtered_sentences = get_filtered_sentences(sentences)

            # get the question type
            q_type, boolean_question = question_type(question)

            base_answer = \
                get_answer_sentence(filtered_sentences, sentences, filtered_question)

            if base_answer=='':
                base_answer = \
                    w2v_get_answer_sentence(filtered_sentences, sentences, filtered_question)
            
            if (q_type == "where"):
                parse_trees = read_con_parses(file[1] + story_type + '.par')
                answer = where(parse_trees, filtered_question, question, difficulty)
            elif (q_type == "who"):
                if 'about' and 'story' in question:
                    answer = about(sentences)
                else:
                    answer = who(qgraphs[qID], sgraphs, base_answer)
            elif (q_type == "why"):
                answer = why(base_answer)
            elif (q_type == "what"):
                parse_trees = read_con_parses(file[1] + story_type + '.par')
                answer = what(question, qgraphs[qID], sgraphs, parse_trees, filtered_question)
                #print("Q: " + str(question))
                #print("A: " + str(answer) + '\n')
            else:
                answer = base_answer
            if answer is None:
                answer = ''
            write_debug_output(debug_file, question, answer, qID, story_type, difficulty)
            write_output(outfile, answer, qID)
    
    outfile.close()
    debug_file.close()
