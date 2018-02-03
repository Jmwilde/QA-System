import sys, nltk, re, operator
from QA_dependency import *
from QA_what import *
from QA import pattern_matcher, lemmatize_list, distance
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem import LancasterStemmer


# Some global tools
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


def question_type(question):
    # returns a tuple with the first element being the keyword indicating question type
    # returns: (keyword, yesOrNoQuestion)
    # who where when why
    # what
    # how is was ould
    # And the second element is a boolean determining if the question can be answered by a yes or no

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

###################################################################
########## SPECIALIZED CASES ##########################################
###################################################################

# WHO -----------------------------------------------------------------------------
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
# Nice guys this is brilliant
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

# iterate through parse trees to find a prepositional phrase near a question word beggining with a locative verb
# arg type: (trees, list)
# return type: string
def where(trees, filtered_question):
    where = None
    locative_verbs = ['in', 'along', 'on', 'under', 'near', 'at', 'was in']
    for tree in trees:
        np_pattern = nltk.ParentedTree.fromstring("(NP)")
        vp_pattern = nltk.ParentedTree.fromstring("(VP)")
        pp_pattern = nltk.ParentedTree.fromstring("(PP)")
        
        np_subtree = pattern_matcher(np_pattern, tree)
        vp_subtree = pattern_matcher(vp_pattern, tree)
        pp_subtree = pattern_matcher(pp_pattern, vp_subtree)
    
        S = ((" ".join(tree.leaves())).lower()).split(' ')
        lemmatized_tree = lemmatize_list(S)
        NP = (" ".join(np_subtree.leaves())).lower()
        VP = (" ".join(vp_subtree.leaves())).lower()

        for word in filtered_question:
            if (word in lemmatized_tree):
                if (pp_subtree != None):
                    PP = (" ".join(pp_subtree.leaves())).lower()
                    if (PP.split(' '))[0] in locative_verbs:
                        loc_verb = PP.split(' ')[0]
                        if (distance(lemmatized_tree, word, loc_verb) < 3):
                            where = PP
    return where


def why(sentence):
    intention_words = ['because', 'in order', 'for it', 'to']
    for word in intention_words:
        if word in sentence:
            why = word + sentence.split(word, maxsplit=1)[1]
            return why
    return None


def when():
    pass


