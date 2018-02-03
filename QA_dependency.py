#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''

import re, sys, nltk, operator
from nltk.parse import DependencyGraph
from nltk.stem.wordnet import WordNetLemmatizer
from QA import get_wordnet_pos

stopwords = set(nltk.corpus.stopwords.words("english"))

def read_dep(fh, id_list):
    # returns a tuple, first element is the dependency graph's lines
    # second element is a list of the question ids
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

# Note: the dependency tags return by Stanford Parser are slightly different than
# what NLTK expects. We tried to change all of them, but in case we missed any, this
# method should correct them for you.
def update_inconsistent_tags(old):
    return old.replace("root", "ROOT")


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


# Finds first word with this relation in the graph
def find_node_word(graph, relation):
    for node in graph.nodes.values():
        if node['rel'] == relation:
            return node['word']
    return None


def find_root(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'ROOT':
            return node
    return None


# find the node with similar word
def find_node(word, graph):
    for node in graph.nodes.values():
        # check for exact match first
        if 'word' in node and node["word"] and node['word'].lower() == word.lower():
            return node
        # return matching lemma if not exact
        if 'lemma' in node and node["lemma"] and node['lemma'].lower() == word.lower():
            return node
    return None

def node_from_address(index, graph):
    for node in graph.nodes.values():
        if 'address' in node and node['address'] and node['address'] == index:
            return node
    return None

def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results += get_dependents(dep, graph)
    return results

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


def pretty_question(qgraph):
    question = []
    for q in qgraph.nodes.values():
        if 'word' in q and q['word'] is not None:
            question.append(q['word'])
    return " ".join(question)

def find_answer(qgraph, sgraphs):
    qword = find_node_word(qgraph, 'ROOT')
    # look for answer in the sgraphs, return the first match
    for sgraph in sgraphs:
        snode = find_node(qword, sgraph)
        if snode is None or 'address' not in snode:
            continue
        for node in sgraph.nodes.values():
            #print("node in nodelist:", node)
            #print("Our relation is:", node['rel'], ", and word is:", node['word'])
            #print("Our node is:", node)
            if node is None or 'head' not in node:
                continue
            if node['head'] == snode["address"]:
                if node['rel'] == "nmod":
                    deps = get_dependents(node, sgraph)
                    deps.append(node)
                    deps = sorted(deps, key=operator.itemgetter("address"))
                    return " ".join(dep["word"] for dep in deps)


if __name__ == '__main__':
    text_file = "fables-01.sch"
    dep_file = "fables-01.sch.dep"
    q_file = "fables-01.questions.dep"

    # Read the dependency graphs into a list
    sgraphs, s_id_to_graph = read_dep_parses(dep_file)
    qgraphs, q_id_to_graph = read_dep_parses(q_file)

    # TODO: You may need to include different rules in find_answer() for
    # different types of questions. For example, the rule here is good for
    # answering "Where was the crow sitting?", but not necessarily the others.
    # You would have to figure this out like in the chunking demo
    for qgraph in qgraphs:
        print("Question:", pretty_question(qgraph), "?")
        answer = find_answer(qgraph, sgraphs)
        print("Answer:", answer)
        print()

    # example of how to use a lemmatizer
    print("\nLemma:")
    lmtzr = WordNetLemmatizer()
    for node in sgraphs[1].nodes.values():
        tag = node["tag"]
        word = node["word"]
        if word is not None:
            if tag.startswith("V"):
                print(lmtzr.lemmatize(word, 'v'))
            else:
                print(lmtzr.lemmatize(word, 'n'))
    print()
