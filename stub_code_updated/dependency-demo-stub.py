#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''

import re, sys, nltk, operator
from nltk.parse import DependencyGraph
from nltk.stem.wordnet import WordNetLemmatizer

# Read the lines of an individual dependency parse
def read_dep(fh):
    dep_lines = []
    for line in fh:
        line = line.strip()
        if len(line) == 0:
            return "\n".join(dep_lines)
        elif re.match(r"^QuestionId:\s+(.*)$", line):
            # You would want to get the question id here and store it with the parse
            continue
        dep_lines.append(line)
        
    return "\n".join(dep_lines) if len(dep_lines) > 0 else None
            

# Read the dependency parses from a file
def read_dep_parses(depfile):
    fh = open(depfile, 'r')

    # list to store the results
    graphs = []
    
    # Read the lines containing the first parse.
    dep = read_dep(fh)
    
    # While there are more lines:
    # 1) create the DependencyGraph
    # 2) add it to our list
    # 3) try again until we're done
    while dep is not None:
        graph = DependencyGraph(dep)
        graphs.append(graph)
        
        dep = read_dep(fh)
    fh.close()
    
    return graphs 
    
def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'ROOT':
            return node
    return None
    
def find_node(word, graph):
    for node in graph.nodes.values():
        if node["word"] == word:
            return node
    return None
    
def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
        
    return results


def find_answer(qgraph, sgraph):
    qmain = find_main(qgraph)
    qword = qmain["word"]
    
    # Finds the story node that has the 
    # question graph's ROOT word
    snode = find_node(qword, sgraph)

    if snode:
        for node in sgraph.nodes.values():
            #print("node in nodes.values():", node)
            if node.get('head', None) == snode["address"]:
                #print("Our parent is:", snode)
                #print("Our relation is:", node['rel'])
                if node['rel'] == "prep":
                    deps = get_dependents(node, sgraph)
                    deps = sorted(deps, key=operator.itemgetter("address"))
                    
                    return " ".join(dep["word"] for dep in deps)
    return None

if __name__ == '__main__':
    text_file = "../hw6_dataset/fables-01.sch"
    dep_file = "../hw6_dataset/fables-01.sch.dep"
    q_file = "../hw6_dataset/fables-01.questions.dep"
    
    # Read the dependency graphs into a list 
    sgraphs = read_dep_parses(dep_file)
    qgraphs = read_dep_parses(q_file)

    # Get the first question
    qgraph = qgraphs[0]
    
    # The answer is in the second sentence
    # You would have to figure this out like in the chunking demo
    sgraph = sgraphs[1]
    
    lmtzr = WordNetLemmatizer()
    for node in sgraph.nodes.values():
        tag = node["tag"]
        word = node["word"]
        if word is not None:
            if tag.startswith("V"):
                print(lmtzr.lemmatize(word, 'v'))
            else:
                print(lmtzr.lemmatize(word, 'n'))
    print()

    answer = find_answer(qgraph, sgraph)
    print(answer)
