from QA import *


# WHAT -------------------------------------------------------------------------------------------------
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
            if keyword in candidate_sentence:
                current_matches += 1
        # Check to see if a new record was met
        if current_matches > best_match:
            best_candidate = cand
            best_match = current_matches
    return best_candidate



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

def print_nodes (list_of_nodes):
    for node in list_of_nodes:
        print (str(node['word']) + ', ')

def purge_subject_list(subjects, question):
    for tok in subjects:
        # remove tokens that are found in the question
        if tok['word'].lower() in question.lower():
            subjects.remove(tok)
    return subjects
'''
def what_generic(question_dep_graph, story_dep_trees):
    # STEP 1: Get the subject of the question- What
    wh_word_node = find_node("what", question_dep_graph)

    # Step 2: Get the action of the subject
    # We could get here just by taking the root, but sometimes that might not be the case
    if wh_word_node:
        wh_root_node = node_from_address(wh_word_node['head'], question_dep_graph )
    else :
        # If the node doesnt exist use the root of the sent
        print("Using root instead...")
        wh_root_node = find_root(question_dep_graph)

    print("Question type: " + wh_root_node['word']) # This print is helpful
    # Fix question type
    if (wh_root_node and \
        (wh_root_node['word'].lower() == 'was' or \
        wh_root_node['word'].lower() == 'is' or \
        wh_root_node['word'].lower() == 'does' or \
        wh_root_node['word'].lower() == 'did' )):
        new_wh_root_node = node_from_address(wh_root_node['head'], question_dep_graph)
        if new_wh_root_node:
            print ("Extra: " + str(wh_root_node['word']))
            wh_root_node = new_wh_root_node

    subjects = []
    # Step 3: Match the action in the question with the same word in the story
    for story_dep in story_dep_trees:
        #story_dep.tree().pretty_print()
        action_in_story_node = find_node(wh_root_node['lemma'], story_dep)
        if action_in_story_node:
            print()
            # get the direct objects of the action in question
            subjects += get_deps_of_type(action_in_story_node, story_dep, 'dobj')
            subjects += get_deps_of_type(action_in_story_node, story_dep, 'nsubj')
            subjects += get_deps_of_type(action_in_story_node, story_dep, 'nsubjpass')
            if len(subjects) == 0:
                print("No deps " + str(action_in_story_node['deps']))

    if len(subjects) > 0:
        return " ".join([subj_node['word'] for subj_node in subjects] )
    return None

'''

def thorough_pattern_matcher(pattern, tree):
    # doesn't just return the topmost instance of the pattern, it returns all instances
    all_patterns = []
    if tree:
        for subtree in tree.subtrees():
            node = matches(pattern, subtree)
            if node is not None:
                all_patterns.append(node)
    return all_patterns