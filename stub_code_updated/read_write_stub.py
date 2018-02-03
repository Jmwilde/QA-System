import nltk
import re
import baseline_stub as base
from collections import OrderedDict

###############################################################################
# Utility Functions ##########################################################
###############################################################################


# returns a dictionary where the question numbers are the key
# and its items are another dict of difficulty, question, type, and answer
# e.g. story_dict = {'fables-01-1': {'Difficulty': x, 'Question': y, 'Type':},
# 'fables-01-2': {...}, ...}
def getQA(filename):
    content = open(filename, 'rU', encoding='latin1').read()
    question_dict = {}
    qa_regex = (r'QuestionID:\s*(?P<id>.*)\n'
                r'Question:\s*(?P<ques>.*)\n('
                r'Answer:\s*(?P<answ>.*)\n){0,1}'
                r'Difficulty:\s*(?P<diff>.*)\n'
                r'Type:\s*(?P<type>.*)\n*')
    for m in re.finditer(qa_regex, content):
        qid = m.group("id")
        question_dict[qid] = {}
        question_dict[qid]['Question'] = m.group("ques")
        question_dict[qid]['Answer'] = m.group("answ")
        question_dict[qid]['Difficulty'] = m.group("diff")
        question_dict[qid]['Type'] = m.group("type")
    return question_dict


def get_data_dict(fname):
    data_dict = {}
    data_types = ["story", "sch", "questions"]
    parser_types = ["par", "dep"]
    for dt in data_types:
        data_dict[dt] = read_file(fname + "." + dt)
        for tp in parser_types:
            data_dict['{}.{}'.format(dt, tp)] = (
                read_file(fname + "." + dt + "." + tp))
    return data_dict


# Read the file from disk
# filename can be fables-01.story, fables-01.sch, fables-01-.story.dep,
# fables-01.story.par
# Returns the raw text of the file
def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()
    return text

##############################################################################
# Question Answering Functions Baseline ######################################
##############################################################################
#######################################################################


stopwords = set(nltk.corpus.stopwords.words("english"))

if __name__ == '__main__':

    # Loop over the files in fables and blogs in order.
    output_file = open("train_my_answers.txt", "w", encoding="utf-8")
    cname_size_dict = OrderedDict()
    cname_size_dict.update({"fables": 2})
    cname_size_dict.update({"blogs": 1})
    for cname, size in cname_size_dict.items():
        for i in range(0, size):
            # File format as fables-01, fables-11
            directory = "../hw6_dataset/"
            fname = "{0}-{1:02d}".format(cname, i+1)
            # print("File Name: ", fname)
            data_dict = get_data_dict(directory + fname)
            questions = getQA(directory + "{}.questions".format(fname))
            for j in range(0, len(questions)):
                qname = "{0}-{1}".format(fname, j+1)
                if qname in questions:
                    print("QuestionID: " + qname)
                    question = questions[qname]['Question']
                    print(question)
                    qtypes = questions[qname]['Type']

            # Read the content of fname.questions.par,
            # fname.questions.dep for hint.
                    question_par = data_dict["questions.par"]
                    question_dep = data_dict["questions.dep"]

                    answer = None
                    # qtypes can be "Story", "Sch", "Sch | Story"
                    for qt in qtypes.split("|"):
                        qt = qt.strip().lower()
                        # These are the text data where you
                        # can look for answers.
                        raw_text = data_dict[qt]
                        par_text = data_dict[qt + ".par"]
                        dep_text = data_dict[qt + ".dep"]

                        # TODO: You need to find the answer for this question.

                        # Basic baseline using just the raw text
                        qbow = base.get_bow(base.get_sentences(question)[0], stopwords)
                        sentences = base.get_sentences(raw_text)
                        answer = base.baseline(qbow, sentences, stopwords)
                    print("Answer: ", end=" ")
                    print(" ".join(t[0] for t in answer))
                    print("")

                    # Save your results in output file.
                    output_file.write("QuestionID: {}\n".format(qname))
                    output_file.write("Answer: {}\n\n".format(answer))
    output_file.close()
