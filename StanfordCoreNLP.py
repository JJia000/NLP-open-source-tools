'''
####### Official document address:
        
        https://stanfordnlp.github.io/CoreNLP/index.html

####### Install stanfordnlp as a python package:

        1. pip install stanfordcorenlp    or    git clone https://github.com/Lynten/stanford-corenlp.git
        2. download corresponding file from https://stanfordnlp.github.io/CoreNLP/history.html

####### Input and Output:

        1. Input: text(str)
        2. Output: [('relation', num_1, num_2), ......]

####### Reference:

        @inproceedings{2014The,
            title={The Stanford CoreNLP Natural Language Processing Toolkit},
            author={ Manning, C. D.  and  Surdeanu, M.  and  Bauer, J.  and  Finkel, J.  and  Mcclosky, D. },
            booktitle={Proceedings of 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
            year={2014},
        }
}
'''

# Sample code
from stanfordcorenlp import StanfordCoreNLP
import warnings
warnings.filterwarnings("ignore")

nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2021-05-14/stanford-corenlp-4.2.2')

sentence = 'WangShiqi is a lovely girl, she likes to study'
print('Tokenize:', nlp.word_tokenize(sentence))
print('Dependency Parsing:', nlp.dependency_parse(sentence))
print('Coref:', nlp.coref(sentence))

nlp.close()