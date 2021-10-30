'''
####### Official document address:
        
        https://opennre-docs.readthedocs.io/en/latest/index.html

####### Install OpenNRE as a python package:

        1. git clone https://github.com/thunlp/OpenNRE.git
        2. pip install -r requirements.txt

####### Input and Output:

        1. Model selection: wiki80_cnn_softmax, wiki80_bert_softmax
        2. Input: {'text':xxx, 'h':{'pos':(3,5)}, 't':{'pos':(10,15)}}
                    h: The first entity;   
                    t: The second entity;
                    pos: The index of each entity. [3,5)
        3. Output: (relation_word(str), probability(float))

####### Reference:

        @inproceedings{han-etal-2019-opennre,
            title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
            author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
            booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
            year = "2019",
            url = "https://www.aclweb.org/anthology/D19-3029",
            doi = "10.18653/v1/D19-3029",
            pages = "169--174"
}
'''

# Sample code
import opennre
import warnings
 
warnings.filterwarnings("ignore")

model = opennre.get_model('wiki80_cnn_softmax')  # wiki80_bert_softmax
result = model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
print(result)  # ('father', 0.7500478625297546)


result_1 = model.infer({'text': 'Liang Zhao is a friend of Yinqiu Huang.', 'h': {'pos': (0, 10)}, 't': {'pos': (26, 38)}})
result_2 = model.infer({'text': 'Beijing is the capital of China.', 'h': {'pos': (0, 7)}, 't': {'pos': (25, 30)}})
print(result_1)  
print(result_2)