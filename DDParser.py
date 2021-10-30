'''
####### Official document address:
        
        https://github.com/baidu/DDParser

####### Install DDParser as a python package:

        1. pip install ddparser

####### Input and Output:

        1. Input: text(str)
        2. Output: [{'word':xxx(list), 'head': xxx(list), 'deprel': xxx(list)}]

####### Reference:

        @misc{zhang2020practical,
            title={A Practical Chinese Dependency Parser Based on A Large-scale Dataset},
            author={Shuai Zhang and Lijie Wang and Ke Sun and Xinyan Xiao},
            year={2020},
            eprint={2009.00901},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }
}
'''

# Sample code
from ddparser import DDParser
import warnings
warnings.filterwarnings("ignore")

ddp = DDParser()
# 单条句子
result_1 = ddp.parse("陶影辉写关于推荐系统的论文")
# 输出概率和词性标签
ddp = DDParser(prob=True, use_pos=True)
result_2 = ddp.parse("百度是一家高科技公司")

print(result_1)
print(result_2)

