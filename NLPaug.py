'''
####### Official document address:
        
        https://github.com/makcedward/nlpaug

####### Install nlpaug as a python package:

        1. pip install numpy requests nlpaug
        2. ensure torch>=1.6.0 and transformers>=4.11.3
        3. (Optional) pandas!=1.3.1

####### Input and Output:

        1. Input:  text: str(a sentence)   or   [str(a sentence),str(a sentence)]
        2. Output: Sentences enhanced by various data augmentation methods

####### Reference:

        @misc{ma2019nlpaug,
            title={NLP Augmentation},
            author={Edward Ma},
            howpublished={https://github.com/makcedward/nlpaug},
            year={2019}
        }
'''

# Sample code
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import warnings
warnings.filterwarnings("ignore")

# Input
text_one = 'The quick brown fox jumps over the lazy dog .'
text_two = 'I hate him'
text_list = [
    'The quick brown fox jumps over the lazy dog .',
    'It is proved that augmentation is one of the anchor to success of computer vision model.'
]

# char level, Keyboar error data augmentation
print("\n****************************char level, Keyboar error data augmentation****************************")
aug_char_key = nac.KeyboardAug()
augmented_text_char_key = aug_char_key.augment(text_one, n=2)   # n: the number of the output sentences 
print("Original:")
print(text_one)
print("Augmented Text:")
print(augmented_text_char_key)

augmented_text_char_key_1 = aug_char_key.augment(text_list)  
print("Original:") 
print(text_list)
print("Augmented Text:")
print(augmented_text_char_key_1)

# word level, antonym replace data augmentation
'''
    please run:
    >>> import nltk
    >>> nltk.download('averaged_perceptron_tagger')
    >>> nltk.download('wordnet')

    or:
    download from https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip
        and unzip it in /home/cqu/anaconda3/envs/wj/nltk_data/taggers
    download from https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip
        and unzip it in /home/cqu/anaconda3/envs/wj/nltk_data/corpora
'''
print("\n****************************word level, antonym replace data augmentation****************************")
aug_word_ant = naw.AntonymAug()
augmented_text_word_ant = aug_word_ant.augment(text_two, n=1)   # n: the number of the output sentences 
print("Original:")
print(text_two)
print("Augmented Text:")
print(augmented_text_word_ant)


# word level, synonym replace data augmentation
print("\n****************************word level, synonym replace data augmentation****************************")
aug_word_syn = naw.SynonymAug()
augmented_text_word_syn = aug_word_syn.augment(text_two, n=1)   # n: the number of the output sentences 
print("Original:")
print(text_two)
print("Augmented Text:")
print(augmented_text_word_syn)
