'''
####### Official document address:
        
        https://huggingface.co/transformers/master/quicktour.html

####### Install Transformers as a python package:

        1. pip install transformers

####### Input:

        1. Input:  text: str(a sentence or a document)
'''

# Sample code
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
import warnings
warnings.filterwarnings("ignore")

# Summarization
# ***You need to download the model locally in advance(t5-base).
text = ''''It's a massacre. There are dead!' employee of Charlie Hebdo tells French media outlet before call disconnects'''
model_path_Summarization = "../time_fakenew/fakenew_HAN_simple_2/model/t5-base/"
tokenizer_Summarization = AutoTokenizer.from_pretrained(model_path_Summarization)
model_Summarization = AutoModelWithLMHead.from_pretrained(model_path_Summarization)
summarizer = pipeline("summarization", model=model_Summarization, tokenizer=tokenizer_Summarization)
abstract_temp = summarizer(text, max_length=20, min_length=5)[0]['summary_text']

print(abstract_temp)

# sentiment-analysis
# ***You can use the model_name from https://huggingface.co/models?sort=downloads directly.
text = 'We are very happy to show you the 🤗 Transformers library.'
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
sentiment_temp = classifier(text)

print(sentiment_temp)



# 
