#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
import os


# # Data Import

# In[2]:


script_directory = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(script_directory, "Data/Final_Data.csv")
data = pd.read_csv(data_file_path)


# # Library Import

# In[3]:


#Functions
    #Hashtag Extraction
import re
    #Topic Modelling
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import PorterStemmer
from itertools import chain
import string
import gensim
from gensim import corpora
from collections import Counter


# # Build Functions

# In[4]:


#Hashtags
hashtags = re.compile('#\w*') #search only for words that come after hashtags

#Web Links
links = re.compile('http\S+') #define links


# In[5]:


#Functions to Extract Hashtags and Key Words from message
def extract_hashtags(df,column):
    hashtaglist = [hashtags.findall(str(i)) for i in df[column]] #extract hashtags
    return hashtaglist

#Pre-Processing and Cleaning of Text
stopwordlist = set(stopwords.words('english')) #stopwords
punctuation = set(string.punctuation) #list of common punctuation
lemmatize = WordNetLemmatizer() #create lemmatizing function
stem = PorterStemmer() #create stemming function

#function to simplify tagging of words the way they are used in sentences
#function from GeeksForGeeks
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#lemmatizing function
def word_lemmatizer(string):
    #p = re.compile('#\w*') #define hashtags
    #string = re.sub(p,'',string) #remove hashtags from sentence
    string = re.sub(links,'',string) #remove links from sentence    
    string = " ".join([i for i in string.lower().split() if i not in stopwordlist]) #remove stopwords
    string = ''.join(ch for ch in string if ch not in punctuation) #remove punctuation

    tagged_string = nltk.pos_tag(nltk.word_tokenize(string)) #tag each word in the string with how it is used in the sentence (verb, noun, etc.)
    retagged_string = set(map(lambda x: (x[0], pos_tagger(x[1])), tagged_string)) #use simplified tags
    #following function derived from GeeksForGeeks
    lemmatized_words = set(word if tag is None else lemmatize.lemmatize(word,tag) for word,tag in retagged_string)
    lemmatized_string = " ".join(lemmatized_words) #recombine list of words into message
    return lemmatized_string


#topic modeling function
def topic_modeler(strings,num_topics,num_passes,num_words):
    lda = gensim.models.ldamodel.LdaModel #Define Latent Dirichlet Allocation Model
    strings = [word_lemmatizer(string) for string in strings] #lemmatize messaGES
    strings = [string.split() for string in strings] #split each message up into individual words
    dct = corpora.Dictionary(strings) #create corpus
    matrix = [dct.doc2bow(doc) for doc in strings] #creating a document term matrix
    ldamodel = lda(matrix, num_topics=num_topics, id2word = dct, passes=num_passes)
    return ldamodel.print_topics(num_topics=num_topics, num_words=num_words)

#get all lemmas of word
def fetch_lemmas(word):
    word = word.replace(" ","_")#replace space with underscore to allow python to understand the word better
    pos_tags = pos_tag([word]) #how is the word used in a sentence
    pos_retagged = [(word,pos_tagger(tag)) for (word,tag) in pos_tags] #simplified tagging
    lemmas = set(word if tag is None else lemmatize.lemmatize(word,tag) for word,tag in pos_retagged) #extract lemmatized words
    lemmas = set(word.replace("_"," ") for word in lemmas) #convert phrases back to using space instead of underscore
    synsets = wordnet.synsets(word) #find words with similar meanings
    all_lemmas = set(chain.from_iterable([synset.lemma_names() for synset in synsets]))
    all_lemmas = set(lemma.replace("_"," ") for lemma in all_lemmas)#convert phrases back to using space instead of underscore
    return set(lemmas), all_lemmas
"""note for later, include lemmas for the separate words in phrases as well"""

#extract a certain word form a string
def word_compiler(word):
    wordre = re.compile(fr'\b{word}\b',re.IGNORECASE)
    return wordre

def word_extracter_index(stringlist, word):
    wordre = word_compiler(word)
    stringswithkeyword = list((i,wordre.findall(stringlist[i]))  if isinstance(stringlist[i],str) 
                              else (i,[]) for i in range(len(stringlist)))
    stringswithkeyword = list((index,word) for (index,word) in stringswithkeyword if word != [])
    stringswithkeyword = list(index for (index,word) in stringswithkeyword)
    return stringswithkeyword


# In[6]:


#Hashtag remover
def remove_hashtag_in_list_of_strings(lst):
    return [item.lstrip('#') for item in lst]

def most_popular_hashtags_by_topic(df,column,keyphrase):
    topic = keyphrase
    #topic_stem = stem.stem(topic)
    topic_tokens, topic_synonyms = fetch_lemmas(topic)
    allsearchwords = topic_tokens | topic_synonyms
    #search through column for rows with these words inside
    stringswithkeywords = [word_extracter_index(df[column],word) 
                           for word in allsearchwords]#find indices for all words
    indexlist = sorted(list(set([item 
                                           for sublist in stringswithkeywords for item in sublist
                                          ])))#extrapolate all lists to a single list and remove duplicates
    #search for all hashtags in those rows
    relevantrows = df.iloc[indexlist].copy()
    relevantrows = relevantrows.reset_index(drop=True)
    #extract hashtags
    relevantrows['Hashtags'] = extract_hashtags(relevantrows,column)
    #drop rows that do not have hashtags
    relevantrows = relevantrows[relevantrows.Hashtags.apply(len) > 0]
    #remove hashtags for easier handling
    #relevantrows.Hashtags = relevantrows.Hashtags.apply(remove_hashtag_in_list_of_strings)
    #lemmatize hashtags
    #df.Hashtags = df.Hashtags.apply(lambda x: [])
    #rank hashtags
    hashtaglist = [item for sublist in relevantrows.Hashtags for item in sublist] #flatten list
    counter = Counter(hashtaglist) #count occurences of hashtags
    hashtagdf = pd.DataFrame(counter.items(),columns=['Hashtags','Number_Of_Occurences'])#create df with hashtags and no of occurences
    hashtagdf_sorted = hashtagdf.sort_values('Number_Of_Occurences', ascending=False).reset_index(drop=True) #sort by number of occurences
    #remove empty hashtags
    hashtagdf_sorted = hashtagdf_sorted[hashtagdf_sorted["Hashtags"] != "#"].reset_index(drop=True)
    return hashtagdf_sorted

if len(sys.argv) < 2:
    print("Usage: python script.py keyphrase")
    sys.exit(1)

# The first argument after the script name is the keyphrase
phrase = sys.argv[1]

output = most_popular_hashtags_by_topic(data,'Captions',phrase)
print(output.to_json(orient="records"))


