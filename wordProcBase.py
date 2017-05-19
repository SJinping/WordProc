# _*_ coding: utf-8 _*_
import os, glob
import sys
import csv
import nltk
import operator
import random
import string
import numpy as np
from types import *
import platform
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize, sent_tokenize, WhitespaceTokenizer, TweetTokenizer

from progressbar import Bar, ETA, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer


_MAIN_DIR_ = ''
_system_ = platform.system()
if _system_ == 'Darwin':
	_MAIN_DIR_ = '/Users/Pan/Idealab'
elif _system_ == 'Linux':
	_MAIN_DIR_ = '/home/pan/Idealab'



VD_norms           = _MAIN_DIR_ + "/Data/Corpus/VAD_norms/Ratings_Warriner_et_al.csv"
fb_post_file       = _MAIN_DIR_ + "/Data/VA_Proc/emtion_tweets/FB_valence_arousal/dataset-fb-valence-arousal-anon.csv"
Pronouncing_stress = _MAIN_DIR_ + "/Data/Corpus/pronouncing/cmudict-0.7b"
Pronouncing        = _MAIN_DIR_ + "/Data/Corpus/pronouncing/cmudict_SPHINX_40"
phones             = _MAIN_DIR_ + "/Data/Corpus/pronouncing/cmudict-0.7b.phones"

def get_pronounce_withStress_dict(infile = Pronouncing_stress):
	pronounce_dict = {}
	f = open(infile, 'rb')
	line = f.readline()
	while line:
		if line.find(';;;', 0) >= 0:
			line = f.readline()
			continue
		word, pronounce = line.split("  ", 1)
		pronounce_dict[word.lower()] = pronounce
		line = f.readline()
	f.close()
	return pronounce_dict

def get_pronounce_withoutStress_dict(infile = Pronouncing):
	pronounce_dict = {}
	f = open(infile, 'rb')
	line = f.readline()
	while line:
		word, pronounce = line.split("\t", 1)
		pronounce_dict[word.lower()] = pronounce
		line = f.readline()
	f.close()
	return pronounce_dict

def get_pronounce_phones(infile = phones):
	phones_dict = {}
	f = open(infile, 'rb')
	line = f.readline()
	while line:
		phoneme, label       = line.split('\t', 1)
		phones_dict[phoneme] = label.strip()
		line                 = f.readline()
	f.close()
	return phones_dict

# if data is a dict, it will be split by keys
# This function can be repalced by sklearn function train_test_split
def split_data_random(data, ratio):
	d_type = type(data)
	if d_type == ListType:
		index = 0
		data_1 = []
		data_2 = []
		sample = random.sample(xrange(len(data)), int(len(data)*ratio))
		for s in sample:
			data_1.append(data[s])
		data_2 = list(set(data) - set(data_1))
		return data_1, data_2

	elif d_type == DictType:
		data_1 = {}
		data_2 = {}
		keys   = data.keys()
		sample = random.sample(xrange(len(keys)), int(len(keys)*ratio))
		keys_1 = [keys[i] for i in sample]
		keys_2 = list(set(keys) - set(keys_1))
		for key in keys_1:
			data_1[key] = data[key]
		for key in keys_2:
			data_2[key] = data[key]
		return data_1, data_2




def tokenize(text):
	tknzr = WhitespaceTokenizer()
	tokens = tknzr.tokenize(text)
	# tokens = nltk.word_tokenize(text)
	return tokens

# word tokenized
def tokenize2(text):
	tokens = word_tokenize(text)
	return tokens

# word tokenized
# noun lemmatized
# verb lemmatized
def tokenize3(text):
	wordnet_lemmatizer = WordNetLemmatizer()
	tokens             = word_tokenize(text)
	tokens             = [wordnet_lemmatizer.lemmatize(token, NOUN) for token in tokens]
	tokens             = [wordnet_lemmatizer.lemmatize(token, VERB) for token in tokens]
	tokens             = [wordnet_lemmatizer.lemmatize(token, ADJ) for token in tokens]
	return tokens

# word tokenized
# formal English words
# noun lemmatized
# verb lemmatized
def tokenize4(text):
	wordnet_lemmatizer = WordNetLemmatizer()
	tokens             = word_tokenize(text)
	wordset            = set(words.words())
	tokens             = [wordnet_lemmatizer.lemmatize(token, NOUN) for token in tokens]
	tokens             = [wordnet_lemmatizer.lemmatize(token, VERB) for token in tokens]
	tokens             = [wordnet_lemmatizer.lemmatize(token, ADJ) for token in tokens]
	tokens             = [token for token in tokens if token in wordset]
	return tokens

# same to tokenize3, but remove punctuation with string.punctuation
# remove punctuation
# word tokenized
# noun lemmatized
# verb lemmatized
def tokenize5(text):
	wordnet_lemmatizer = WordNetLemmatizer()
	translate_table = dict((ord(char), None) for char in string.punctuation)
	if type(text) == str:
		tokens = word_tokenize(text.translate(None, string.punctuation)) # remove punctuation
		tokens = [wordnet_lemmatizer.lemmatize(token, NOUN) for token in tokens]
		tokens = [wordnet_lemmatizer.lemmatize(token, VERB) for token in tokens]
		tokens = [wordnet_lemmatizer.lemmatize(token, ADJ) for token in tokens]
		return tokens
	elif type(text) == unicode:
		tokens = word_tokenize(text.translate(translate_table))
		tokens = [wordnet_lemmatizer.lemmatize(token, NOUN) for token in tokens]
		tokens = [wordnet_lemmatizer.lemmatize(token, VERB) for token in tokens]
		tokens = [wordnet_lemmatizer.lemmatize(token, ADJ) for token in tokens]
		return tokens

# tokenize tweet
# reduce_len: Replace repeated character sequences of length 3 or greater with sequences of length 3
# e.g. waaaaayyyy will be replaced by waaayyy
# return a list
def tokenize_tweet(text):
	tknzr = TweetTokenizer(reduce_len = True)
	tokens = tknzr.tokenize(text)
	return tokens


# Only process the words that appear in English word dictionary
# Stopwords removing
# Nouns lemmatized
# Verb lemmatized
# return an unsorted word dict
def word_count(infile):
	if infile == '':
		return

	wordDict = {}
	wordnet_lemmatizer = WordNetLemmatizer()
	wordset = set(words.words())
	fread = open(infile, 'rb')
	line = fread.readline()
	index = 0
	while line:
		line = line.decode('utf-8')
		wordlist = [word for word in word_tokenize(line) if not word in nltk.corpus.stopwords.words('english')]
		wordlist = [word for word in wordlist if word in wordset]
		wordlist = [wordnet_lemmatizer.lemmatize(word, NOUN) for word in wordlist]
		wordlist = [wordnet_lemmatizer.lemmatize(word, VERB) for word in wordlist]
		for word in wordlist:
			wordDict[word] = wordDict.get(word, 0) + 1
			sys.stdout.write('Processing line number: ' + str(index) + '\r')
		index += 1
		line = fread.readline()
	wordfreq = sorted(wordDict.items(), key=operator.itemgetter(1), reverse = True) # sorting
	
	return wordDict


def get_fb_post_dict(filename = fb_post_file):
	post_dict = {}
	with open(filename, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
		header = spamreader.next()

		for row in spamreader:
			post_dict[row[0]] = dict()
			post_dict[row[0]]['text'] = row[1]
			post_dict[row[0]]['Valence1'] = row[2]
			post_dict[row[0]]['Valence2'] = row[3]
			post_dict[row[0]]['Arousal1'] = row[4]
			post_dict[row[0]]['Arousal2'] = row[5]

	return post_dict


def get_VA_word_dict(filename = VD_norms):
	print ("Processing VA words...")
	word_dict = {}
	widgets = [FormatLabel('Processed: %(value)d records (in: %(elapsed)s)')]
	pbar = ProgressBar(widgets = widgets)
	with open(filename, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
		headers = next(spamreader)

		# for row in spamreader:
		for row in pbar((row for row in spamreader)):
			word                = row[1] # Word
			V_value             = row[2] # V.Mean.Sum
			A_value             = row[5] # A.Mean.Sum
			VA_value            = {}
			VA_value['V_value'] = float(V_value)
			VA_value['A_value'] = float(A_value)
			word_dict[word]     = VA_value
			# time.sleep(0.03)
	print ("Finish Processing VA words.")
	pbar.finish()
	return word_dict


def get_rangeVA_word_dict(maxV, minV, maxA, minA, filename = VD_norms):
	if maxV > 9 or minV < 1 or maxA > 9 or minA < 1:
		print ("Values out of range!")
		return 

	print ("Processing VA words...")
	word_dict = {}
	widgets = [FormatLabel('Processed: %(value)d records (in: %(elapsed)s)')]
	pbar = ProgressBar(widgets = widgets)
	with open(filename, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
		headers = next(spamreader)

		# for row in spamreader:
		for row in pbar((row for row in spamreader)):
			word = row[1] # Word
			V_value = row[2] # V.Mean.Sum
			A_value = row[5] # A.Mean.Sum
			if (float(V_value) > maxV or float(V_value) < minV) or (float(A_value) > maxA or float(A_value) < minA):
				continue
			VA_value = {}
			VA_value['V_value'] = float(V_value)
			VA_value['A_value'] = float(A_value)
			word_dict[word] = VA_value
	print ("Finish Processing VA words.")
	return word_dict


# Calculate TF-IDF
# doclist: list, contains dicts, each the key of the dict is the identifier of a text, the value of the dict is the text
# ngram: (1, 2), means use gram or bigram
# return: tfidf_results, the values of tfidf, sorted; vector_to_name: feature names
def get_tf_idf(doclist, tokenizer = tokenize3, use_idf = True, vocabulary = None, ngram = (1,1)):
	vectorizer     = CountVectorizer(tokenizer = tokenizer, ngram_range = ngram, stop_words = 'english', lowercase = True, vocabulary = vocabulary)	# Implements both tokenization and occurrence counting 
	X              = vectorizer.fit_transform(doclist)
	transformer    = TfidfTransformer(use_idf = use_idf)
	tfidf          = transformer.fit_transform(X)
	# tfidf_results  = np.argsort(tfidf.toarray())
	# vector_to_name = vectorizer.get_feature_names()
	# return tfidf_results, vector_to_name
	return tfidf