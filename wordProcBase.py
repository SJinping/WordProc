# _*_ coding: utf-8 _*_
import sys
import csv
import nltk
import operator
import numpy as np
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus.reader.wordnet import VERB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize, sent_tokenize, WhitespaceTokenizer

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer



VD_norms = "/home/pan/Idealab/Data/Corpus/VAD_norms/Ratings_Warriner_et_al.csv"

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
	tokens = word_tokenize(text)
	tokens = [wordnet_lemmatizer.lemmatize(token, NOUN) for token in tokens]
	tokens = [wordnet_lemmatizer.lemmatize(token, VERB) for token in tokens]
	return tokens

# word tokenized
# formal English words
# noun lemmatized
# verb lemmatized
def tokenize4(text):
	wordnet_lemmatizer = WordNetLemmatizer()
	wordset = set(words.words())
	tokens = [token for token in word_tokenize(text) if token in wordset]
	tokens = [wordnet_lemmatizer.lemmatize(token, NOUN) for token in tokens]
	tokens = [wordnet_lemmatizer.lemmatize(token, VERB) for token in tokens]
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


def get_VA_word_dict(filename = VD_norms):
	print "Processing VA words..."
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
			VA_value = {}
			VA_value['V_value'] = float(V_value)
			VA_value['A_value'] = float(A_value)
			word_dict[word] = VA_value
			# time.sleep(0.03)
	print "Finish Processing VA words."
	return word_dict


def get_rangeVA_word_dict(maxV, minV, maxA, minA, filename = VD_norms):
	if maxV > 9 or minV < 1 or maxA > 9 or minA < 1:
		print "Values out of range!"
		return 

	print "Processing VA words..."
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
	print "Finish Processing VA words."
	return word_dict


# Calculate TF-IDF
# doclist: list, contains dicts, each the key of the dict is the identifier of a text, the value of the dict is the text
# ngram: (1, 2), means use gram or bigram
# return: tfidf_results, the values of tfidf, sorted; vector_to_name: feature names
def get_tf_idf(doclist, ngram):
	print "Processing " + str(len(doclist)) + " documents..."
	vectorizer = CountVectorizer(tokenizer=tokenize4, ngram_range=ngram, stop_words='english', lowercase=True)	# Implements both tokenization and occurrence counting 
	X = vectorizer.fit_transform(doclist)
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(X)
	tfidf_results = np.argsort(tfidf.toarray())
	vector_to_name = vectorizer.get_feature_names()
	return tfidf_results, vector_to_name