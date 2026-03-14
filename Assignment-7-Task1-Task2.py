#!/usr/bin/env python
# coding: utf-8

import sys
import os
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm

# point Spark's JVM worker processes to the same python binary running this script
# on windows, spark tries to run "python" which hits MS Store alias
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession

builder = SparkSession.builder.appName("WikipediaKNN")
# on Windows my hostname has  underscores (invalid in Spark's RPC URLs),
# so force localhost. On Linux/cloud VMs this is not needed and would break cluster mode.
if os.name == 'nt':
    builder = builder.config("spark.driver.host", "localhost")
spark = builder.getOrCreate()
sc = spark.sparkContext

# spark-submit assignment7.py <wikiPagesFile> <wikiCategoryFile>
wikiPagesFile = sys.argv[1] if len(sys.argv) > 1 else "WikipediaPagesOneDocPerLine1000LinesSmall.txt.bz2"
wikiCategoryFile = sys.argv[2] if len(sys.argv) > 2 else "wiki-categorylinks-small.csv.bz2"

wikiCategoryLinks = sc.textFile(wikiCategoryFile)
wikiCats = wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))

wikiPages = sc.textFile(wikiPagesFile)
numberOfDocs = wikiPages.count()
print(numberOfDocs)

validLines = wikiPages.filter(lambda x : 'id' in x and 'url=' in x)
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))

regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


# Builds a normalized term-frequency array of size 20000.
# Uses np.add to count repeated indices
def buildArray(listOfIndices):
    returnVal = np.zeros(20000)
    indicesArray = np.array(list(listOfIndices), dtype=np.int32)
    np.add.at(returnVal, indicesArray, 1)
    mysum = np.sum(returnVal)
    returnVal = np.divide(returnVal, mysum)
    return returnVal


################### TASK 1 ###################

# Emit (word, 1) per occurrence, then sum to get corpus-wide word counts
allWords = keyAndListOfWords.flatMap(lambda x: [(w, 1) for w in x[1]])

# reduceByKey does map-side combining, reducing shuffle vs groupByKey
allCounts = allWords.reduceByKey(lambda a, b: a + b)

# Pull top-20K words to driver as a Python list sorted descending by count
topWords = allCounts.top(20000, key=lambda x: x[1])

print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

# dictionary RDD: (word, rank_position) where position 0 = most frequent
topWordsK = sc.parallelize(range(20000))
dictionary = topWordsK.map(lambda x : (topWords[x][0], x))

print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x : x[1]))





################### TASK 2 ###################

# (word, docID) for every word in every document
allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# join on word: (word, position) x (word, docID) -> (word, (position, docID))
allDictionaryWords = dictionary.join(allWordsWithDocID)

# drop the word key -> (docID, position)
justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))

# group positions by doc -> (docID, [pos1, pos2, ...])
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

# build TF array per document
allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))
print(allDocsAsNumpyArrays.take(3))

# binary array - 1 if word appears in doc, 0 otherwise
zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], np.where(x[1] > 0, 1, 0)))

# dfArray[i] = number of documents containing word i
dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

# IDF(w) = log(corpus_size / document_frequency(w))
idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray))

# TF-IDF vector per document
allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
print(allDocsAsNumpyArraysTFidf.take(2))

wikiCats.take(1)

# Build featuresRDD: (category, tfidf_array) for each page in our corpus.
#
# PROBLEM with naive wikiCats.join(allDocsAsNumpyArraysTFidf):
#   wikiCats covers all of Wikipedia — tens of millions of (pageID, category) rows
#   Shuffling all of wikicats is hella work
#
# SOLUTION — broadcast-filter-then-join:
#   1. collect only corpus docID strings to the driver as python set
#   2. broadcast that to all executors
#   3. filter wikiCats to only pages in our corpus
#   4. join the filtered wikiCats with allDocsAsNumpyArraysTFidf.

corpusDocIDs = set(allDocsAsNumpyArraysTFidf.map(lambda x: x[0]).collect())
corpusDocIDsBroadcast = sc.broadcast(corpusDocIDs)

# Map-side filter (no shuffle)
wikiCatsFiltered = wikiCats.filter(lambda x: x[0] in corpusDocIDsBroadcast.value)

# (pageID, category) join (pageID, tfidf_array) -> (category, tfidf_array)
featuresRDD = wikiCatsFiltered.join(allDocsAsNumpyArraysTFidf).map(
    lambda x: (x[1][0], x[1][1])
)

featuresRDD.cache()
featuresRDD.take(10)
print("featuresRDD count:", featuresRDD.count())


# kNN prediction: returns the top predicted categories for a text input
def getPrediction(textInput, k):
    myDoc = sc.parallelize([textInput])
    wordsInThatDoc = myDoc.flatMap(lambda x : ((j, 1) for j in regex.sub(' ', x).lower().split()))

    # join against dictionary, group all positions into one list keyed by 1
    allDictionaryWordsInThatDoc = dictionary.join(wordsInThatDoc).map(lambda x: (x[1][1], x[1][0])).groupByKey()

    myArray = buildArray(allDictionaryWordsInThatDoc.top(1)[0][1])
    myArray = np.multiply(myArray, idfArray)

    # cosine similarity between query and each (category, tfidf_array)
    distances = featuresRDD.map(lambda x : (x[0], np.dot(x[1], myArray)))
    topK = distances.top(k, lambda x : x[1])

    # count vote over top-k categories
    docIDRepresented = sc.parallelize(topK).map(lambda x : (x[0], 1))
    numTimes = docIDRepresented.reduceByKey(lambda a, b: a + b)
    return numTimes.top(k, lambda x: x[1])


print(getPrediction('Sport Basketball Volleyball Soccer', 10))
print(getPrediction('What is the capital city of Australia?', 10))
print(getPrediction('How many goals did Vancouver score last year?', 10))
