#!/usr/bin/env python
# coding: utf-8

import sys
import os
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm
from operator import add

VOCAB_SIZE = 5000

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


# Builds a normalized term-frequency array of size 5000.
# Uses np.add to count repeated indices
def buildArray(listOfIndices):
    returnVal = np.zeros(VOCAB_SIZE)
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
topWords = allCounts.top(VOCAB_SIZE, key=lambda x: x[1])

print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

# dictionary RDD: (word, rank_position) where position 0 = most frequent
topWordsK = sc.parallelize(range(VOCAB_SIZE))
dictionary = topWordsK.map(lambda x : (topWords[x][0], x))

print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x : x[1]))





################### TASK 2 ###################



dictionaryMap = {topWords[i][0]: i for i in range(len(topWords))}
dictionaryBC = sc.broadcast(dictionaryMap)


justDocAndPos = keyAndListOfWords.flatMap(
    lambda x: [(x[0], dictionaryBC.value[w]) for w in x[1] if w in dictionaryBC.value]
)


docPosCounts = justDocAndPos.map(
    lambda x: ((x[0], x[1]), 1)
).reduceByKey(
    lambda a, b: a + b
)

docWordCounts = docPosCounts.map(
    lambda x: (x[0][0], (x[0][1], x[1]))
)

# Build normalized TF vectors
def seqOp(arr, pc):
    pos, cnt = pc
    arr[pos] = cnt
    return arr


allDocsAsNumpyArrays = (
    docWordCounts.aggregateByKey(np.zeros(VOCAB_SIZE), seqOp, add)
    .mapValues(lambda arr: arr / arr.sum() if arr.sum() > 0 else arr)
)

# print(allDocsAsNumpyArrays.take(3))

# Compute document frequency directly from unique (docID, pos) pairs
# Each ((docID, pos), count) means that word-position pos appears in that doc at least once
dfCounts = docPosCounts.map(
    lambda x: (x[0][1], 1)
).reduceByKey(
    lambda a, b: a + b
).collect()

dfArray = np.ones(VOCAB_SIZE)
for pos, df in dfCounts:
    dfArray[pos] = df

# Inverse document frequency
idfArray = np.log(np.divide(np.full(VOCAB_SIZE, numberOfDocs), dfArray))

# TF-IDF vectors for pages
pageTfidfRDD = allDocsAsNumpyArrays.map(
    lambda x: (x[0], np.multiply(x[1], idfArray))
).persist()

print("tfidf docs:", pageTfidfRDD.count())

# Keep only category rows whose pageID is in our corpus
corpusDocIDs = set(pageTfidfRDD.map(lambda x: x[0]).collect())
corpusDocIDsBroadcast = sc.broadcast(corpusDocIDs)

wikiCatsFiltered = wikiCats.filter(
    lambda x: x[0] in corpusDocIDsBroadcast.value
).persist()

print("wikiCatsFiltered count:", wikiCatsFiltered.count())


# kNN prediction:
# 1. compare query against page TF-IDF vectors
# 2. take top-k pages
# 3. vote using categories attached to those pages
def getPrediction(textInput, k):
    words = regex.sub(' ', textInput).lower().split()
    positions = [dictionaryBC.value[w] for w in words if w in dictionaryBC.value]

    myArray = np.zeros(VOCAB_SIZE)
    if len(positions) > 0:
        np.add.at(myArray, positions, 1)
        mysum = np.sum(myArray)
        if mysum > 0:
            myArray = myArray / mysum

    myArray = np.multiply(myArray, idfArray)

    # similarity to each page
    distances = pageTfidfRDD.map(lambda x: (x[0], np.dot(x[1], myArray)))

    # top-k nearest pages
    topKPages = distances.top(k, key=lambda x: x[1])

    # get page ids of top-k pages
    topKPageIDs = set([x[0] for x in topKPages])
    topKPageIDsBroadcast = sc.broadcast(topKPageIDs)

    # vote over categories attached to those pages
    numTimes = (
        wikiCatsFiltered
        .filter(lambda x: x[0] in topKPageIDsBroadcast.value)
        .map(lambda x: (x[1], 1))
        .reduceByKey(lambda a, b: a + b)
    )

    return numTimes.top(k, key=lambda x: x[1])


print(getPrediction('Sport Basketball Volleyball Soccer', 10))
print(getPrediction('What is the capital city of Australia?', 10))
print(getPrediction('How many goals Vancouver score last year?', 10))

