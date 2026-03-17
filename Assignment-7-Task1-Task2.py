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




# Build dictionary from topWords from Task 1
# Make sure Task 1 also uses:
# topWords = allCounts.top(VOCAB_SIZE, key=lambda x: x[1])
dictionaryMap = {word: i for i, (word, count) in enumerate(topWords)}
dictionaryBC = sc.broadcast(dictionaryMap)

# (docID, pos) for every word in every document that is in our vocabulary
justDocAndPos = keyAndListOfWords.flatMap(
    lambda x: ((x[0], dictionaryBC.value[w]) for w in x[1] if w in dictionaryBC.value)
)

# Count occurrences of each vocabulary position inside each document
# ((docID, pos), count)
docPosCounts = justDocAndPos.map(
    lambda x: ((x[0], x[1]), 1)
).reduceByKey(
    lambda a, b: a + b
).persist()

# Rearrange to (docID, (pos, count))
docWordCounts = docPosCounts.map(
    lambda x: (x[0][0], (x[0][1], x[1]))
)

# ---- MAIN PERFORMANCE FIX ----
# Instead of aggregateByKey(np.zeros(VOCAB_SIZE), ...),
# aggregate sparse dictionaries and normalize after.
def seqOp(d, pc):
    pos, cnt = pc
    d[pos] = cnt
    return d

def combOp(d1, d2):
    d1.update(d2)
    return d1

# (docID, {pos: raw_count, ...})
docCountDicts = docWordCounts.aggregateByKey({}, seqOp, combOp)

# Normalize to TF: (docID, {pos: tf, ...})
docTfSparse = docCountDicts.mapValues(
    lambda d: {pos: cnt / float(sum(d.values())) for pos, cnt in d.items()} if len(d) > 0 else d
)

print(docTfSparse.take(3))

# Compute DF directly from unique (docID, pos) pairs
# (pos, document_frequency)
dfCounts = docPosCounts.map(
    lambda x: (x[0][1], 1)
).reduceByKey(
    lambda a, b: a + b
).collect()

dfArray = np.ones(VOCAB_SIZE)
for pos, df in dfCounts:
    dfArray[pos] = df

# IDF(w) = log(numberOfDocs / df(w))
idfArray = np.log(np.divide(np.full(VOCAB_SIZE, numberOfDocs), dfArray))

# Build sparse TF-IDF vectors: (docID, {pos: tfidf, ...})
pageTfidfRDD = docTfSparse.mapValues(
    lambda d: {pos: tf * idfArray[pos] for pos, tf in d.items()}
).persist()

print("tfidf docs:", pageTfidfRDD.count())

# Keep only categories for pages that exist in our corpus
corpusDocIDs = set(pageTfidfRDD.map(lambda x: x[0]).collect())
corpusDocIDsBroadcast = sc.broadcast(corpusDocIDs)

wikiCatsFiltered = wikiCats.filter(
    lambda x: x[0] in corpusDocIDsBroadcast.value
).persist()

print("wikiCatsFiltered count:", wikiCatsFiltered.count())

# Precompute pageID -> list of categories once
pageCatsMap = wikiCatsFiltered.groupByKey().mapValues(list).collectAsMap()
pageCatsBC = sc.broadcast(pageCatsMap)

# kNN prediction using sparse query and sparse page vectors
def getPrediction(textInput, k):
    words = regex.sub(' ', textInput).lower().split()
    positions = [dictionaryBC.value[w] for w in words if w in dictionaryBC.value]

    queryCounts = Counter(positions)
    total = sum(queryCounts.values())

    if total == 0:
        return []

    # Sparse query TF-IDF: {pos: tfidf_value}
    querySparse = {
        pos: (cnt / float(total)) * idfArray[pos]
        for pos, cnt in queryCounts.items()
    }

    # Score each page using only nonzero query positions
    pageDistances = pageTfidfRDD.map(
        lambda x: (
            x[0],
            sum(x[1].get(pos, 0.0) * qval for pos, qval in querySparse.items())
        )
    )

    topKPages = pageDistances.top(k, key=lambda x: x[1])

    # Vote over categories of the top-k pages
    categoryCounts = Counter()
    for page_id, _score in topKPages:
        for cat in pageCatsBC.value.get(page_id, []):
            categoryCounts[cat] += 1

    return categoryCounts.most_common(k)


print(getPrediction('Sport Basketball Volleyball Soccer', 10))
print(getPrediction('What is the capital city of Australia?', 10))
print(getPrediction('How many goals Vancouver score last year?', 10))
