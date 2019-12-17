#!/usr/bin/env python
# coding: utf-8

import os
import time
import json

# May be necessary on Unix systems. Make sure to uncomment above sklearn imports
# os.environ['OPENBLAS_NUM_THREADS'] = '15'
# os.environ['JOBLIB_START_METHOD'] = 'forkserver'

from sklearn import neural_network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim import models
import numpy as np


def getPosTrainData():
    posTrainFileDir = "./Problem3/training/pos/"
    posTrainFilePaths = []
    for _, _, filenames in os.walk(posTrainFileDir):
        posTrainFilePaths = [posTrainFileDir + path for path in filenames]

    posTrainData = []
    for trainFilePath in posTrainFilePaths:
        trainFile = open(trainFilePath, encoding="utf8", mode="r")

        posTrainData.append(trainFile.read().lower())

        trainFile.close()

    return posTrainData


def getNegTrainData():
    negTrainFileDir = "./Problem3/training/neg/"
    negTrainFilePaths = []
    for _, _, filenames in os.walk(negTrainFileDir):
        negTrainFilePaths = [negTrainFileDir + path for path in filenames]

    negTrainData = []
    for trainFilePath in negTrainFilePaths:
        trainFile = open(trainFilePath, encoding="utf8", mode="r")

        negTrainData.append(trainFile.read().lower())

        trainFile.close()

    return negTrainData


def makeBagOfWords(posTrainData, negTrainData):
    bagOfWords = set()

    for posSentence in posTrainData:
        for posWord in posSentence.split():
            bagOfWords.add(posWord)

    for negSentence in negTrainData:
        for negWord in negSentence.split():
            bagOfWords.add(negWord)

    return bagOfWords


def makeFeatureVectors():
    featureVectors = []
    targetLabels = []

    posTrainData = getPosTrainData()
    negTrainData = getNegTrainData()
    bagOfWords = makeBagOfWords(posTrainData, negTrainData)

    len(posTrainData)
    len(negTrainData)
    len(bagOfWords)

    for posSentence in posTrainData:
        featureVector = []
        sentenceWords = posSentence.split()

        for word in bagOfWords:
            if word in sentenceWords:
                featureVector.append(1)
            else:
                featureVector.append(0)

        featureVectors.append(featureVector)
        targetLabels.append(0)  # pos reviews have target 0

    for negSentence in negTrainData:
        featureVector = []
        sentenceWords = negSentence.split()

        for word in bagOfWords:
            if word in sentenceWords:
                featureVector.append(1)
            else:
                featureVector.append(0)

        featureVectors.append(featureVector)
        targetLabels.append(1)  # neg reviews have target 1

    return featureVectors, targetLabels, bagOfWords


def makeSentenceEmeddings():
    print("Started loading word2vec model")
    w = models.KeyedVectors.load_word2vec_format(
        './Problem3/GoogleNews-vectors-negative300.bin', binary=True)
    print("Finished loading word2vec model")

    posData = getPosTrainData()
    negData = getNegTrainData()

    sentenceEmbeddings = []
    targetLabels = []

    for posSentence in posData:
        wordEmbeddings = []

        for posWord in posSentence.split():
            try:
                wordEmbeddings.append(w[posWord])
            except:
                # could not find word in word2vec embeddings
                pass

        if len(wordEmbeddings) == 0:
            print("Could not find any word embeddings for pos review: " + posSentence)
            continue

        wordEmbeddings = np.array(wordEmbeddings)

        # take average of word vectors to get sentence embedding
        sentenceEmbedding = []
        for index in range(len(wordEmbeddings[0])):
            embeddingValuesAtIndex = wordEmbeddings[:, index]
            avgEmbeddingValueAtIndex = np.average(embeddingValuesAtIndex)
            sentenceEmbedding.append(avgEmbeddingValueAtIndex)

        sentenceEmbeddings.append(sentenceEmbedding)
        targetLabels.append(0)  # pos reviews have target 0

    for negSentence in negData:
        wordEmbeddings = []

        for negWord in negSentence.split():
            try:
                wordEmbeddings.append(w[negWord])
            except:
                # could not find word in word2vec embeddings
                pass

        if len(wordEmbeddings) == 0:
            print("Could not find any word embeddings for neg review: " + negSentence)
            continue

        wordEmbeddings = np.array(wordEmbeddings)

        # take average of word vectors to get sentence embedding
        sentenceEmbedding = []
        for index in range(len(wordEmbeddings[0])):
            embeddingValuesAtIndex = wordEmbeddings[:, index]
            avgEmbeddingValueAtIndex = np.average(embeddingValuesAtIndex)
            sentenceEmbedding.append(avgEmbeddingValueAtIndex)

        sentenceEmbeddings.append(sentenceEmbedding)
        targetLabels.append(1)  # neg reviews have target 1

    return sentenceEmbeddings, targetLabels


def runCrossValidation(featureVectors, targetLabels, activationFunction, firstHiddenLayerSize):
    # assumes first half of featureVectors/targetLabels are for pos and second half are for neg
    midPoint = int(len(targetLabels) / 2)
    posFeatureVectors = featureVectors[:midPoint]
    posTargetLabels = targetLabels[:midPoint]
    negFeatureVectors = featureVectors[midPoint:]
    negTargetLabels = targetLabels[midPoint:]

    crossValidationScores = []

    # 10 folds
    for i in range(10):
        print("i=" + str(i), flush=True)
        scalar = i / 10.0
        nextScalar = (i + 1) / 10.0

        startSlice = int(scalar * len(posFeatureVectors))
        endSlice = int(nextScalar * len(posFeatureVectors))

        holdOutSet = posFeatureVectors[startSlice:endSlice]
        holdOutSet.extend(negFeatureVectors[startSlice:endSlice])

        holdOutSetLabels = [0 for pos in range(int(len(holdOutSet) / 2))]
        holdOutSetLabels.extend([1 for neg in range(int(len(holdOutSet) / 2))])

        trainSet = posFeatureVectors[:startSlice]
        trainSet.extend(posFeatureVectors[endSlice:])
        trainSet.extend(negFeatureVectors[:startSlice])
        trainSet.extend(negFeatureVectors[endSlice:])

        trainSetLabels = [0 for pos in range(int(len(trainSet) / 2))]
        trainSetLabels.extend([1 for neg in range(int(len(trainSet) / 2))])

        # train
        mlp = neural_network.MLPClassifier(activation=activationFunction, hidden_layer_sizes=(firstHiddenLayerSize, 10))
        mlp.fit(trainSet, trainSetLabels)

        # score with accuracy
        score = mlp.score(holdOutSet, holdOutSetLabels)
        crossValidationScores.append(score)

    averageFoldScore = sum(crossValidationScores) / len(crossValidationScores)

    return "Average cross validation score using activation function " + activationFunction + " and " + str(firstHiddenLayerSize) + " nodes in the first hidden layer was " + str(averageFoldScore)


# create feature vectors
# featureVectors, targetLabels = makeFeatureVectors()

# ---------------------------  Start 3.1

# run cross validation
# activationFunctions = ["identity", "logistic", "tanh", "relu"]
# firstHiddenLayerSizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# results = []

# # # test activation functions with first layer nodes = 10
# for func in activationFunctions:
#     print("Starting validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
#     result = runCrossValidation(featureVectors, targetLabels, func, 10)
#     print("Finished validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)

#     print(result)
#     results.append(result)

# # Found 'identity' to be best function. Try it with different # first layer nodes
# bestActivationFunction = 'identity'
# for size in firstHiddenLayerSizes:
#     print("Starting validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
#     result = runCrossValidation(featureVectors, targetLabels, bestActivationFunction, size)
#     print("Finished validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)

#     print(result)
#     results.append(result)


# ---------------------- End 3.1

# --------------------- Start 3.2

# train model on whole training set
# bestActivationFunction = 'identity'
# bestFirstHiddenLayerSize = 1

# mlp = neural_network.MLPClassifier(activation=bestActivationFunction, hidden_layer_sizes=(bestFirstHiddenLayerSize, 10))
# mlp.fit(featureVectors, targetLabels)

# fullTrainingDataScore = mlp.score(featureVectors, targetLabels)
# print(fullTrainingDataScore)


# --------------------- End 3.2


# ------------------- Start 3.3

# sentenceEmbeddings, targetLabels = makeSentenceEmeddings()

# # The following section can be used to export/import the calculated sentence embeddings

# sentenceEmbeddingsFile = open("./Problem3/training/sentenceEmbeddings.txt", mode="w")
# sentenceEmbeddingsFile.write(str(sentenceEmbeddings))
# sentenceEmbeddingsFile.close()

# targetLabelsFile = open("./Problem3/training/targetLabelsFile.txt", mode="w")
# targetLabelsFile.write(str(targetLabels))
# targetLabelsFile.close()

# # read in sentence embeddings that were calculated from word2vec
# sentenceEmbeddingsFile = open("./Problem3/training/sentenceEmbeddings.txt", mode="r")
# sentenceEmbeddingsStr = sentenceEmbeddingsFile.read()
# sentenceEmbeddings = ast.literal_eval(sentenceEmbeddingsStr)
# sentenceEmbeddingsFile.close()

# targetLabelsFile = open("./Problem3/training/targetLabelsFile.txt", mode="r")
# targetLabelsStr = targetLabelsFile.read()
# targetLabels = ast.literal_eval(targetLabelsStr)
# targetLabelsFile.close()

# # do cross-validation
# activationFunctions = ["identity", "logistic", "tanh", "relu"]
# firstHiddenLayerSizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# results = []

# # # test activation functions with first layer nodes = 10
# for func in activationFunctions:
#     print("Starting validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
#     result = runCrossValidation(sentenceEmbeddings, targetLabels, func, 10)
#     print("Finished validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)

#     print(result)
#     results.append(result)

# # Found 'logistic' to be best function. Try it with different # first layer nodes
# bestActivationFunction = 'logistic'
# for size in firstHiddenLayerSizes:
#     print("Starting validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
#     result = runCrossValidation(sentenceEmbeddings, targetLabels, bestActivationFunction, size)
#     print("Finished validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)

#     print(result)
#     results.append(result)

# # Found 8 nodes in the first hidden layer to be optimal. Try on full training set:
# bestFirstHiddenLayerSize = 8

# mlp = neural_network.MLPClassifier(activation=bestActivationFunction, hidden_layer_sizes=(bestFirstHiddenLayerSize, 10))
# mlp.fit(sentenceEmbeddings, targetLabels)

# fullTrainingDataScore = mlp.score(sentenceEmbeddings, targetLabels)
# print(fullTrainingDataScore)

# -------------------- End 3.3

# -------------------- Start 3.4
# data = getPosTrainData()
# data.extend(getNegTrainData())

# vectorizer = TfidfVectorizer()

# tfIdfVectors = vectorizer.fit_transform(data)
# tfIdfVectors = tfIdfVectors.toarray()

# targetLabels = [0 for pos in range(int(len(tfIdfVectors) / 2))]
# targetLabels.extend([1 for neg in range(int(len(tfIdfVectors) / 2))])

# # possible parameters to optimize between
# dimensions = [50, 100, 200, 300]
# activationFunctions = ["identity", "logistic", "tanh", "relu"]
# firstHiddenLayerSizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# results = []

# for numDimensions in dimensions:
#     print("Starting svd with dimensions = " + str(numDimensions) + " at " +
#           str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
#     svd = TruncatedSVD(n_components=numDimensions)
#     svdVectors = svd.fit_transform(tfIdfVectors).tolist()

#     # determine best activation function
#     for activationFunction in activationFunctions:
#         print("Starting validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
#         result = runCrossValidation(svdVectors, targetLabels, activationFunction, 10)
#         print("Finished validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
#         print(result)
#         results.append(result)

# # determined 'relu' to be best activation function with svd dimensions = 300
# bestActivationFunction = 'relu'
# bestDimensionSize = 300

# print("Starting svd with dimensions = " + str(bestDimensionSize) + " at " +
#       str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
# svd = TruncatedSVD(n_components=bestDimensionSize)
# svdVectors = svd.fit_transform(tfIdfVectors).tolist()

# for size in firstHiddenLayerSizes:
#     print("Starting validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)
#     result = runCrossValidation(svdVectors, targetLabels, bestActivationFunction, size)
#     print("Finished validation at " + str(time.strftime("%a, %d %b %Y %H:%M:%S")), flush=True)

#     print(result)
#     results.append(result)

# # determined best first hidden layer size is 7
# bestFirstHiddenLayerSize = 7

# # test on whole training set
# svd = TruncatedSVD(n_components=bestDimensionSize)
# svdVectors = svd.fit_transform(tfIdfVectors).tolist()

# mlp = neural_network.MLPClassifier(activation=bestActivationFunction, hidden_layer_sizes=(bestFirstHiddenLayerSize, 10))
# mlp.fit(svdVectors, targetLabels)

# fullTrainingDataScore = mlp.score(svdVectors, targetLabels)
# print(fullTrainingDataScore)

# ---------------- End 3.4

# ------------ Start 3.5
# data = getPosTrainData()
# data.extend(getNegTrainData())

# vectorizer = TfidfVectorizer()

# tfIdfVectors = vectorizer.fit_transform(data)

# bestDimensionSize = 300
# svd = TruncatedSVD(n_components=bestDimensionSize)
# svd.fit_transform(tfIdfVectors)

# featureNames = vectorizer.get_feature_names()

# for index, topic in enumerate(svd.components_[:5]):
#     # for given topic,
#     indexedTopicValues = [(i, val) for i, val in enumerate(topic)]
#     indexedTopicValues.sort(key=lambda pair: pair[1], reverse=True)

#     # get top 20 words
#     top20WordsIndex = [pair[0] for pair in indexedTopicValues[:20]]
#     top20Words = [featureNames[index] for index in top20WordsIndex]
#     print(top20Words)

# --------------- End 3.5

# ------------- Start 3.6
# # Using bag of words with MLP
# featureVectors, targetLabels, bagOfWords = makeFeatureVectors()

# bestActivationFunction = 'identity'
# bestFirstHiddenLayerSize = 1

# # train on training data
# mlp = neural_network.MLPClassifier(activation=bestActivationFunction, hidden_layer_sizes=(bestFirstHiddenLayerSize, 10))
# mlp.fit(featureVectors, targetLabels)

# # collect test data
# testFileDir = "./Problem3/test/"
# testFilePaths = []
# for _, _, filenames in os.walk(testFileDir):
#     testFilePaths = [testFileDir + path for path in filenames]

# testData = []
# for testFilePath in testFilePaths:
#     testFile = open(testFilePath, encoding="utf8", mode="r")

#     testData.append(testFile.read().lower())

#     testFile.close()

# testFeatureVectors = []
# for doc in testData:
#     featureVector = []
#     sentenceWords = doc.split()

#     for word in bagOfWords:
#         if word in sentenceWords:
#             featureVector.append(1)
#         else:
#             featureVector.append(0)

#     testFeatureVectors.append(featureVector)

# # predict test data
# predictions = mlp.predict(testFeatureVectors)

# posOutputFile = open("./Problem3/labels/pos.txt", mode="w")
# negOutputFile = open("./Problem3/labels/neg.txt", mode="w")

# for index, file in enumerate(testFilePaths):
#     fileName = file.split("/")[-1].split(".")[0]

#     prediction = predictions[index]

#     if prediction == 0:
#         posOutputFile.write(fileName + "\n")
#     elif prediction == 1:
#         negOutputFile.write(fileName + "\n")
#     else:
#         print("Got unexpected predicted value")

# posOutputFile.close()
# negOutputFile.close()

# ---------------- End 3.6
