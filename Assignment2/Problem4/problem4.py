import json
import spacy
import readability
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from gensim import models
import numpy as np

nlp = spacy.load("en_core_web_sm")

import neuralcoref
neuralcoref.add_to_pipe(nlp)


def getGrammaticalityFeatureVectors(summariesWithGrammarScore, useExtraFeatures=False):
    featureVectors = []
    targetLabels = []

    for summaryWithGrammarScore in summariesWithGrammarScore:
        summary = summaryWithGrammarScore[0]
        docGrammaticality = summaryWithGrammarScore[1]

        summaryDoc = nlp(summary)

        unigrams = [term.text for term in summaryDoc if not term.is_space]
        repeatedUnigrams = set([term for term in unigrams if unigrams.count(term) > 1])
        wordLengths = [len(word) for word in unigrams]

        bigrams = []
        for i in range(len(unigrams) - 1):
            word1 = unigrams[i]
            word2 = unigrams[i + 1]
            bigrams.append(word1 + "_" + word2)
        repeatedBigrams = set([bigram for bigram in bigrams if bigrams.count(bigram) > 1])

        readabilityScores = []
        sentenceLengths = []
        for sent in summaryDoc.sents:
            sentenceLengths.append(len(sent))
            try:
                readabilityScore = readability.getmeasures(sent.text, lang='en')['readability grades']['FleschReadingEase']
                readabilityScores.append(readabilityScore)
            except:
                # Not a real sentence
                pass

        minReadabilityScore = min(readabilityScores)
        avgWordLength = sum(wordLengths) / len(wordLengths)
        avgSentenceLength = sum(sentenceLengths) / len(sentenceLengths)

        featureVector = [len(repeatedUnigrams), len(repeatedBigrams), minReadabilityScore]
        if useExtraFeatures:
            featureVector.append(avgWordLength)
            featureVector.append(avgSentenceLength)

        featureVectors.append(featureVector)
        targetLabels.append(docGrammaticality)

    return featureVectors, targetLabels


def getNonRedundancyFeatureVectors(summariesWithNonRedundancyScore, word2vec, useExtraFeatures=False):
    featureVectors = []
    targetLabels = []

    for summaryWithNonRedundancyScore in summariesWithNonRedundancyScore:
        summary = summaryWithNonRedundancyScore[0]
        docNonRedundancy = summaryWithNonRedundancyScore[1]

        summaryDoc = nlp(summary)

        unigramFrequency = {}
        bigramFrequency = {}
        for i, term in enumerate(summaryDoc):
            word = term.text

            # handle bigrams
            if i < len(summaryDoc) - 1:
                word2 = summaryDoc[i + 1].text
                bigram = word + "_" + word2

                if bigram in bigramFrequency:
                    bigramFrequency[bigram] = bigramFrequency[bigram] + 1
                else:
                    bigramFrequency[bigram] = 1

            # handle unigrams
            if term.is_stop:
                continue

            if word in unigramFrequency:
                unigramFrequency[word] = unigramFrequency[word] + 1
            else:
                unigramFrequency[word] = 1

        maxUnigramFrequency = max(unigramFrequency.items(), key=lambda pair: pair[1])[1]
        maxBigramFrequency = max(bigramFrequency.items(), key=lambda pair: pair[1])[1]

        sentenceEmbeddings = []
        for sent in summaryDoc.sents:
            wordEmbeddings = []
            for word in sent.text.split():
                try:
                    wordEmbeddings.append(word2vec[word])
                except:
                    # could not find word in word2vec embeddings
                    pass

            if len(wordEmbeddings) == 0:
                print("Could not find any word embeddings for sentence: " + sent.text)
                continue

            wordEmbeddings = np.array(wordEmbeddings)

            # take average of word vectors to get sentence embedding
            sentenceEmbedding = []
            for index in range(len(wordEmbeddings[0])):
                embeddingValuesAtIndex = wordEmbeddings[:, index]
                avgEmbeddingValueAtIndex = np.average(embeddingValuesAtIndex)
                sentenceEmbedding.append(avgEmbeddingValueAtIndex)

            sentenceEmbeddings.append(sentenceEmbedding)

        maxSimilarity = 0
        if len(sentenceEmbeddings) > 1:
            sentenceEmeddingsCosineSimilarities = cosine_similarity(sentenceEmbeddings)

            for i, row in enumerate(sentenceEmeddingsCosineSimilarities.tolist()):
                del row[i]  # diagnol is self-similarity, remove
                maxSimilarity = max(row) if max(row) > maxSimilarity else maxSimilarity

        featureVector = [maxUnigramFrequency, maxBigramFrequency, maxSimilarity]

        if useExtraFeatures:
            # count named entities
            namedEntityFrequency = len(summaryDoc.ents) / len(summaryDoc)
            textEnts = [ent.text for ent in summaryDoc.ents]
            repeatedNamedEntities = set([ent for ent in textEnts if textEnts.count(ent) > 1])

            featureVector.append(namedEntityFrequency)
            featureVector.append(len(repeatedNamedEntities))

        featureVectors.append(featureVector)
        targetLabels.append(docNonRedundancy)

    return featureVectors, targetLabels


def getCoherenceFeatureVectors(summariesWithCoherenceScores):
    featureVectors = []
    targetLabels = []

    for summaryWithCoherenceScore in summariesWithCoherenceScores:
        summary = summaryWithCoherenceScore[0]
        docCoherence = summaryWithCoherenceScore[1]

        summaryDoc = nlp(summary)

        chunkFrequency = {}
        for chunk in summaryDoc.noun_chunks:
            chunk = chunk.text

            if chunk in chunkFrequency:
                chunkFrequency[chunk] = chunkFrequency[chunk] + 1
            else:
                chunkFrequency[chunk] = 1

        repeatedChunks = 0
        for _, freq in chunkFrequency.items():
            if freq > 1:
                repeatedChunks += 1

        totalCoreferredEntities = len(summaryDoc._.coref_clusters)

        featureVector = [repeatedChunks, totalCoreferredEntities]

        featureVectors.append(featureVector)
        targetLabels.append(docCoherence)

    return featureVectors, targetLabels

# ----------- Start 4.1

# # read train data
# with open("./summary_quality/train_data.json", 'r') as fin:
#     trainMap = json.load(fin)

# trainSummariesWithGrammarScore = []
# for fileName in trainMap.keys():
#     docGrammaticality = trainMap[fileName]['grammaticality']

#     with open("./summary_quality/summaries/" + fileName, encoding="latin-1") as summaryFile:
#         summary = summaryFile.read().lower()

#     trainSummariesWithGrammarScore.append((summary, int(docGrammaticality)))

# # read test data
# with open("./summary_quality/test_data.json", 'r') as fin:
#     testMap = json.load(fin)

# testSummariesWithGrammarScore = []
# for fileName in testMap.keys():
#     docGrammaticality = testMap[fileName]['grammaticality']

#     with open("./summary_quality/summaries/" + fileName, encoding="latin-1") as summaryFile:
#         summary = summaryFile.read().lower()

#     testSummariesWithGrammarScore.append((summary, int(docGrammaticality)))

# # train model
# # Note: set useExtraFeatures to False to remove added features
# trainFeatureVectors, trainTargetLabels = getGrammaticalityFeatureVectors(trainSummariesWithGrammarScore, useExtraFeatures=True)
# model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=350)
# model.fit(trainFeatureVectors, trainTargetLabels)

# # test model
# # Note: set useExtraFeatures to False to remove added features
# testFeatureVectors, testTargetLabels = getGrammaticalityFeatureVectors(testSummariesWithGrammarScore, useExtraFeatures=True)
# testDataPredictions = model.predict(testFeatureVectors)

# # evaluate
# mse = mean_squared_error(testTargetLabels, testDataPredictions)
# pearson, _ = pearsonr(testTargetLabels, testDataPredictions)
# print(mse)
# print(pearson)

# -------------- End 4.1

# -------------- Start 4.2


# # read train data
# with open("./Problem4/summary_quality/train_data.json", 'r') as fin:
#     trainMap = json.load(fin)

# trainSummariesWithNonRedundancyScore = []
# for fileName in trainMap.keys():
#     docNonRedundancy = trainMap[fileName]['nonredundancy']

#     with open("./Problem4/summary_quality/summaries/" + fileName, encoding="latin-1") as summaryFile:
#         summary = summaryFile.read().lower()

#     trainSummariesWithNonRedundancyScore.append((summary, int(docNonRedundancy)))

# # read test data
# with open("./Problem4/summary_quality/test_data.json", 'r') as fin:
#     testMap = json.load(fin)

# testSummariesWithNonRedundancyScore = []
# for fileName in testMap.keys():
#     docNonRedundancy = testMap[fileName]['nonredundancy']

#     with open("./Problem4/summary_quality/summaries/" + fileName, encoding="latin-1") as summaryFile:
#         summary = summaryFile.read().lower()

#     testSummariesWithNonRedundancyScore.append((summary, int(docNonRedundancy)))


# w = models.KeyedVectors.load_word2vec_format('./Problem3/GoogleNews-vectors-negative300.bin', binary=True)

# # Note: set useExtraFeatures to False to use original 3 features
# trainFeatureVectors, trainTargetLabels = getNonRedundancyFeatureVectors(trainSummariesWithNonRedundancyScore, w, useExtraFeatures=True)

# # train model
# model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=350)
# model.fit(trainFeatureVectors, trainTargetLabels)

# # test model
# # Note: set useExtraFeatures to False to use original 3 features
# testFeatureVectors, testTargetLabels = getNonRedundancyFeatureVectors(testSummariesWithNonRedundancyScore, w, useExtraFeatures=True)
# testDataPredictions = model.predict(testFeatureVectors)

# # evaluate
# mse = mean_squared_error(testTargetLabels, testDataPredictions)
# pearson, _ = pearsonr(testTargetLabels, testDataPredictions)
# print(mse)
# print(pearson)

# ------------------- End 4.2

# ------------------- Start 4.3

# # read train data
# with open("./Problem4/summary_quality/train_data.json", 'r') as fin:
#     trainMap = json.load(fin)

# trainSummariesWithCoherenceScores = []
# for fileName in trainMap.keys():
#     docCoherence = trainMap[fileName]['coherence']

#     with open("./Problem4/summary_quality/summaries/" + fileName, encoding="latin-1") as summaryFile:
#         summary = summaryFile.read()

#     trainSummariesWithCoherenceScores.append((summary, int(docCoherence)))

# # read test data
# with open("./Problem4/summary_quality/test_data.json", 'r') as fin:
#     testMap = json.load(fin)

# testSummariesWithCoherenceScores = []
# for fileName in testMap.keys():
#     docCoherence = testMap[fileName]['coherence']

#     with open("./Problem4/summary_quality/summaries/" + fileName, encoding="latin-1") as summaryFile:
#         summary = summaryFile.read()

#     testSummariesWithCoherenceScores.append((summary, int(docCoherence)))


# # train model
# trainFeatureVectors, trainTargetLabels = getCoherenceFeatureVectors(trainSummariesWithCoherenceScores)
# model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100)
# model.fit(trainFeatureVectors, trainTargetLabels)

# # test model
# testFeatureVectors, testTargetLabels = getCoherenceFeatureVectors(testSummariesWithCoherenceScores)
# testDataPredictions = model.predict(testFeatureVectors)

# # evaluate
# mse = mean_squared_error(testTargetLabels, testDataPredictions)
# pearson, _ = pearsonr(testTargetLabels, testDataPredictions)
# print(mse)
# print(pearson)


# ------- End 4.3
