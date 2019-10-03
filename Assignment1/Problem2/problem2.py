import os
import re
from bs4 import BeautifulSoup, element
import sys
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

input_path = "./Problem2/WSD/WSD/wsd_data.xml"
with open(input_path) as fin:
    content = fin.read()

content = re.sub('\n*', '', content)
soup = BeautifulSoup(content, 'html.parser')

# put together bag of words
bagOfWords = set()
for context in soup.find_all('context'):
    splitSentence = context.contents

    for sentencePart in splitSentence:
        if type(sentencePart) is not element.Tag:
            bagOfWords = bagOfWords | set(sentencePart.split())

# put together feature labels
featureVectorLabels = ["senseId", 'previousBigram',
                       'previousTrigram', 'nextBigram', 'nextTrigram',
                       "nearEndOfSentence"]
featureVectorLabels.extend(["BoW count for word: " + w for w in bagOfWords])

# put together feature vectors
featureVectors = None
for welt in soup.find_all('welt'):
    instances = welt.find_all("instance")

    for instance in instances:
        ans = instance.ans
        senseId = ans['senseid']

        context = instance.context
        splitSentence = context.contents

        featuresForSentence = []

        sentenceStr = ""
        for index, sentencePart in enumerate(splitSentence):
            if type(sentencePart) is element.Tag:  # found <head> tag
                featureForSentence = []

                # first item in feature is the target (senseId)
                featureForSentence.append(senseId)

                lemma = sentencePart.string

                previousContext = splitSentence[index - 1]
                nextContext = splitSentence[index + 1]

                previousBigram = "_".join(
                    previousContext.split()[-1:] + [lemma])
                previousTrigram = "_".join(
                    previousContext.split()[-2:] + [lemma])

                nextBigram = "_".join([lemma] + nextContext.split()[:1])
                nextTrigram = "_".join([lemma] + nextContext.split()[:2])

                # add colloctional features to vector
                featureForSentence += [abs(hash(previousBigram)), abs(hash(
                    previousTrigram)), abs(hash(nextBigram)), abs(hash(nextTrigram))]

                # add feature representing if trigram includes the end of sentence
                # in other words, this sense appears within 3 words/punctuations from the end of the sentence
                nearEndOfSentence = "." in nextTrigram or "!" in nextTrigram or "?" in nextTrigram
                featureForSentence.append(1 if nearEndOfSentence else 0)

                # add to total list of feature vectors for this sentence
                featuresForSentence.append(featureForSentence)
            else:
                # found normal string, prepare for BoW count search
                sentenceStr = sentenceStr + " " + sentencePart

        # add BoW counts
        for word in bagOfWords:
            wordFreqInContext = sentenceStr.count(word)

            for featureForSentence in featuresForSentence:
                featureForSentence.append(wordFreqInContext)

        # # add vector (possibly multiple per sentence) to list of vectors
        npFeaturesForSentence = np.array(featuresForSentence)
        if featureVectors is None:
            featureVectors = npFeaturesForSentence
        else:
            np.concatenate([featureVectors, npFeaturesForSentence])

print("Finished putting together feature vectors")

# at this point 'featureVectors' should be a 2D array containing our feature vectors
targets = featureVectors[:, 0]
featureVectors = featureVectors[:, 1:]

# run feature selection
test = SelectKBest(score_func=f_classif, k=10)
selectedFeatures = test.fit_transform(featureVectors, targets)
selectedIndices = test.get_support(indices=True)

print("Finished running feature selection")

# find labels that were selected
selectedLabels = []
for index in selectedIndices:
    selectedLabels.append(featureVectorLabels[index])

print(selectedLabels)
