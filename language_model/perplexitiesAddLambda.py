from nltk.tokenize import word_tokenize
import os
import math

########## Function Definitions #######################


def getGramCount(gram, countMap):
    if gram in countMap:
        return countMap[gram]

    return 0


def calculatePerplexity(alpha, tokenizedWordSet, bigramMap, trigramMap):
    setLogProbability = 0
    for index, word in enumerate(tokenizedWordSet):
        word = word.lower()

        trigramProbability = 0
        if index > 1:
            previousPreviousWord = tokenizedWordSet[index - 2]
            previousWord = tokenizedWordSet[index - 1]

            previousBigram = "_".join([previousPreviousWord, previousWord])
            previousBigramUNK = "_".join([previousPreviousWord, "UNK"])

            previousBigramCount = getGramCount(previousBigram, bigramMap)
            if previousBigramCount == 0:
                previousBigramCount = getGramCount(
                    previousBigramUNK, bigramMap)

            trigram = "_".join([previousPreviousWord, previousWord, word])
            trigramUNK = "_".join([previousPreviousWord, previousWord, "UNK"])

            trigramCount = getGramCount(trigram, trigramMap)
            if trigramCount == 0:
                trigramCount = getGramCount(trigramUNK, trigramMap)

            trigramProbability = (
                trigramCount + alpha) / (previousBigramCount + (alpha * len(tokenizedWordSet)))

            setLogProbability += math.log(trigramProbability, 2)

    # perplexity of set for given alpha
    # Note: because we're in log base 2 space, perplexity = 2 ^ ((-1/n) * logProbability)
    setPerplexity = 2 ** ((-1/len(tokenizedWordSet)) * setLogProbability)
    return setPerplexity

############### End function definitions #################################
############## Start Training #################################


trainFileDir = "./Problem3/train/"

for _, _, filenames in os.walk(trainFileDir):
    trainFilePaths = [trainFileDir + path for path in filenames]

trainFilesContents = ""
for trainFilePath in trainFilePaths:
    trainFile = open(trainFilePath, encoding="utf8", mode="r")

    trainFilesContents += trainFile.read()

    trainFile.close()

# get tokens
tokenizedTrainContents = word_tokenize(trainFilesContents)

trainSet = tokenizedTrainContents[:int(len(tokenizedTrainContents) * 0.8)]
holdOutSet = tokenizedTrainContents[int(len(tokenizedTrainContents) * 0.8):]

# put together unigram map
unigramMap = {}
for token in trainSet:
    token = token.lower()

    if token in unigramMap:
        curCount = unigramMap[token]
        unigramMap[token] = curCount + 1
    else:
        unigramMap[token] = 1

# cleanup unigram map
wordsToRemove = []
UNKCount = 0
for word, count in unigramMap.items():
    if count <= 3:
        UNKCount += count
        wordsToRemove.append(word)

unigramMap["UNK"] = UNKCount
for word in wordsToRemove:
    del unigramMap[word]

# replace infrequent words in master list with UNK
trainSet = [
    w if w not in wordsToRemove else "UNK" for w in trainSet]

print("finished unigram map")

# put together bigram and trigram maps
bigramMap = {}
trigramMap = {}
for index, token in enumerate(trainSet):
    token = token.lower()

    if index > 0:
        previousToken = trainSet[index - 1]

        bigram = "_".join([previousToken, token])

        if bigram in bigramMap:
            curCount = bigramMap[bigram]
            bigramMap[bigram] = curCount + 1
        else:
            bigramMap[bigram] = 1

    if index > 1:
        previousPreviousToken = trainSet[index - 2]
        previousToken = trainSet[index - 1]

        trigram = "_".join([previousPreviousToken, previousToken, token])

        if trigram in trigramMap:
            curCount = trigramMap[trigram]
            trigramMap[trigram] = curCount + 1
        else:
            trigramMap[trigram] = 1

print("finished bigram and trigram map")

###################### End Training  ######################
###################### Start Testing ######################

testFilePaths = ["./Problem3/test/test01.txt", "./Problem3/test/test02.txt"]

testFile1 = open(testFilePaths[0], encoding="utf8", mode="r")
testFile1Contents = testFile1.read()
testFile1.close()

testFile2 = open(testFilePaths[1], encoding="utf8", mode="r")
testFile2Contents = testFile2.read()
testFile2.close()

tokenizedTestFile1 = word_tokenize(testFile1Contents)
testFile1Perplexity1 = calculatePerplexity(
    0.1, tokenizedTestFile1, bigramMap, trigramMap)
testFile1Perplexity3 = calculatePerplexity(
    0.3, tokenizedTestFile1, bigramMap, trigramMap)

print("Got perplexity " + str(testFile1Perplexity1) +
      " for test file 1 with alpha=0.1")
print("Got perplexity " + str(testFile1Perplexity3) +
      " for test file 1 with alpha=0.3")

tokenizedTestFile2 = word_tokenize(testFile2Contents)
testFile2Perplexity1 = calculatePerplexity(
    0.1, tokenizedTestFile2, bigramMap, trigramMap)
testFile2Perplexity3 = calculatePerplexity(
    0.3, tokenizedTestFile2, bigramMap, trigramMap)

print("Got perplexity " + str(testFile2Perplexity1) +
      " for test file 2 with alpha=0.1")
print("Got perplexity " + str(testFile2Perplexity3) +
      " for test file 2 with alpha=0.3")
