from nltk.tokenize import word_tokenize
import os

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

# put together unigram map
unigramMap = {}
for token in tokenizedTrainContents:
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

# replace word in master list with UNK
tokenizedTrainContents = [
    w if w not in wordsToRemove else "UNK" for w in tokenizedTrainContents]

print("finished unigram map")

# put together bigram and trigram maps
bigramMap = {}
trigramMap = {}
for index, token in enumerate(tokenizedTrainContents):
    if index > 0:
        previousToken = tokenizedTrainContents[index - 1]

        bigram = "_".join([previousToken, token])

        if bigram in bigramMap:
            curCount = bigramMap[bigram]
            bigramMap[bigram] = curCount + 1
        else:
            bigramMap[bigram] = 1

    if index > 1:
        previousPreviousToken = tokenizedTrainContents[index - 2]
        previousToken = tokenizedTrainContents[index - 1]

        trigram = "_".join([previousPreviousToken, previousToken, token])

        if trigram in trigramMap:
            curCount = trigramMap[trigram]
            trigramMap[trigram] = curCount + 1
        else:
            trigramMap[trigram] = 1

print("finished bigram and trigram map")

# save results of unigram, bigram, and trigram counts
ngramCounts = open("./Problem3/ngramCounts.txt", "w")

ngramCountStr = ""
for word, count in unigramMap.items():
    ngramCountStr += word + " : " + str(count) + "\n"
for word, count in bigramMap.items():
    ngramCountStr += word + " : " + str(count) + "\n"
for word, count in trigramMap.items():
    ngramCountStr += word + " : " + str(count) + "\n"

ngramCounts.write(ngramCountStr)
ngramCounts.close()
