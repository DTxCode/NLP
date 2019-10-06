import os
import random

ALPHA = 0.5


def write_map(filePath, map):
    file = open(filePath, "w")

    string = ""
    for key, value in map.items():
        string += key + " : " + str(value) + "\n"

    file.write(string)

    file.close()


def increment_map(map, key):
    if key in map:
        curVal = map[key]
        map[key] = curVal + 1
    else:
        map[key] = 1


def get_emission_probability(emissionProbabilities, word, tag, tagUnigramCounts):
    key = word + "_" + tag
    unknownKey = "<UNK>_" + tag

    if key in emissionProbabilities:
        return emissionProbabilities[key]
    elif unknownKey in emissionProbabilities:
        return emissionProbabilities[unknownKey]

    return (0 + ALPHA) / (tagUnigramCounts[tag] + ALPHA * len(tagUnigramCounts))


def get_transition_probability(transitionProbabilities, prevTag, tag, tagUnigramCounts):
    key = prevTag + "_" + tag

    if key in transitionProbabilities:
        return transitionProbabilities[key]

    return (0 + ALPHA) / (tagUnigramCounts[prevTag] + ALPHA * len(tagUnigramCounts))


def viterbi(wordsToTag, tags, tagUnigramCounts, emissionProbabilities, transitionProbabilities):
    probabilityMatrix = [[0] * len(wordsToTag) for i in range(len(tags))]
    backpointers = [[None] * len(wordsToTag) for i in range(len(tags))]

    # start state probabilites
    totalTagCount = sum(tagUnigramCounts.values())
    initialTagProbabilities = {}
    for tag in tags:
        initialTagProbabilities[tag] = tagUnigramCounts[tag] / totalTagCount

    # initialize matrix
    firstWord = wordsToTag[0]
    for index, tag in enumerate(tags):
        firstWordEmission = get_emission_probability(
            emissionProbabilities, firstWord, tag, tagUnigramCounts)
        probabilityMatrix[index][0] = initialTagProbabilities[tag] * \
            firstWordEmission
        backpointers[index][0] = None

    # recursion steps
    for wordIndex, word in enumerate(wordsToTag):
        if wordIndex == 0:
            continue

        for tagIndex, tag in enumerate(tags):
            maxProbabilityForTag = 0
            maxProbabilityTag = None

            for prevTagIndex, prevTag, in enumerate(tags):
                prevProbability = probabilityMatrix[prevTagIndex][wordIndex - 1]
                transitionProbability = get_transition_probability(
                    transitionProbabilities, prevTag, tag, tagUnigramCounts)
                emissionProbability = get_emission_probability(
                    emissionProbabilities, word, tag, tagUnigramCounts)

                probability = prevProbability * transitionProbability * emissionProbability
                if probability > maxProbabilityForTag:
                    maxProbabilityForTag = probability
                    maxProbabilityTag = prevTag

            probabilityMatrix[tagIndex][wordIndex] = maxProbabilityForTag
            backpointers[tagIndex][wordIndex] = maxProbabilityTag

    bestLastTagProbability = 0
    bestLastTag = None
    bestLastTagIndex = 0
    for index, tag in enumerate(tags):
        prob = probabilityMatrix[index][len(wordsToTag) - 1]

        if prob > bestLastTagProbability:
            bestLastTag = tag
            bestLastTagIndex = index

    bestTagsReversed = [bestLastTag]
    for i in range(1, len(wordsToTag)):
        wordIndex = len(wordsToTag) - i
        bestLastTag = backpointers[bestLastTagIndex][wordIndex]
        bestLastTagIndex = tags.index(bestLastTag)
        bestTagsReversed.append(bestLastTag)

    bestTagsReversed.reverse()
    bestTags = bestTagsReversed
    taggedWords = []
    for index, word in enumerate(wordsToTag):
        taggedWords.append(word + "/" + bestTags[index])

    return taggedWords

############ End function definitions ##################


trainFileDir = "./Problem4/brown_train/"

for _, _, filenames in os.walk(trainFileDir):
    trainFilePaths = [trainFileDir + path for path in filenames]

trainFilesContents = ""
for trainFilePath in trainFilePaths:
    trainFile = open(trainFilePath, encoding="utf8", mode="r")

    trainFilesContents += trainFile.read()

    trainFile.close()

trainContents = trainFilesContents.split()

# Put together count maps
wordCounts = {}
wordTagCounts = {}
tagUnigramCounts = {}
tagBigramCounts = {}
for index, taggedWord in enumerate(trainContents):
    splitTaggedWord = taggedWord.split("/")

    if len(splitTaggedWord) < 2:
        print("Got word that wasn't in format 'word/tag': " + taggedWord)
        continue

    word = "".join(splitTaggedWord[:-1]).lower()
    tag = splitTaggedWord[-1:][0]
    wordTag = word + "_" + tag

    increment_map(wordCounts, word)

    increment_map(wordTagCounts, wordTag)

    increment_map(tagUnigramCounts, tag)

    if index > 0:
        previousTaggedWord = trainContents[index - 1]
        splitPreviousTaggedWord = previousTaggedWord.split("/")

        if len(splitPreviousTaggedWord) < 2:
            print("Got previous word that wasn't in format 'word/tag': " + taggedWord)
            continue

        previousTag = splitPreviousTaggedWord[-1:][0]
        tagBigram = previousTag + "_" + tag

        increment_map(tagBigramCounts, tagBigram)


# Replace infrequent words with <UNK> in wordTagCounts
wordsToRemove = []
for word, count in wordCounts.items():
    if count < 5:
        wordsToRemove.append(word)

filteredWordTagCounts = {}
for wordTag, count in wordTagCounts.items():
    splitWordTag = wordTag.split("_")
    word = splitWordTag[0]
    tag = splitWordTag[1]

    if word in wordsToRemove:
        newWordTag = "<UNK>_" + tag

        if newWordTag in filteredWordTagCounts:
            curVal = filteredWordTagCounts[newWordTag]
            filteredWordTagCounts[newWordTag] = curVal + count
        else:
            filteredWordTagCounts[newWordTag] = count

    else:
        filteredWordTagCounts[wordTag] = count

wordTagCounts = filteredWordTagCounts

# Write count maps
# write_map("./Problem4/wordTagCounts.txt", wordTagCounts)
# write_map("./Problem4/tagUnigramCounts.txt", tagUnigramCounts)
# write_map("./Problem4/tagBigramCounts.txt", tagBigramCounts)

transitionProbabilities = {}
for tagBigram, tagBigramCount in tagBigramCounts.items():
    splitBigram = tagBigram.split("_")
    prevTag = splitBigram[0]

    prevTagCount = tagUnigramCounts[prevTag]

    transitionProbability = (tagBigramCount + ALPHA) / \
        (prevTagCount + ALPHA * len(tagUnigramCounts))
    transitionProbabilities[tagBigram] = transitionProbability

# Write transition probabilities map
# write_map("./Problem4/transitionProbabilities.txt", transitionProbabilities)

emissionProbabilties = {}
for wordTag, wordTagCount in wordTagCounts.items():
    splitWordTag = wordTag.split("_")
    tag = splitWordTag[1]

    tagCount = tagUnigramCounts[tag]

    emissionProbability = (wordTagCount + ALPHA) / \
        (tagCount + ALPHA * len(tagUnigramCounts))
    emissionProbabilties[wordTag] = emissionProbability

# Write emission probabilites map
# write_map("./Problem4/emissionProbabilities.txt", emissionProbabilties)

# Generate 5 random sentences
# for _ in range(5):
#     sentence = ""
#     sentenceLikelihood = 1

#     tags = list(tagUnigramCounts)
#     totalTagCount = sum(tagUnigramCounts.values())
#     tagProbabilities = [tagUnigramCounts[t] / totalTagCount for t in tags]

#     # choose starting tag
#     tag = random.choices(tags, tagProbabilities)[0]
#     sentenceLikelihood *= tagProbabilities[tags.index(tag)]

#     while True:
#         words = []
#         probabilities = []
#         for wordTag, probability in emissionProbabilties.items():
#             wordTagSplit = wordTag.split("_")
#             word = wordTagSplit[0]
#             sampleTag = wordTagSplit[1]

#             if tag == sampleTag:
#                 words.append(word)
#                 probabilities.append(probability)

#         chosenWord = random.choices(words, probabilities)[0]
#         sentenceLikelihood *= probabilities[words.index(chosenWord)]

#         if "." in tag:  # end tag
#             sentence += chosenWord + "/" + tag
#             break
#         else:
#             sentence += chosenWord + "/" + tag + " "

#         tags = []
#         tagProbabilities = []
#         for tagBigram, probabilty in transitionProbabilities.items():
#             tagBigramSplit = tagBigram.split("_")
#             prevTag = tagBigramSplit[0]
#             nextTag = tagBigramSplit[1]

#             if prevTag == tag:
#                 tags.append(nextTag)
#                 tagProbabilities.append(probability)

#         tag = random.choices(tags, tagProbabilities)[0]
#         sentenceLikelihood *= tagProbabilities[tags.index(tag)]

#     print("Generated Sentence: \n" + sentence)
#     print("Sentence Probability: \n" + str(sentenceLikelihood) + "\n")


########## Test POS Tagging #########################

testFile = open("./Problem4/test/Test_file.txt", encoding="utf8", mode="r")
testFileContents = testFile.read()
testFile.close()

splitTestFileContents = testFileContents.split("\n")

sentenceIdTag = 0
sentenceStartIndex = 0
stringToWrite = ""
tags = list(tagUnigramCounts)
for index, word in enumerate(splitTestFileContents):
    if "< sentence ID =" in word:
        sentenceIdTag = word
        sentenceStartIndex = index + 1
    elif "< EOS >" == word:
        sentenceEndIndex = index
        taggedWords = viterbi(
            splitTestFileContents[sentenceStartIndex: sentenceEndIndex], tags,
            tagUnigramCounts, emissionProbabilties, transitionProbabilities)

        stringToWrite += sentenceIdTag + "\n"
        for taggedWord in taggedWords:
            stringToWrite += taggedWord + "\n"
        stringToWrite += "< EOS >\n"

taggedTestFile = open("./Problem4/test/Test_file_tagged.txt",
                      encoding="utf-8", mode="w")
taggedTestFile.write(stringToWrite)
taggedTestFile.close()
