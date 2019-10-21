import os
import spacy
from io import open
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 6000000  # Warning: requires large amount of RAM

trainFileDir = "./Problem1/dataset/"
trainFilePaths = []
for _, _, filenames in os.walk(trainFileDir):
    trainFilePaths = [trainFileDir + path for path in filenames]

trainFilesContents = ""
for trainFilePath in trainFilePaths:
    trainFile = open(trainFilePath, encoding="utf8", mode="r")

    trainFilesContents += trainFile.read()

    trainFile.close()


doc = nlp(trainFilesContents)

numberOfSentences = 0
for sent in doc.sents:
    numberOfSentences += 1

numberOfVerbs = 0
for token in doc:
    if token.pos_ == "VERB":
        numberOfVerbs += 1


print("Number of sentences: " + str(numberOfSentences))
print("Number of verbs: " + str(numberOfVerbs) +
      ". Average per sentence: " + str(float(numberOfVerbs) / numberOfSentences))

# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#           token.shape_, token.is_alpha, token.is_stop)
