#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import spacy
from io import open
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 6000000  # Warning: requires large amount of RAM

trainFileDir = "./dataset/" # NOTE: add back Problem1
trainFilePaths = []
for _, _, filenames in os.walk(trainFileDir):
    trainFilePaths = [trainFileDir + path for path in filenames]

trainFilesContents = ""
for trainFilePath in trainFilePaths:
    trainFile = open(trainFilePath, encoding="utf8", mode="r")

    trainFilesContents += trainFile.read()

    trainFile.close()


# In[14]:


doc = nlp(unicode(trainFilesContents))


# In[19]:


from collections import Counter


# In[29]:


numberOfSentences = 0
for sent in doc.sents:
    numberOfSentences += 1

numberOfVerbs = 0
numberOfPreps = 0
preps = []
for token in doc:
    if token.pos_ == "VERB":
        numberOfVerbs += 1
    if token.dep_ == "prep":
        numberOfPreps += 1
        preps.append(token.text)

prepsFreq = Counter(preps)
mostCommonPreps = prepsFreq.most_common(3)
mostCommonPrepNames = [str(tuple[0]) for tuple in mostCommonPreps]

uniqueEntities = set()
totalEntities = 0
for ent in doc.ents:
    totalEntities += 1
    uniqueEntities.add(str(ent.label_))
        
print("Number of sentences: " + str(numberOfSentences))
print("Number of verbs: " + str(numberOfVerbs) +
      ". Average per sentence: " + str(float(numberOfVerbs) / numberOfSentences))
print("Number of prepositions: " + str(numberOfPreps) + ". Three most common prepositions: " + str(mostCommonPrepNames))
print("Number of entities: " + str(totalEntities) + ". Unique entity lables: " + str(uniqueEntities))


# In[38]:


# look at all the dependency parsings
depMap = {}
for token in doc:
    if str(token.dep_) not in depMap:
        depMap[str(token.dep_)] = []
    depMap[str(token.dep_)].append({
        "text": token.text,
        "test pos": str(token.pos_),
        "head": token.head.text,
        "head pos": str(token.head.pos_)
    })

print(depMap.keys())
for dep, obj in depMap.items():
    print(dep)
    print(obj)
    break


# In[ ]:





# In[ ]:




