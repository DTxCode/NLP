For problem 2, you can run the feature selection by executing:

`python Problem2/problem2.py`

The top 10 feature labels will print at the end.

--------------------------------

For problem 3, you can run the create the ngramCounts.txt file by running:

`python Problem3/ngramCounts.py`

To see the perplexity values for the holdout set/test sets, you can run:

`python Problem3/perplexitiesInterpolation.py`
`python Problem3/perplexitiesAddLambda.py`

For the interpolation approach and the add-alpha approach respectively.
Note that these two scripts will not create any files. They will print the results to the console.

-------------------------------------

For problem 4, you can run the POS Tagging on the test file by running:

`python Problem4/problem4.py`

To re-create the files containing the word-tag counts, the tag unigram counts, the tag bigram counts,
the transition probabilites, the emission probabilities, or the 5 generated sentences, just un-comment
the releveant lines in the file. 
See lines:
197-199
213
227
230-278
