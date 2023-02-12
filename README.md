# association-regression
## About regression of word-pairs based on neuralnets

Here is a report on making a predictor function from data cloud of given words and its foremost related cowords.

The words are represented by 100-D vectors.

The word-pairs are generated from random words found in glove gigaword wiki database and its determined cowords as the most similar words. The similarity have to be above a critical value like 80%, 90% or 95% to get better R2-scores. It is also possible to get or to append databases like twitter or the same databeses but with lower or higher dimensions like from 50 to 300.

The word-pairs are in a 200-D hyperspace as a data cloud on which I try to make as precise predictor function as I can. The function is based on a multi-layer neurnet with hidden layers 1000 and 1000, and the input and the output are 100-100 dimensional. In fact it is a word -> word function based on its vectors.

To measure its goodness we call the R2-scores on test data that could be scored from 0% to 100%. The score 100% means every point (word) is on the hyperplane defined by the predictor function while score 0% means there is no correlation between the points and the function. It is important to notice that the average similarity between the both of the words of the pairs defines an upper limit to the R2-score. In this view the score about 85% let me say is enough good.

To check my experiments on the parameters of the neuralnet and the input data, have a look at xlsx or its screenshot, png.
