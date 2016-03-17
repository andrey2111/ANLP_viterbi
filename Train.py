from __future__ import division
from nltk.corpus.reader import ConllCorpusReader
from nltk.probability import FreqDist, DictionaryProbDist, LaplaceProbDist, SimpleGoodTuringProbDist, MLEProbDist

conllreader = ConllCorpusReader(".", "de-train.tt", ('words', 'pos'))  # getting a train corpus from file
states = ('VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.')  # list of 12 POS tags
sentslen = len(conllreader.tagged_sents())  # getting number of sentences

tagfdist = FreqDist(pair[1] for pair in conllreader.tagged_words())   # getting frequence of (word,tag)

firsttagfdist = FreqDist(pair[0][1] for pair in conllreader.tagged_sents())  # getting frequence of first tags
A0j = DictionaryProbDist(dict(map(lambda (k, x): (k, x/sentslen), firsttagfdist.iteritems())))
A0jLap = LaplaceProbDist(firsttagfdist)
A0jGT = SimpleGoodTuringProbDist(firsttagfdist)
A0jMLE = MLEProbDist(firsttagfdist)

TagPair = []
words = conllreader.tagged_words()
for i in range(0, len(words)-1):
    TagPair.append((words[i][1], words[i+1][1]))

TagPairfdist = FreqDist(TagPair)
Aij = DictionaryProbDist(dict(map(lambda (k, x): (k, x/tagfdist.get(k[0])), TagPairfdist.iteritems())))
AijLap = LaplaceProbDist(TagPairfdist)
AijGT = SimpleGoodTuringProbDist(TagPairfdist)
AijMLE = MLEProbDist(TagPairfdist)

TagWordfdist = FreqDist(conllreader.tagged_words())
Biw = DictionaryProbDist(dict(map(lambda (k, x): (k, x/tagfdist.get(k[1])), TagWordfdist.iteritems())))
BiwLap = LaplaceProbDist(TagWordfdist)
BiwGT = SimpleGoodTuringProbDist(TagWordfdist)
BiwMLE = MLEProbDist(TagWordfdist)
