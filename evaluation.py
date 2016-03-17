from __future__ import division
from nltk.corpus.reader import ConllCorpusReader
import Train

conllreader = ConllCorpusReader(".", "de-test.t", ('words', 'pos'))
states = Train.states

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    # Initialize base cases (t == 0)
    for y in states:
        if sum(emit_p.prob((obs[0], y1)) for y1 in states) != 0:
            V[0][y] = start_p.logprob(y) + emit_p.logprob((obs[0], y))
        else:
            V[0][y] = start_p.logprob(y)
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            if sum(emit_p.prob((obs[t], y1)) for y1 in states) != 0:
                (prob, state) = max((V[t-1][y0] + trans_p.logprob((y0, y)) + emit_p.logprob((obs[t], y)), y0) for y0 in states)
            else:
                (prob, state) = max((V[t-1][y0] + trans_p.logprob((y0, y)), y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        # Don't need to remember the old paths
        path = newpath

    # Return the most likely sequence over the given time frame
    n = len(obs) - 1
    (prob, state) = max((V[n][y], y) for y in states)
    return path[state]

Vit1 = []
Vit2 = []
Vit3 = []
Vit4 = []
for sent in conllreader.sents():
    Vit1.append(zip(sent, viterbi(sent, states, Train.A0j, Train.Aij, Train.Biw)))
    Vit2.append(zip(sent, viterbi(sent, states, Train.A0jLap, Train.AijLap, Train.BiwLap)))
    Vit3.append(zip(sent, viterbi(sent, states, Train.A0jGT, Train.AijGT, Train.BiwGT)))
    Vit4.append(zip(sent, viterbi(sent, states, Train.A0jMLE, Train.AijMLE, Train.BiwMLE)))


# function for writing tagged corpora to files in CoNLL format
def write_conll(filename, tagged_corpus):
    with open(filename, 'w') as out_file:
        for tagged_sent in tagged_corpus:
            tagged_words = ('\t'.join(w_t) for w_t in tagged_sent)
            out_file.write('\n'.join(tagged_words) + '\n\n')

write_conll('unsmoothed.tt', Vit1)
write_conll('laplace.tt', Vit2)
write_conll('good_turing.tt', Vit3)
write_conll('MLE.tt', Vit4)