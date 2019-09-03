from NGramLMPart2 import NGramLMPart2
from ngram import *


class NGramInterpolator(NGramLMPart2):
    def __init__(self, n, lambdas):
        self.n = n
        self.lambdas = lambdas
        self.NGramLMs = [NGramLMPart2(ngram) for ngram in reversed(range(1, self.n + 1))]

    def update(self, text):
        for model in self.NGramLMs:
            model.update(text)

    def word_prob(self, word, context, delta=0):
        each_ngram_prob = [ngram_obj.word_prob(word, context, delta) for ngram_obj in self.NGramLMs]
        total_interpolation_prob = 0
        for index in range(len(self.lambdas)):
            total_interpolation_prob = total_interpolation_prob + (self.lambdas[index] * each_ngram_prob[index])
        return total_interpolation_prob


if __name__ == "__main__":
    text1 = "* God has given it to me, let him who touches it beware!".split()
    text2 = "Where is the prince, my Dauphin?".split()
    text = text2
    n = 3
    corpus = "../../../files/warpeace.txt"
    delta = 0.1

    # To get the interpolated probability
    model = NGramInterpolator(n, [0.33, 0.33, 0.33])
    masked_corpus = model.mask_rare(corpus)
    model.create_ngramlm(masked_corpus)
    interpol_prob = text_prob(model, text)
    print("Interpolated prob: " + str(interpol_prob))

    # To get the individual prob with and without smoothing
    for NGramLM in model.NGramLMs:
        NGramLM.create_ngramlm(masked_corpus)
    model.update(text)
    for NGramLM in model.NGramLMs:
        print("Prob for " + str(NGramLM.n) + "gram, with smoothing, delta = " + str(delta) + " : " + str(text_prob(NGramLM, text, delta)))
        print("Prob for " + str(NGramLM.n) + "gram, without smoothing: " + str(text_prob(NGramLM, text)))
