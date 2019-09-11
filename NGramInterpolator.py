from ngram import *


class NGramInterpolator(NGramLM):
    def __init__(self, n, lambdas):
        super(NGramInterpolator, self).__init__(n)
        self.n = n
        self.lambdas = lambdas
        self.NGramLMs = [NGramLM(ngram) for ngram in reversed(range(1, self.n + 1))]

    def update(self, text):
        for models in self.NGramLMs:
            models.update(text)

    def word_prob(self, word, context, delta=0):
        each_ngram_prob = [ngram_obj.word_prob(word, context, delta) for ngram_obj in self.NGramLMs]
        total_interpolation_prob = 0
        for index in range(len(self.lambdas)):
            total_interpolation_prob = total_interpolation_prob + (self.lambdas[index] * each_ngram_prob[index])
        return total_interpolation_prob


if __name__ == "__main__":
    text1 = "God has given it to me, let him who touches it beware!".split()
    text2 = "Where is the prince, my Dauphin?".split()
    text = text1
    n = 3
    corpus_path = "../../../files/warpeace.txt"
    delta = 0

    # To get the interpolated probability
    model = NGramInterpolator(n, [0.33, 0.33, 0.33])
    masked_corpus = model.mask_rare(corpus_path)
    model.create_ngramlm(masked_corpus)  # All of the internal ngram's update method has been called
    interpol_prob = text_prob(model, text2)
    print("Interpolated prob: " + str(interpol_prob))

    # To get the individual prob with and without smoothing
    # for NGramLM in model.NGramLMs:
    #     NGramLM.create_ngramlm(masked_corpus)
    #     print(NGramLM.n)
    #     print("Prob for " + str(NGramLM.n) + "gram, with smoothing, delta = " + str(delta) + " : " + str(text_prob(NGramLM, text, delta)))
    #     print("Prob for " + str(NGramLM.n) + "gram, without smoothing: " + str(text_prob(NGramLM, text)))
    # #
    model2 = NGramLM(3)
    masked_file_name = model2.mask_rare(corpus_path)
    model2.create_ngramlm(masked_file_name)
    print(text_prob(model2, text1, delta))

    model3 = NGramLM(3)
    masked_file_name1 = model3.mask_rare(corpus_path)
    model3.create_ngramlm(masked_file_name1)
    print(text_prob(model3, text2, delta))


