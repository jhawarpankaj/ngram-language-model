from NGramLM import NGramLM
from NGramLMPart2 import NGramLMPart2
import math


def get_ngrams(n, text):
    # Adding begin and end word
    [text.insert(i, "<s>") for i in range(0, n-1)]
    text.append("</s>")

    # return a generator object for all n-grams
    for i in range(n-1, len(text)):
        yield text[i], tuple(text[i - n + 1: i])
    return


def text_prob(model, text, delta=0):
    prob = 0

    # Pad the input list with <s> and </s>
    [text.insert(i, "<s>") for i in range(model.n - 1)]
    text.insert(len(text), "</s>")

    # Calculate all ngram probabilities
    for i in range(model.n - 1, len(text)):
        # print("Probability of word: " + str(text[i]) + ", given context: " + str(
        #     [text[context] for context in range(i - model.n + 1, i)]) + ": ")
        val = model.word_prob(text[i], [text[context] for context in range(i - model.n + 1, i)], delta=0)
        # print("Probability: " + str(val))
        prob = prob + math.log(val + 1) if val == 0 else prob + math.log(val)

    return prob


def perplexity(model, corpus_path, delta=0):
    total_word = set()
    file = open(corpus_path)
    perplexity = 0
    for sentence in file:
        perplexity = perplexity + text_prob(model, sentence.split("."), delta)
        for word in sentence:
            total_word.add(word)
    perplexity = (1 / len(total_word)) * perplexity
    return perplexity


if __name__ == "__main__":
    n = 3
    text = ["ab", "cd", "ed", "df", "df"]
    for (a, b) in get_ngrams(n, text):
        print("Word:" + a + ", Context: " + str(b))

    # text = ["ab", "cd", "ed", "df", "df"]
    print("dsdddddddddd" + str(text))
    for (a, b) in get_ngrams(n, text):
        print("Word:" + a + ", Context: " + str(b))

    # model = NGramLM(n)
    # masked_file_name = model.mask_rare("../../../files/warpeace.txt")
    # model.create_ngramlm(masked_file_name)

    # Debug prints
    # file_ngram = open("../ngram-counts.log", "w")
    # file_ngram.write(str(model.ngram_counts))
    # file_ngram.close()
    #
    # file_context = open("../context-counts.log", "w")
    # file_context.write(str(model.context_counts))
    # file_context.close()

    # print(model.ngram_counts)
    # print(model.context_counts)
    # print(model.vocabulary)
    # print(model.word_prob("than", ["out", "of", "malice"]))
    # obj.update(["I", "love", "Computer", "Science", "Computer", "Science"])



    # # Part1
    # file = "../../../files/warpeace.txt"
    # model = NGramLM(n)
    # model.create_ngramlm(file)

    # # Part2
    # model = NGramLMPart2(n)
    # masked_file_name = model.mask_rare(file)
    # model.create_ngramlm(masked_file_name)


    # print("Log probability: " + str(text_prob(model, list("* God has given it to me, let him who touches it beware!".split(" ")))))
    # print(model.text_prob(model, list("Where is the prince, my Dauphin?".split(" "))))

    # # Perplexity
    # model1 = NGramLMPart2(3)
    # model2 = NGramLMPart2(3)
    # training_data1 = "../../../files/shakespeare.txt"
    # training_data2 = "../../../files/warpeace.txt"
    # test_data = "../../../files/sonnets.txt"
    # model1.create_ngramlm(training_data1)
    # model2.create_ngramlm(training_data2)
    # p1 = perplexity(model1, test_data, delta=0)
    # p2 = perplexity(model2, test_data, delta=0.5)
    # print(str(p1))
    # print(str(p2))

