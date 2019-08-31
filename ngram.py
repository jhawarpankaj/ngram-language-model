from NGramLM import NGramLM
import math


def get_ngrams(n, text):

    # Adding begin and end word
    [text.insert(i, "<s>") for i in range(0, n-1)]
    text.append("</s>")

    # return a generator object for all n-grams
    for i in range(n-1, len(text)):
        yield text[i], tuple(text[i-n+1: i])


def text_prob(model, text):
    prob = 0

    # Pad the input list with <s> and </s>
    [text.insert(i, "<s>") for i in range(model.n - 1)]
    text.insert(len(text), "</s>")

    # Calculate all ngram probabilities
    for i in range(model.n - 1, len(text)):
        val = model.word_prob(text[i], [text[context] for context in range(i - model.n + 1, i)])
        prob = prob + math.log(val + 1) if val == 0 else prob + math.log(val)

    return prob


if __name__ == "__main__":
    n = 4
    # text = ["ab", "cd", "ed", "df", "df"]
    # for (a, b) in get_ngrams(n, text):
    #     print("Word:" + a + ", Context: " + str(b))

    model = NGramLM(n)
    model.create_ngramlm("../../../files/warpeace.txt")

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

    print(text_prob(model, list("God has given it to me, let him who touches it beware!".split(" "))))
    print(text_prob(model, list("Where is the prince, my Dauphin?".split(" "))))
