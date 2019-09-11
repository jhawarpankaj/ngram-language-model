import os
import random
import math
import copy
random.seed(1)


class NGramLM:

    def __init__(self, n):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()
        self.vocabulary.add('<s>')
        self.vocabulary.add('</s>')

    def update(self, text):
        # Could have used the get_ngram method but mistakenly repeated the code.
        # Ultimately same thing is achieved.
        temp = copy.copy(text)
        if not temp:
            return
        [temp.insert(i, "<s>") for i in range(self.n - 1)]
        temp.insert(len(temp), "</s>")

        # Count the no of ngrams
        for i in range(len(temp) - self.n + 1):
            ngram_list = []
            for j in range(i, i + self.n):
                ngram_list.append(temp[j])
            ngram_tuple = tuple(ngram_list)
            if ngram_tuple in self.ngram_counts:
                count = self.ngram_counts.get(ngram_tuple)
                self.ngram_counts[ngram_tuple] = count + 1
            else:
                self.ngram_counts[ngram_tuple] = 1

        # Count the no of contexts
        for i in range(len(temp) - self.n + 2):
            context_list = []
            for j in range(i, i + self.n - 1):
                context_list.append(temp[j])
            context_tuple = tuple(context_list)
            if context_tuple in self.context_counts:
                count = self.context_counts.get(context_tuple)
                self.context_counts[context_tuple] = count + 1
            else:
                self.context_counts[context_tuple] = 1

        # Update the vocabulary
        for line in temp:
            for word in line.split(" "):
                if word not in '':
                    self.vocabulary.add(word.rstrip('\n'))

    def mask_rare(self, corpus):
        file = open(corpus)
        word_count_dict = {}

        # Build a dictionary with count of each word
        for line in file:
            for word in line.split(" "):
                if word in word_count_dict:
                    word_count_dict[word] = word_count_dict[word] + 1
                else:
                    word_count_dict[word] = 1

        # Create a new file corpus to contain the <unk> word
        dir_name = os.path.dirname(corpus)
        base_filename = os.path.basename(corpus).split(".")[0]
        file_format = os.path.basename(corpus).split(".")[1]
        masked_file_name = os.path.join(dir_name, base_filename + "-masked." + file_format)
        if os.path.exists(masked_file_name):
            os.remove(masked_file_name)

        # Replace all single occurring word in the file with <unk>
        with open(corpus) as infile, open(masked_file_name, "w+") as outfile:
            for line in infile:
                word_list = line.split(" ")
                masked_list = list(map(lambda word: "<unk>" if word_count_dict[word] == 1 else word, word_list))
                outfile.write(" ".join(masked_list))
                if "<unk>" == masked_list[-1]:
                    outfile.write("\n")
        infile.close()
        outfile.close()

        # Update the vocabulary with <unk>
        self.vocabulary.add('<unk>')
        return masked_file_name

    def create_ngramlm(self, corpus_path):
        file = open(corpus_path)
        for line in file:
            self.update(line.split())

    def word_prob(self, word, context, delta=0):

        if delta is 0:
            prob = 1 / len(self.vocabulary)

        # Replace word in ngram with <unk> if word not in vocab
        ngram_tuple = tuple(context) + (word,)
        ngram_list_with_unk = []
        for words in ngram_tuple:
            if words not in self.vocabulary:
                ngram_list_with_unk.append("<unk>")
            else:
                ngram_list_with_unk.append(words)
        ngram_count = self.ngram_counts.get(tuple(ngram_list_with_unk))

        if ngram_count is not None:
            numerator = ngram_count + delta
        else:
            numerator = 0 + delta

        # Replace word in context with <unk> if word not in vocab
        context_list_with_unk = []
        for words in context:
            if words not in self.vocabulary:
                context_list_with_unk.append("<unk>")
            else:
                context_list_with_unk.append(words)
        context_count = self.context_counts.get(tuple(context_list_with_unk))

        if context_count is not None:
            denominator = context_count + delta * len(self.vocabulary)
        else:
            denominator = 0 + delta * len(self.vocabulary)

        if denominator is not 0:
            prob = numerator / denominator

        # print("Word: " + str(word) + ", context: " + str(
        #     context) + ", ngram_count: " + str(ngram_count) + ", context_count: " + str(
        #     context_count) + ", prob: " + str(prob))

        return prob

    def random_word(self, context, delta=0):
        r = random.random()
        start_prob = 0

        # insert <s> if length of context is less than context size
        if len(context) < self.n - 1:
            for i in range(self.n - len(context) - 1):
                context = ["<s>"] + context

        elif len(context) > self.n - 1:
            context = context[-self.n + 1:]

        for key in sorted(self.vocabulary):
            end_prob = start_prob + self.word_prob(key, context, delta)
            if start_prob <= r < end_prob:
                return key
            start_prob = end_prob

    def likeliest_word(self, context, delta=0):
        max_prob = 0

        # insert <s> if length of context is less than context size
        if len(context) < self.n - 1:
            for i in range(self.n - len(context) - 1):
                context = ["<s>"] + context
        elif len(context) > self.n - 1:
            context = context[-self.n + 1:]

        for key in sorted(self.vocabulary):
            prob = self.word_prob(key, context, delta)
            if prob > max_prob:
                max_prob = prob
                word = key

        return word


def get_ngrams(n, text):
    # Adding begin and end word
    [text.insert(i, "<s>") for i in range(0, n-1)]
    text.append("</s>")

    # return a generator object for all n-grams
    for i in range(n-1, len(text)):
        yield text[i], tuple(text[i - n + 1: i])


def text_prob(model, text, delta=0):
    prob = 0
    temp = copy.copy(text)
    for word, context in get_ngrams(model.n, temp):
        val = model.word_prob(word, list(context), delta)
        prob = prob + math.log(val)
    return prob


def perplexity(model, corpus_path, delta=0):
    total_word = []
    file = open(corpus_path)
    perplexity = 0
    N = 0

    # Removing empty strings and new line from the end.
    for sentence in file:
        temp_list = sentence.split(" ")
        temp_list2 = []
        for word in temp_list:
            if word is not '':
                temp_list2.append(word.rstrip('\n'))
        N = N + len(temp_list2)
        perplexity = perplexity + text_prob(model, temp_list2, delta)

    perplexity = (1 / N) * perplexity
    return math.exp(-perplexity)


def random_text(model, max_length, delta=0):
    sentence = '<s>'
    initial_length = len(sentence.split(" ")) - 1
    i = 1
    while i <= max_length - initial_length:
        next_word = model.random_word(sentence.split(" "), delta)
        sentence = sentence + " " + next_word
        i = i + 1
        if next_word == '</s>':
            break
    return sentence


def likeliest_text(model, max_length, delta=0):
    sentence = '<s>'
    initial_length = len(sentence.split(" ")) - 1
    i = 1
    while i <= max_length - initial_length:
        next_word = model.likeliest_word(sentence.split(" "), delta)
        sentence = sentence + " " + next_word
        i = i + 1
        if next_word == '</s>':
            break
    return sentence


if __name__ == "__main__":
    # Part2
    n = 3
    corpus_path = "../../../files/warpeace.txt"
    model = NGramLM(n)
    masked_file_name = model.mask_rare(corpus_path)
    model.create_ngramlm(corpus_path)
    sen1 = "God has given it to me, let him who touches it beware!".split()
    sen2 = "Where is the prince, my Dauphin?".split()
    print(text_prob(model, sen1, 0))
    print(text_prob(model, sen1, 0))
    print(text_prob(model, sen1, 0.01))
    print(text_prob(model, sen2, 2))

    # # Part 4 Random word generator

    # # model = NGramLMPart2(3)
    # # model.create_ngramlm(corpus_path)
    # # print(random_text(model, 10, 0))
    #
    # model2 = NGramLM(2)
    # model3 = NGramLM(3)
    # model4 = NGramLM(4)
    # model5 = NGramLM(5)
    # # #
    # model2.create_ngramlm(corpus_path)
    # model3.create_ngramlm(corpus_path)
    # model4.create_ngramlm(corpus_path)
    # model5.create_ngramlm(corpus_path)
    #
    # print(likeliest_text(model2, 10, 0))
    # print(likeliest_text(model3, 10, 0))
    # print(likeliest_text(model4, 10, 0))
    # print(likeliest_text(model5, 10, 0))


