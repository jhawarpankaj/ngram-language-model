import os
import math


class NGramLMPart2:

    def __init__(self, n):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = {"<s>": self.n - 1, "</s>": 1}

    def update(self, text):
        # Pad the input list with <s> and </s>
        [text.insert(i, "<s>") for i in range(self.n-1)]
        text.insert(len(text), "</s>")

        # Count the no of ngrams
        for i in range(len(text) - self.n + 1):
            ngram_list = []
            for j in range(i, i + self.n):
                ngram_list.append(text[j])
            ngram_tuple = tuple(ngram_list)
            if ngram_tuple in self.ngram_counts:
                count = self.ngram_counts.get(ngram_tuple)
                self.ngram_counts[ngram_tuple] = count + 1
            else:
                self.ngram_counts[ngram_tuple] = 1

        # Count the no of contexts
        for i in range(len(text) - self.n + 2):
            context_list = []
            for j in range(i, i + self.n - 1):
                context_list.append(text[j])
            context_tuple = tuple(context_list)
            if context_tuple in self.context_counts:
                count = self.context_counts.get(context_tuple)
                self.context_counts[context_tuple] = count + 1
            else:
                self.context_counts[context_tuple] = 1

        # Update the vocabulary
        for line in text:
            for word in line.split(" "):
                if word in self.vocabulary:
                    val = self.vocabulary[word]
                    self.vocabulary[word] = val + 1
                else:
                    self.vocabulary[word] = 1

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
        # self.vocabulary.add("<unk>")
        return masked_file_name

    def create_ngramlm(self, corpus_path):
        # corpus_path = corpus_path.split(".")[0] + "-masked" + corpus_path.split(".")[1]
        file = open(corpus_path)
        for line in file:
            self.update(line.split())

    def word_prob(self, word, context, delta=0):
        if delta is 0:
            prob = 1 / len(self.vocabulary)

        # Replace word in ngram with <unk> if word not in vocab
        ngram_tuple = tuple(context) + (word,)
        ngram_list_with_unk = []
        for word in ngram_tuple:
            if word not in self.vocabulary:
                ngram_list_with_unk.append("<unk>")
            else:
                ngram_list_with_unk.append(word)
        ngram_count = self.ngram_counts.get(tuple(ngram_list_with_unk))

        if ngram_count is not None:
            numerator = ngram_count + delta
        else:
            numerator = 0 + delta

        # Replace word in context with <unk> if word not in vocab
        context_list_with_unk = []
        for word in context:
            if word not in self.vocabulary:
                context_list_with_unk.append("<unk>")
            else:
                context_list_with_unk.append(word)
        context_count = self.context_counts.get(tuple(context_list_with_unk))

        if context_count is not None:
            denominator = context_count + delta * len(self.vocabulary)
        else:
            denominator = 0 + delta * len(self.vocabulary)

        # print("Count of Ngram : " + str(tuple(ngram_list_with_unk)) + ": " + str(ngram_count))
        # print("Count of Context :" + str(tuple(context_list_with_unk)) + ": " + str(context_count))

        # if delta is not 0:
        #     print("After smoothing, numerator = " + str(numerator) + ", denominator: " + str(denominator))
        if denominator is not 0:
            prob = numerator / denominator
        return prob


