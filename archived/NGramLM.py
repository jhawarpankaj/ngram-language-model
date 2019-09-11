import copy


class NGramLM:

    def __init__(self, n):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = {"<s>", "</s>"}

    def update(self, text):
        temp = copy.copy(text)
        # Pad the input list with <s> and </s>
        [temp.insert(i, "<s>") for i in range(self.n-1)]
        text.insert(len(temp), "</s>")

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
        {self.vocabulary.add(word) for word in temp}

    def create_ngramlm(self, corpus_path):
        file = open(corpus_path)
        for line in file:
            self.update(line.split())

    def word_prob(self, word, context):
        prob = 1/len(self.vocabulary)
        ngram_tuple = tuple(context) + (word,)
        ngram_count = self.ngram_counts.get(ngram_tuple)
        context_count = self.context_counts.get(tuple(context))
        if ngram_count is None:
            ngram_count = 0
        if context_count is not None:
            prob = ngram_count/context_count
        print("Word: " + str(word) + ", context: " + str(
            context) + ", ngram_count: " + str(ngram_count) + ", context_count: " + str(context_count) + ", prob: " + str(prob))
        return prob