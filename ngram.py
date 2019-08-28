def get_ngrams(n, text):

    # Adding begin and end word
    [text.insert(i, "<s>") for i in range(0, n-1)]
    text.append("</s>")

    # return a generator object for all n-grams
    for i in range(n-1, len(text)):
        yield text[i], tuple(text[i-n+1: i])


if __name__ == "__main__":
    a = get_ngrams(4, ["ab", "cd", "ed", "df", "df"])
    print(next(a))