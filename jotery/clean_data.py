import csv
import re
import pandas
import time
import nltk
import string
from nltk import word_tokenize

start_time = time.time()

# Data Cleaning function
cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


def cleanhtml(raw_html):
    """
    This function provides cleaning html tags in a given sentence
    """
    return re.sub(cleaner, '', raw_html)


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.
    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.lower()
    tokenized = word_tokenize(document)
    for word in tokenized.copy():
        if word in string.punctuation or word in nltk.corpus.stopwords.words("english"):
            tokenized.remove(word)
            continue
        if not word.isalpha():
            tokenized.remove(word)
            continue
    return tokenized


# Data Collection
details = []
questions = []
with open("support_forum_questions.csv", 'r', encoding="latin-1") as file:
    csvreader = csv.DictReader(file)

    for row in csvreader:
        details.append(row['details'])
        questions.append(row['question'])


# Cleaning Data
data = [cleanhtml(i) for i in details]
data = [" ".join(tokenize(datum)) for datum in data]
smooth_data = [cleanhtml(i) for i in details]
cleandata = pandas.DataFrame(data)
smooth_data = pandas.DataFrame(smooth_data)

# Reading and writing CSV file
supform = pandas.read_csv("support_forum_questions.csv", encoding="latin-1")
supform["smoothed details"] = smooth_data
supform["cleaned details"] = cleandata
supform.to_csv("cleaned_sfq.csv", encoding="latin-1", index=False)

print(supform)
