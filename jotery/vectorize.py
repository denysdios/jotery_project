import csv
import time

import numpy
import spacy


# Data Collection
details = []
soft_details = []
with open("cleaned_sfq.csv", 'r', encoding="latin-1") as file:
    csvreader = csv.DictReader(file)

    for row in csvreader:
        details.append(row['cleaned details'])
        soft_details.append(row['smoothed details'])

# Vectorization
nlp = spacy.load("en_core_web_lg",
                 disable=['ner', 'tagger', 'parser', 'lemmatizer', 'tok2vec', 'attribute_ruler'])
vectors = [nlp(detail).vector for detail in details]

# Save the vectors
numpy.save('dummyvectors', dict(zip(soft_details, vectors)))
