# Libraries
import string
import time
import nltk
import pandas
from nltk import word_tokenize
from sklearn.cluster import KMeans
import numpy as np
import spacy
import csv
from scipy import spatial
import tensorflow as tf
import keras
import tkinter
from sklearn.model_selection import train_test_split
from pandastable import Table

pandas.options.mode.chained_assignment = None  # default='warn'


def csv_data(wdata=('id', 'smoothed details', 'added', 'cleaned details', 'category'), document='categorized.csv'):
    """
    Data Collection from the csv file
    """
    data_dict = {datum: [] for datum in wdata}
    with open(document, 'r', encoding="latin-1") as file:
        csvreader = csv.DictReader(file)

        for row in csvreader:
            for datum in data_dict:
                data_dict[datum].append(row[datum])

    return data_dict


def get_model(NUM_CATEGORIES):
    """
    Returns a compiled neural network model.
    """
    model = keras.Sequential  (
        [
            # Add a hidden layers
            tf.keras.layers.Dense(512, activation="sigmoid"),

            # Add an output layer
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_evaluate(vectors, labels, modelsave=False, testsize=0.2, num_categories=100):
    """
    This Function trains and evaluates the given neural network model
    """
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(vectors), np.array(labels), test_size=testsize
    )
    # Get a compiled neural network
    model = get_model(NUM_CATEGORIES=num_categories)
    # Fit model on training data
    model.fit(x_train, y_train, epochs=10)
    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    if modelsave:
        model.save('nn_model')


def clustering(det_vec, save_to_file=False):
    """
    Compared to classic K-means approach Spectral Clustering with NN algorithm is more
    sufficient to determine boundaries !
    However our purpose is to determine nearest entities.
    """
    # K Means Algorithm
    model = KMeans(n_clusters=100, random_state=0).fit_predict(det_vec)

    # Print How many items in each Category
    dict_data = {i: 0 for i in range(100)}
    for mod in model:
        dict_data[mod] += 1
    print(dict_data)

    model = pandas.DataFrame(model)
    supform = pandas.read_csv("cleaned_sfq.csv", encoding="latin-1")
    supform["category"] = model

    # Save the document
    if save_to_file:
        supform.to_csv("categorized.csv", encoding="latin-1", index=False)

    return supform


def get_sentences(model, all_vec, sent_vector, data, n=3):
    """
    This function Evaluates cosine similarity based on vectors in the given category and the vector in the input.
    """
    # Track Running Time
    start_time = time.time()

    # Neural Network Model Predicts
    predict_cat = model.predict(np.expand_dims(sent_vector, 0))
    category = np.argmax(predict_cat[0])

    # Choose only category = category values
    categorized_data = (data.loc[data['category'] == str(category)])

    # Find Similarities and add into data
    det_vec = [all_vec[i] for i in categorized_data['smoothed details']]
    sim_det = [1 - spatial.distance.cosine(sent_vector, i) if i[0] != 0.0 else 0 for i in det_vec]
    categorized_data['similarity'] = sim_det

    calculation_time = round(time.time() - start_time, 4)
    return categorized_data.sort_values(by='similarity', ascending=False)[0:n], calculation_time


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
    return " ".join(tokenized)


def main():
    # NLP
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'tagger', 'parser', 'lemmatizer', 'tok2vec', 'attribute_ruler'])

    # Read NN Model
    model = tf.keras.models.load_model('nn_model')

    # Read Vectors
    vectors = np.load('sentencevectors.npy', allow_pickle=True).item()

    # Get the main Data
    data = csv_data()

    def printresult(numb):
        sentence = q_entry.get()
        sentence = tokenize(sentence)
        sentence = nlp(sentence)
        finalval, runtime = get_sentences(model=model, all_vec=vectors,
                                          sent_vector=sentence.vector, data=pandas.DataFrame(data), n=numb)

        pt = Table(frame, showtoolbar=True, showstatusbar=True, dataframe=finalval)
        runtimet.replace(1.0, 2.0, runtime)
        pt.show()
        pt.redraw()

    """
    Graphical User Interface
    """
    g_u_i = tkinter.Tk()
    g_u_i.title('Query Wizard')
    g_u_i.iconbitmap('jot_icon.ico')
    pilot = tkinter.Canvas(g_u_i, height=300, width=1300)
    pilot.pack()

    # Labels
    q_label = tkinter.Label(g_u_i, text='Query', font='Arial 14')
    app_name = tkinter.Label(g_u_i, text='Jotery', font='Times 18 italic bold')
    dev_label = tkinter.Label(g_u_i, text='Developed by Boran Deniz BAKIR')
    dev_label.configure(font=('Times New Roman', 10, 'bold'))
    numb_label = tkinter.Label(g_u_i, text='Output size')
    runtimet = tkinter.Text(g_u_i, width=6, height=1)
    runtimel = tkinter.Label(g_u_i, text='Run Time(seconds)')

    # Entries
    q_entry = tkinter.Entry(g_u_i, width=110, fg='red', font='Arial 14')
    numb_entry = tkinter.Entry(g_u_i, width=3, fg='red')
    numb_entry.insert(0, '3')

    # Buttons
    q_button = tkinter.Button(g_u_i, text='Run', command=lambda: printresult(numb=int(numb_entry.get())), height=2, width=7)
    clear_button = tkinter.Button(g_u_i, text='clear', command=lambda: q_entry.delete(first=0, last='end'))

    # Frames
    frame = tkinter.Frame(g_u_i)

    # Place Things
    app_name.place(relx=0.47, rely=0.01)
    q_label.place(relx=0.005, rely=0.10)
    q_entry.place(relx=0.05, rely=0.10)
    numb_label.place(relx=0.10, rely=0.2)
    numb_entry.place(relx=0.15, rely=0.2)
    q_button.place(relx=0.92, rely=0.2)
    clear_button.place(relx=0.87, rely=0.2)
    runtimet.place(relx=0.51, rely=0.25)
    runtimel.place(relx=0.43, rely=0.25)
    frame.pack(fill='both', expand=True)
    dev_label.pack()

    g_u_i.mainloop()


if __name__ == "__main__":
    main()
