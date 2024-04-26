#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn.neural_network
import sklearn.preprocessing

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default="fiction-train.txt", type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2324/datasets/"):
        # if not os.path.exists(name):
        #     print("Downloading dataset {}...".format(name), file=sys.stderr)
        #     licence_name = name.replace(".txt", ".LICENSE")
        #     urllib.request.urlretrieve(url + licence_name, filename=licence_name)
        #     urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        # if not os.path.exists(name):
        #     print("Downloading dataset {}...".format(name), file=sys.stderr)
        #     licence_name = name.replace(".txt", ".LICENSE")
        #     urllib.request.urlretrieve(url + licence_name, filename=licence_name)
        #     urllib.request.urlretrieve(url + name, filename=name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants


class_1_dict = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "y": "ý", "A": "Á", "E": "É", "I": "Í",
                "O": "Ó", "U": "Ú", "Y": "Ý"}
class_2_dict = {"c": "č", "d": "ď", "e": "ě", "n": "ň", "r": "ř", "s": "š", "t": "ť", "z": "ž", "C": "Č",
                "D": "Ď", "E": "Ě", "N": "Ň", "R": "Ř", "S": "Š", "T": "Ť", "Z": "Ž"}
class_3_dict = {"u": "ů", "U": "Ů"}
classes_dict = [{}, class_1_dict, class_2_dict, class_3_dict]


def parse_data(input_text, output_text, target_needed):
    LETTERS_CLASSES = ["abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ&!(),-.:;?'\"\n 0123456789", "áéíóúýÁÉÍÓÚÝ",
                       "čďěňřšťžČĎĚŇŘŠŤŽ", "ůŮ"]
    LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
               "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
               "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "&", "!", "(", ")", ",", "-", ".", ":", ";", "?", "'",
               "\"", "\n", " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "á", "é", "í", "ó", "ú", "ý", "Á",
               "É", "Í", "Ó", "Ú", "Ý", "č", "ď", "ě", "ň", "ř", "š", "ť", "ž", "Č", "Ď", "Ě", "Ň", "Ř", "Š", "Ť", "Ž",
               "ů", "Ů"]
    train_data = []
    train_target = []
    input_text = "    " + input_text + "    "
    text_length = len(input_text)

    for i in range(4, text_length - 4):
        train_data.append([*input_text[i - 4:i + 5]])
        if target_needed:
            train_target.append([output_text[i - 4] in c for c in LETTERS_CLASSES])

    ohe = sklearn.preprocessing.OneHotEncoder(categories=[LETTERS for i in range(9)])
    train_data = ohe.fit_transform(train_data)
    return train_data, train_target


def reproduce_text(input_text, model, d):
    LETTERS_NODIA = "acdeeinorstuuyzACDEEINORSTUUYZ"
    test_data, _ = parse_data(input_text, "", target_needed=False)
    pred = model.predict_proba(test_data)
    sentences = input_text.split('\n')
    for i in range(len(sentences)):
        words = sentences[i].split()
        sentences[i] = words

    output_text = ""
    i = 0
    for sentence in sentences:
        output_sentence = ""
        for word in sentence:
            found = False
            search_needed = True
            classes = []
            for letter in word:
                if letter in LETTERS_NODIA:
                    classes.append(pred[i])
                    if search_needed:
                        found = word in d.variants
                        search_needed = False
                else:
                    classes.append([0])
                i += 1
            output_word = reproduce_word(word, classes)
            if found:
                output_word = correct_word(output_word, d.variants[word])

            output_sentence += output_word + " "
            i += 1
        output_text += output_sentence[:len(output_sentence) - 1] + "\n"

    return output_text


def reproduce_word(word, classes):
    output_word = ""
    for i in range(len(word)):
        flag = False
        while not flag:
            c = np.argmax(classes[i])
            if c == 0:
                output_word += word[i]
                flag = True
            else:
                if word[i] in classes_dict[c].keys():
                    output_word += classes_dict[c][word[i]]
                    flag = True
                else:
                    classes[i][c] = 0
    return output_word


def correct_word(current_word, variants):
    variants_dist = []
    for word in variants:
        variants_dist.append(sum(current_word[i] != word[i] for i in range(len(current_word))))
    return variants[np.argmin(variants_dist)]


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.

        train_data, train_target = parse_data(train.data, train.target, target_needed=True)

        model = sklearn.neural_network.MLPClassifier((500,), verbose=True, max_iter=45)
        model.fit(train_data, train_target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        d = Dictionary()

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        text = reproduce_text(test.data, model, d)

        with open("output.txt", "w") as file:
            file.write(text)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
