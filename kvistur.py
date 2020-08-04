#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 40


class Node(object):
    __slots__ = ["form", "mod", "head"]

    def __init__(self, form):
        self.form = form
        self.mod = None
        self.head = None

    def __repr__(self):
        if self.mod:
            return f"({self.mod}, {self.head})"

        return f'"{self.form}"'

    def get_tree(self):
        if self.mod:
            return self.mod.get_tree(), self.head.get_tree()

        return self.form

    def get_binary(self):
        if self.mod:
            return self.mod.form, self.head.form

        return None, self.form

    def flatten(self):
        if self.mod:
            return self.mod.flatten() + self.head.flatten()
        else:
            return [self.form.lower()]

    def split(self, pos):
        self.mod = Node(self.form[:pos])
        self.head = Node(self.form[pos:])


class CharEncoder(dict):
    def __init__(self, chars):
        self.unknown = 1
        self.number = 2

        super().__init__(chars)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            if item.isdecimal():
                return self.number  # Numerical character
            return self.unknown  # Unknown character

    def encode(self, word):
        return [self[c] for c in word]


class Kvistur(object):
    def __init__(self, model_path, chars_path):
        self.model = tf.keras.models.load_model(model_path)

        with open(chars_path, encoding='utf-8') as f:
            self.chars = CharEncoder(json.load(f))

    def split_binary(self, words):
        encoded = [self.chars.encode(w.lower()) for w in words]
        encoded = pad_sequences(encoded, maxlen=MAX_LENGTH, padding='post')
        predictions = self.model.predict(encoded, batch_size=32)

        splits = {}

        for word, prediction in zip(words, predictions):
            max_pos = np.argmax(prediction)
            max_val = prediction[max_pos]

            if max_val >= 0.5:
                splits[word] = max_pos
            else:
                splits[word] = None

        return splits

    def split(self, words):
        nodes = {w: Node(w) for w in set(words)}

        remainder = list(nodes.values())
        while remainder:
            remainder_forms = list({n.form for n in remainder})
            remainder_splits = self.split_binary(remainder_forms)

            new_remainder = []
            for node in remainder:
                pos = remainder_splits[node.form]

                if pos:
                    node.split(pos)
                    new_remainder.append(node.mod)
                    new_remainder.append(node.head)

            remainder = new_remainder

        return [nodes[w] for w in words]


def main():
    model_path = 'models/kvistur-20g.hdf5'
    chars_path = 'models/chars.json'

    kvistur = Kvistur(model_path, chars_path)

    words = ["krisengeschüttelten",
             "Gesundheitsministerium",
             "Dringlichkeitssitzung",
             "zurückkehren",
             "Coronakrise"]

    nodes = kvistur.split(words)

    for node in nodes:
        print(node.form)
        print("Tree:", node.get_tree())
        print("Binary:", node.get_binary())
        print("Flat:", node.flatten())
        print()


if __name__ == '__main__':
    main()
