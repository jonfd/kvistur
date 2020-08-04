#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from tensorflow.keras.optimizers import Adam

MAX_LENGTH = 40


def read_data(path):
    """Read an input text file which contains one word per line. Compound words contain a single underscore
    character denoting where a binary split occurs (e.g., "þorsteins_son") while base words (non-compounds) contain
    no underscores at all (e.g., "fylgdi"). Each character in a word is labelled with 0 if it is not followed by an
    underscore, or 1 otherwise.

    :param path: The path to the input file.
    :return: A randomly shuffled list of words and their labels.
    """
    words = []
    labels = []

    # Create a dictionary of all possible labels
    index_label = {-1: [0] * MAX_LENGTH}
    for index in range(MAX_LENGTH):
        label = [0] * MAX_LENGTH
        label[index] = 1
        index_label[index] = label

    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            w = word.replace('_', '')

            # Skip words that are longer than MAX_LENGTH
            if len(w) > MAX_LENGTH:
                continue

            index = word.find('_')
            labels.append(index_label[index])
            words.append(w)

    words, labels = shuffle(words, labels, random_state=42)
    return words, labels


def encode_data(words, labels, chars):
    """Encode and pad words and labels.

    :param words: A list of words.
    :param labels: A list of labels.
    :param chars: A character-to-integer map.
    :return: Words (encoded as integers) and labels that have been padded to MAX_LENGTH.
    """
    encoded_words = [[chars.get(c, 1) for c in w] for w in words]
    encoded_labels = pad_sequences(labels, maxlen=MAX_LENGTH, padding='post')
    x = pad_sequences(encoded_words, maxlen=MAX_LENGTH, padding='post')
    y = np.array([label.reshape(MAX_LENGTH, 1) for label in encoded_labels])

    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", help="Learning rate for the Adam optimizer (default: 1e-3)",
                        default=1e-3, type=float)
    parser.add_argument("--batch-size", help="Set the batch size (default: 128)", default=128, type=int)
    parser.add_argument("--epochs", help="Set the number of epochs (default: 20)", default=20, type=int)
    parser.add_argument("--vocab", help="Path to a predefined vocabulary")
    parser.add_argument("--model-dir", help="Save checkpoints and vocabulary to the specified directory")
    parser.add_argument("--train", help="Read training data from specified file", required=True)
    parser.add_argument("--val", help="Read validation data from specified file")

    args = parser.parse_args()

    # Input (training data)
    train_path = args.train
    val_path = args.val

    # Output (model checkpoints and vocabulary)
    model_dir = args.model_dir

    # Read training data
    train_words, train_labels = read_data(train_path)

    if args.vocab:
        # Use a predefined vocabulary
        with open(args.vocab, encoding='utf-8') as f:
            char_to_index = json.load(f)
    else:
        # Create a new vocabulary from the training data, mapping characters to integers. Reserve 0 for masking padded
        # characters, 1 for unknown characters and use 2 for all integers.
        char_to_index = {'<mask>': 0, '<unk>': 1}
        char_to_index.update({char: 2 for char in '0123456789'})

        special_chars = max(char_to_index.values()) + 1
        train_chars = {c for w in train_words for c in w} - set(char_to_index)
        train_chars_to_index = {c: num + special_chars for num, c in enumerate(sorted(train_chars))}
        char_to_index.update(train_chars_to_index)

    vocab_size = max(char_to_index.values()) + 1

    # Save the vocabulary dictionary
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    vocab_out_path = os.path.join(model_dir, 'chars.json')
    with open(vocab_out_path, 'w', encoding='utf-8') as f:
        json.dump(char_to_index, f, ensure_ascii=False, indent=2)

    # Encode and pad training and validation data
    x_train, y_train = encode_data(train_words, train_labels, char_to_index)

    if val_path:
        val_words, val_labels = read_data(val_path)
        val_data = encode_data(val_words, val_labels, char_to_index)
    else:
        val_data = None

    # Create the model
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LENGTH,)))
    model.add(Embedding(vocab_size, 128, mask_zero=True))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['binary_accuracy']
                  )

    model.summary()

    # Train the model
    checkpoint_path = model_dir + "/kvistur-{epoch:02d}.hdf5"
    mcp_save = ModelCheckpoint(checkpoint_path, verbose=0, save_best_only=False,
                               save_weights_only=False, mode='auto', period=1)

    history = model.fit(x_train,
                        y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_data=val_data,
                        callbacks=[mcp_save],
                        verbose=2)


if __name__ == '__main__':
    main()
