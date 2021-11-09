import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import chain
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.text import Tokenizer
from functools import partial

inputs = []
targets = []
test = pd.read_excel('CheckingAddress.xlsx')
seed = 500
n = 200

# Shuffle the data
df = pd.DataFrame(test[['detail_list', 'check']]
                  ).sample(frac=1.0)


# Positive and negative indices of the original dataframe randomly sampled
pos_inds = df[(df["check"] == 1)].sample(n=n, random_state=seed).index
neg_inds = df[(df["check"] == 0)].sample(n=n, random_state=seed).index


print(pos_inds.shape, neg_inds.shape)
# Get the data with indices
pos_test_samples = df.loc[pos_inds]
neg_test_samples = df.loc[neg_inds]

# Create the test dataset
test_samples = pd.concat([pos_test_samples, neg_test_samples]).sample(
    frac=1.0, random_state=seed)
print(test_samples.shape)


# Separate inputs and targets
test_inputs, test_labels = test_samples["detail_list"], test_samples["check"]

# Training data
train_samples = df.loc[~df.index.isin(test_samples.index)]
train_inputs, train_labels = train_samples["detail_list"], train_samples["check"]

assert test_labels.sum() + train_labels.sum() == df["check"].sum()
# Get the most common 1000 words
n_vocab = 1000
cnt = Counter(chain(*train_inputs.to_list()))
words, _ = zip(*cnt.most_common(n_vocab))

# class inbalance
print((1-(train_labels.value_counts()/train_labels.shape[0])).to_dict())


# Define a count vectorizer that will create bag-of-words n-gram vectors
count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 5))
# Fit the count vectorizer on the limited vocabulary to learn ngrams
count_vectorizer.fit(words)


def get_transformed_input(inputs, count_vectorizer, max_seq_length=15):
    """ Get the transformed input by adding empty spaces until the defined length is reached """
    for inp in inputs:
        inputs = []
        for row in inp:
            inputs += [row.split()[1:]]

    transformed_inputs = []
    for inp in inputs:
        if len(inp) < max_seq_length:
            inp = inp + ['']*(max_seq_length-len(inp))
        elif len(inp) > max_seq_length:
            inp = inp[:max_seq_length]

        # Get the output transformed by the count vectorizer
        out = count_vectorizer.transform(inp)
        transformed_inputs.append(out.toarray())

    transformed_inputs = np.stack(transformed_inputs, axis=0)
    return transformed_inputs


# Get preprocessed train data
train_transformed_input = get_transformed_input(
    [train_inputs], count_vectorizer)
print("Transformed train set shape: {}".format(train_transformed_input.shape))
# Get preprocessed test data
test_transformed_input = get_transformed_input([test_inputs], count_vectorizer)
print("Transformed test set shape: {}".format(test_transformed_input.shape))

# Get n-grams learnt
feature_names = count_vectorizer.get_feature_names()
print("# of n-grams learnt: {}".format(len(feature_names)))
n_gram_count = len(feature_names)


# Simplified angular LSH (example)
# bins = 32
# R = np.random.normal(size=(n_gram_count, bins//2))
# ngram_out = count_vectorizer.transform(
#     ["now", "Now", "now!", "today", "jungle"]).toarray()
# print("ngram_out.shape = {}".format(ngram_out.shape))
# h_x = np.dot(ngram_out, R)
# print("h_x.shape = {}".format(h_x.shape))
# hash_bins = np.argmax(np.concatenate([h_x, -h_x], axis=-1), axis=-1)
# print("Hash bins for the words are: {}".format(hash_bins))

# Solving LSH for batches of data
n_hashes = 8
bins = 32
R = np.random.normal(size=(n_gram_count, bins//2))

ngram_out = count_vectorizer.transform(words).toarray()
print("ngram_out.shape = {}".format(ngram_out.shape))

ngram_out = np.expand_dims(ngram_out, axis=0)
h_x = np.einsum('bij,jk-> bik', ngram_out, R)

print("x.shape = {}".format(h_x.shape))
# [batch dim, n_hashes, n_words]
hash_vectors = np.concatenate([h_x, -h_x], axis=-1)
print("hash_vectors.shape = {}".format(hash_vectors.shape))


hash_vectors = hash_vectors[0, :, :]
hash_vectors = hash_vectors / \
    np.sqrt((hash_vectors**2).sum(axis=-1, keepdims=True))

# hash_vectors

k = 5
sort_ids = np.argsort(-np.dot(hash_vectors, hash_vectors.T), axis=-1)[:, 1:k+1]
probe_ids = np.random.randint(30, size=10)
for pid in probe_ids:
    print(words[pid], [words[sid] for sid in sort_ids[pid]])


################################################################
tf.keras.backend.clear_session()


class Hash2Vec(tf.keras.layers.Layer):

    def __init__(self, hash_bins):
        """ Initialize the random matrix """
        # A matrix of size [number of hashes, feature dim, number of bins]
        self.R = None
        self.hash_bins = hash_bins

        super().__init__()

    def build(self, input_shape):
        # [b, seq, ngrams]
        self.R = self.add_weight(
            shape=(input_shape[-1], self.hash_bins//2),
            initializer='random_normal',
            trainable=False
        )

    def call(self, x):
        """ The actual computation of the hash vectors """
        # x - [batch size, sequence leng, feature dim]
        # out - [batch size, number of hashes, seqeuence length, number of bins]
        batch_h_x = tf.einsum('bij,jk->bik', x, self.R)

        # out - [batch size, n_hashes, sequence len]
        #batch_closest_neighbor = tf.cast(tf.transpose(tf.argmax(batch_hashed_x, axis=-1), [0, 2, 1]), 'float32')
        batch_concat_h_x = tf.concat([batch_h_x, - batch_h_x], axis=-1)
        # Normalize the vectors
        return tf.math.l2_normalize(batch_concat_h_x, axis=-1)

    def get_config(self):
        """ TODO """
        pass


seq_length = 15
ngram_dim = len(count_vectorizer.get_feature_names_out())
bins = 64


def get_hash2vec_classifier(seq_length, ngram_dim, bins):
    # Defining the model

    # Defining input layer
    ngram_input = tf.keras.layers.Input(shape=(seq_length, ngram_dim))
    # Define the hash2vec layer
    hash2vec_out = Hash2Vec(bins)(ngram_input)
    # Define a layer to flatten the time dimension and create a long vector containing outputs of all steps
    concat_out = tf.keras.layers.Flatten()(hash2vec_out)
    # Final layer
    dense_out = tf.keras.layers.Dense(1, activation='sigmoid')(concat_out)

    # Define the final model and compile it
    model = tf.keras.models.Model(inputs=ngram_input, outputs=dense_out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics='accuracy')

    return model


model = get_hash2vec_classifier(seq_length, ngram_dim, bins)
model.summary()


def train_and_test_model(model_func, train_x, train_y, test_x, test_y, n_trials=10):
    results = []
    for trial in range(n_trials):
        print("Training model for trial {}".format(trial))
        trial_results = {}

        model = model_func()

        history = model.fit(
            train_x,
            train_y,
            class_weight=(1-(train_y.value_counts() /
                          train_y.shape[0])).to_dict(),
            epochs=10,
            batch_size=20,
            verbose=0
        )
        trial_results["train_accuracy"] = history.history['accuracy']

        out = model.evaluate(test_x, test_y, return_dict=True)
        trial_results["test_accuracy"] = out['accuracy']

        print("\tTest accuracy: {}".format(out['accuracy']))

        results.append(trial_results)
    print('\nDone')

    return model, results


get_hash2vec_classifier_partial = partial(
    get_hash2vec_classifier, seq_length=seq_length, ngram_dim=ngram_dim, bins=bins)
model, hash2vec_results = train_and_test_model(
    get_hash2vec_classifier_partial, train_transformed_input, train_labels, test_transformed_input, test_labels)

test_samples['predict'] = model.predict(test_transformed_input)


################################################################
# comparing to BOW
def get_bag_of_ngrams_classifier():

    tf.keras.backend.clear_session()

    # Defining input layer
    ngram_input = tf.keras.layers.Input(shape=(seq_length, ngram_dim))

    # Define a layer to flatten the time dimension and create a long vector containing outputs of all steps
    summed_out = tf.keras.layers.Flatten()(ngram_input)
    # Dense hidden layer
    dense_out = tf.keras.layers.Dense(1, activation='sigmoid')(summed_out)

    # Define the final model and compile it
    model = tf.keras.models.Model(inputs=ngram_input, outputs=dense_out)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics='accuracy')

    return model


model = get_bag_of_ngrams_classifier()
model.summary()

bon_results = train_and_test_model(
    get_bag_of_ngrams_classifier, train_transformed_input, train_labels, test_transformed_input, test_labels)


################################################################
hash2vec_test_accuracies = [dct["test_accuracy"] for dct in hash2vec_results]
bon_test_accuracies = [dct["test_accuracy"] for dct in bon_results]

print("Hash2vec: {} (mean) / {} (std)".format(
    np.array(hash2vec_test_accuracies).mean(),
    np.array(hash2vec_test_accuracies).std()
))
print("Bag-of-ngrams: {} (mean) / {} (std)".format(
    np.array(bon_test_accuracies).mean(),
    np.array(bon_test_accuracies).std()
))
