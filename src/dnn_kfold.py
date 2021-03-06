import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
# from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply
# from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU, GlobalAveragePooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization
# from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import Adam
# from keras.preprocessing.text import Tokenizer
# from keras.layers.core import K

from keras.wrappers.scikit_learn import KerasClassifier  # enables use of sklearn cross_val_score

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import re
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from keras import backend as K

from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# import matplotlib.pyplot as plt
import time


def dnn_model(learning_rate=0.0005, lr_decay=1e-6, drop_out=0.5, input_shape=(25000,)):

    DROPOUT = drop_out

    # Input
    _inputs = Input(shape=input_shape)

    # Try like 5 hidden layers starting big and getting small

    # maybe try kernel_initializer='he_normal' on dense layers
    X = Dense(256, activation="relu", name="dense_1")(_inputs)
    X = BatchNormalization()(X)
    X = Dropout(DROPOUT)(X)

    X = Dense(128, activation="relu", name="dense_2")(X)
    X = BatchNormalization()(X)
    X = Dropout(DROPOUT)(X)

    X = Dense(64, activation="relu", name="dense_3")(X)
    X = BatchNormalization()(X)
    X = Dropout(DROPOUT)(X)

    X = Dense(32, activation="relu", name="dense_4")(X)
    X = BatchNormalization()(X)
    X = Dropout(DROPOUT)(X)

    X = Dense(16, activation="relu", name="dense_5")(X)
    X = BatchNormalization()(X)
    X = Dropout(DROPOUT)(X)

    _outputs = Dense(1, activation='sigmoid', name="output_1")(X)

    # gather model
    model = Model(inputs=_inputs, outputs=_outputs, name="bad_dnn")

    # configure optimizer
    optimizer = Adam(lr=learning_rate, decay=lr_decay)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

    return model


# https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
class roc_callback(Callback):

    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)

        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 6)), str(round(roc_val, 6))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def write_model_timestamp(model_type, kfolds, scores, note):
    '''
    Parameters:
    model_type = string description of the model(s) used to make these scores
    kfolds     = how many folds in kfold cross validation used
    scores     = list of ROC AUC avg scores of models for each class, floats should be like 0.9784
    note       = string, whatever is of note about the model, made a change or whatever

    Returns:
    None, but writes (appends) a line to scores.txt in the root directory so that progress can be tracked
    The format is:
            time(s)~model_type~kfold~avg_roc_auc~toxic_auc~s_toxic_auc~obscene_auc~threat_auc~insult_auc~i_hate_auc~notes

    scores.txt is a tilde '~' seperated CSV like:
        time~model_type~kfold~avg_roc_auc~toxic_auc~s_toxic_auc~obscene_auc~threat_auc~insult_auc~i_hate_auc~notes
        1520303252~0.9794005980274005~note something
    '''

    out_text = "{:10.0f}~{:}~{:2d}~{:0.8f}~{:0.8f}~\
            {:0.8f}~{:0.8f}~{:0.8f}~{:0.8f}~{:0.8f}~{:}\n".format(time.time(),
                                                                  model_type,
                                                                  kfolds,
                                                                  np.mean(scores),
                                                                  scores[0],
                                                                  scores[1],
                                                                  scores[2],
                                                                  scores[3],
                                                                  scores[4],
                                                                  scores[5],
                                                                  note)

    with open("../scores.txt", 'a') as out_file:
        out_file.write(out_text)

        print("wrote:")
        print(out_text)
        print("to file")


def evaluate_models():

    train_data = pd.read_csv('../data/train.csv').fillna(' ')
    test_data = pd.read_csv('../data/test.csv').fillna(' ')

    train_text = train_data['comment_text']
    test_text = test_data['comment_text']
    all_text = pd.concat([train_text, test_text])

    print("(╯°o°）╯︵ ┻━┻  Vectorize combined corpus with TfidfVectorizer.")
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=25000)    # 10k was initial

    word_vectorizer.fit(all_text)
    print("\n(╯^.^）╯︵ ┻━┻  :  Completed TfidfVectorizer fitting.\n")

    print("\n(╯^.^）╯︵ ┻━┻  :  Transform corpora with word_vectorizer.\n")
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)

    print("train shape:", train_word_features.shape)
    print("test shape:", test_word_features.shape)

    print("\n(╯^.^）╯︵ ┻━┻  :  All training/test data transformed.\n")

    scores = []

    NUM_FOLDS = 10

    print(train_word_features.shape)

    print("\n(╯^.^）╯︵ ┻━┻  :  Create KerasClassifier wrapper for cross_val_score to use\n")
    # Wrap Keras model so it can be used by scikit-learn
    classifier = KerasClassifier(build_fn=dnn_model,
                                 epochs=5,
                                 batch_size=512,
                                 verbose=1)

    train_features = train_word_features.copy()

    # submission = pd.DataFrame.from_dict({'id': test['id']})

    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    data = {}   # will store the results of each kfold as a list with key class_name

    for class_name in class_names:
        train_target = train_data[class_name]

        print("\n(╯^.^）╯︵ ┻━┻  :  Starting for class: {:}\n".format(class_name))

        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=1337)

        class_results = []

        idx = -1    # will hold which iteration since it didn't like
                    # enumerate in the kfold.split loop.
                    # start at -1 so that it starts at the 0 index

        # this will get the indices of the train and test splits
        # don't forget to use these indices on the data to actually carve it
        for train, test in kfold.split(train_features, train_target):

            idx += 1

            model = dnn_model(learning_rate=0.005,
                              drop_out=0.50,
                              input_shape=(train_word_features.shape[1],))

            # declare callback here because it's a mess to do inline in model.fit
            _callbacks = [roc_callback(
                training_data=(
                    train_word_features[train],
                    train_target[train]),
                validation_data=(
                    train_word_features[test],
                    train_target[test]))]

            model.fit(train_word_features[train],
                      train_target[train],
                      batch_size=512,
                      epochs=5,
                      verbose=1,
                      validation_data=(train_word_features[test], train_target[test]),
                      callbacks=_callbacks)

            preds = model.predict(train_word_features[test])
            result = roc_auc_score(train_target[test], preds)

            class_results.append(result)

            print("Class: {: <14} Slice: {:2d} ROC AUC: {:0.4f}".format(class_name, idx, result))

            K.clear_session()

        data[class_name] = class_results

        print('\n(╯^.^）╯︵ ┻━┻  :  :  CV Spread for class "{}\n":'.format(class_name))
        for result in data[class_name]:
            print("    {:0.4f}".format(result), end=" ")

        print(" ")

        cv_score = np.mean(data[class_name])
        scores.append(cv_score)

        print('\n(╯^.^）╯︵ ┻━┻  :  CV score for class "{}" is {:0.4}\n'.format(class_name, cv_score))

        classifier.fit(train_features, train_target)
    #     submission[class_name] = classifier.predict_proba(test_features)[:, 1]

    print('\n(╯^.^）╯︵ ┻━┻  :  Total CV score is {:0.4f}\n'.format(np.mean(scores)))


if __name__ == "__main__":
    evaluate_models()
