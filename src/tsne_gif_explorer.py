from FFMWrapper import FFMWrapper  # wraps FFMpeg to enable plot to video directly

import numpy as np
import pandas as pd

# tools to reduce dimensionality
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

from sklearn.feature_extraction.text import TfidfVectorizer

import time

import matplotlib.pyplot as plt
# import matplotlib.colors as colors   # allows use of gamma to tune cmaps
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec  # for multiple plots in the same image


class Tesseract(object):
    '''
    Take in data files from Toxic comment contest and generate TSNE movie

    '''

    def __init__(self,
                 tfidf_feature_count=25000,
                 tsne_feature_count=250,
                 tsne_iter=300,
                 tsne_lr=200,
                 tsne_perp=40
                 ):

        self.start_time = time.time()

        # ultimately determines the shape of the clouds
        self.tfidf_feature_count = tfidf_feature_count
        self.tsne_feature_count = tsne_feature_count
        self.tsne_iter = tsne_iter
        self.tsne_lr = tsne_lr
        self.tsne_perp = tsne_perp

        self.RAGE_SIZE = 15000  # how many toxic comments to carry towards plot
                                # may have to tune against available memory

        self.CALM_SIZE = 5000

        # output movie file name
        self.file_name = "../imgs/movs/tsne_tf{:05d}_ts{:04d}_lr{:03d}_iter{:04d}.mp4".format(
            self.tfidf_feature_count,
            self.tsne_feature_count,
            self.tsne_lr,
            self.tsne_iter)

        # paths to the data
        self.train_path = "../data/train.csv"
        self.test_path = "../data/test.csv"

        self.rage_df = None
        self.calm_df = None

        # placeholders for the data

        # comment strings:
        self.train_data = None
        self.train_text = None
        self.test_text = None
        self.all_text = None

        # tfidf vectors:
        self.rage_features = None   # vectorized words that are toxic
        self.calm_features = None   # random selection of non-toxic words
        # pointless to do test word feature vectorization since no predictions will
        # be done

        # tfidf vectors by classification
        self.rage_tfidf_vecs = None
        self.calm_tfidf_vecs = None

        # tfidf vectorizing 'model'
        self.word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 1),
            max_features=self.tfidf_feature_count)

        # PCA 'model'
        # n_components indicates how many features will be fed into TSNE later
        self.pca = PCA(n_components=self.tsne_feature_count)

        # data resulting from PCA
        self.pca_result = None


        # TSNE 'model'
        # self.tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=tsne_iter)
        self.tsne = TSNE(n_components=3,
                         verbose=1,
                         perplexity=self.tsne_perp,
                         learning_rate=self.tsne_lr,
                         n_iter=tsne_iter,
                         n_jobs=10)

        # data resulting from TSNE
        self.tsne_result = None

        # Data that will be fed into imaging:
        self.calm_x = None
        self.calm_y = None
        self.calm_z = None

        self.rage_x = None
        self.rage_y = None
        self.rage_z = None

        self.step_size = 5   # in degrees
        fps = 30 / self.step_size
        self.ffm = FFMWrapper(self.file_name, _vid_fps=fps, _width=2000, _height=1000)

    def run_me(self):
        '''
        Make running the class a one liner after declaration
        '''

        self.read_data()

        self.vectorize_words()

        self.run_PCA()

        self.run_TSNE()

        self.make_plots()

        self.cleanup()

        print("Total time for run {:}s".format(time.time() - self.start_time))

    def read_data(self):
        '''
        Read in the data to be processed
        '''

        print("Tesseract.read_data(): starting")

        self.train_data = pd.read_csv(self.train_path).fillna(' ')
        test = pd.read_csv(self.test_path).fillna(' ')

        self.train_text = self.train_data['comment_text']
        self.test_text = test['comment_text']   # not self. because this is temp and will be deleted

        del test    # not going to use it anymore

        self.all_text = pd.concat([self.train_text, self.test_text])

        print("Tesseract.read_data(): complete")

    def vectorize_words(self):
        '''
        Fit TfidfVectorizer to self.all_text and then transform the train words
        for later use.
        '''

        print("Tesseract.vectorize_words(): starting")
        print("    fit TfidfVectorizer...")
        self.word_vectorizer.fit(self.all_text)
        print("    TfidfVectorizer fit completed")

        print("    Transform calm and rage word corpus to vectors")

        # rage and calm here are temp variables
        # extract rows where one or more toxic flags is turned on
        self.rage_df = self.train_data.loc[np.sum(self.train_data.iloc[:,2:], axis=1) >= 1]

        # extract rows where no toxic flags are turned on
        self.calm_df = self.train_data.loc[np.sum(self.train_data.iloc[:,2:], axis=1) == 0]

        perms = np.random.permutation(self.RAGE_SIZE)    # big wurm, big perm!

        # take a random subsample since calm > 140k rows, and after it is a full
        # matrix memory requirements will exceed physical memory
        self.calm_df = pd.DataFrame(np.take(self.calm_df.values, perms[:self.CALM_SIZE], axis=0), columns=self.rage_df.columns)
        # TODO: wipe out the row below this, seems pointless
        self.rage_df = pd.DataFrame(np.take(self.rage_df.values, perms, axis=0), columns=self.calm_df.columns)

        self.calm_tfidf_vecs = self.word_vectorizer.transform(self.calm_df.loc[:, "comment_text"])
        self.rage_tfidf_vecs = self.word_vectorizer.transform(self.rage_df.loc[:, "comment_text"])

        print("    training words vectorized")
        print("Tesseract.vectorize_words(): complete")

    def run_PCA(self):
        '''
        Fit the PCA and then save the result in premade var
        '''

        print("Tesseract.run_PCA(): starting")

        # make a temp var that stacks calm and rage for latent feature extraction
        temp_calm = self.calm_tfidf_vecs.todense()
        temp_rage = self.rage_tfidf_vecs.todense()

        calm_rage = np.vstack((temp_calm, temp_rage))

        self.pca_result = self.pca.fit_transform(calm_rage)

        # from experimentation ranges 15%-35% (sucks) depending on how many
        # tfidf features and how many resulting PCA features
        print("    PCA explained variance ratio: {:2.2f}%".format(np.sum(self.pca.explained_variance_ratio_) * 100))

        # clean up since this stuff is memory intensive
        del temp_calm, temp_rage

        # calm_rage

        print("Tesseract.run_PCA(): complete")

    def run_TSNE(self):
        '''
        Fit the TSNE feature extractor and try to squash to 3 dimensions
        '''

        print(type(self.pca_result))
        # print("pca_result.shape", self.pca_result.shape)

        print("Tesseract.run_TSNE(): starting")
        print("    TSNE fit starting...\n")
        self.tsne_result = self.tsne.fit_transform(self.pca_result)
        print("    TSNE fit completed\n")
        print("    carv out raw data to be plotted")

        x_dim = self.tsne_result[:, 0]
        y_dim = self.tsne_result[:, 1]
        z_dim = self.tsne_result[:, 2]

        # slice and dice
        self.calm_x = x_dim[self.RAGE_SIZE:-1]
        self.calm_y = y_dim[self.RAGE_SIZE:-1]
        self.calm_z = z_dim[self.RAGE_SIZE:-1]

        self.rage_x = x_dim[:self.RAGE_SIZE]
        self.rage_y = y_dim[:self.RAGE_SIZE]
        self.rage_z = z_dim[:self.RAGE_SIZE]

        # clean up
        del x_dim, y_dim, z_dim

        print("    data carved into neat dimensional slices")
        print("Tesseract.run_TSNE(): complete\n")

    def cleanup(self):
        '''
        try to prevent memory leaks by deleting all big items used
        before this object itself is deleted
        '''

        del self.rage_df
        del self.calm_df

        # placeholders for the data

        # comment strings:
        del self.train_data
        del self.train_text
        del self.test_text
        del self.all_text

        # tfidf vectors:
        del self.rage_features   # vectorized words that are toxic
        del self.calm_features   # random selection of non-toxic words
        # pointless to do test word feature vectorization since no predictions will
        # be done

        # tfidf vectors by classification
        del self.rage_tfidf_vecs
        del self.calm_tfidf_vecs

        # tfidf vectorizing 'model'
        del self.word_vectorizer

        # PCA 'model'
        # n_components indicates how many features will be fed into TSNE later
        del self.pca

        # data resulting from PCA
        del self.pca_result

        # TSNE 'model'
        # self.tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=tsne_iter)
        del self.tsne

        # data resulting from TSNE
        del self.tsne_result

        # Data that will be fed into imaging:
        del self.calm_x
        del self.calm_y
        del self.calm_z
        del self.rage_x
        del self.rage_y
        del self.rage_z

        del self.ffm

    def make_plots(self):
        '''
        actually make the plots
        '''

        for aaa, ang in enumerate(range(0, 360, self.step_size)):

            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(3, 4)  # allow the merging of plots
        #     gs.update(left=-0.5, right=0.05, wspace=0.0, hspace=0.0)
            ax = plt.subplot(gs[:, 0:2], projection='3d')

        #     fig, ax = plt.subplots(figsize=(10,10))

            COUNT = 5000

            ax.scatter(self.calm_x[:COUNT],
                       self.calm_y[:COUNT],
                       self.calm_z[:COUNT],
                       zdir='z',
                       cmap="viridis",
                       c=self.calm_z[:COUNT],
                       s=75,
                       label="Non-Toxic",
                       alpha=0.5)

            # ax.set_xlim(-3.0, 3.0)
            # ax.set_ylim(-4, 3)
            # ax.set_zlim(-3, 3)
            ax.w_xaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
            ax.w_yaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
            ax.w_zaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))

        #     ax.set_facecolor("black")

            ax.set_axis_off()
            ax.autoscale_view(tight=True)
            ax.view_init(15 + 15 * np.sin(ang * np.pi / 180), ang)
            ax.set_title("Non-Toxic Comment")

            # help organize different characteristics
            class_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

            for class_num, _class in enumerate(class_names):
                _colors = ["Greens_r", "summer", "spring", "afmhot", "copper", "cool"]

                sizes = [20, 40, 20, 40, 20, 40]
                alphas = [0.4, 0.5, 0.4, 0.4, 0.4, 0.5]

                COUNT = 5000

                ################################
                ### Subplot maneuvering  #######
                ################################

                # could do an alg or could be like hecka lazy
                plt_rows = [0, 0, 1, 1, 2, 2]
                plt_cols = [2, 3, 2, 3, 2, 3]

                ax = plt.subplot(gs[plt_rows[class_num], plt_cols[class_num]], projection='3d')

                idx = class_num
                idy = idx + 2

                sampler = np.random.permutation(COUNT)
                class_matches = self.rage_df.iloc[:, idy]==1   # big list of booleans to filter 8 columns to the topic wanted

                temp_x = self.rage_x[self.rage_df.iloc[:, idy]==1]
                temp_y = self.rage_y[self.rage_df.iloc[:, idy]==1]
                temp_z = self.rage_z[self.rage_df.iloc[:, idy]==1]
            #     print("shape of temp_x", temp_x.shape)

                # if the size of the matches is greater than the count
                # then subsample using the sampler
                if temp_x.shape[0] > COUNT:
                    temp_x = np.take(temp_x, sampler)
                    temp_y = np.take(temp_y, sampler)
                    temp_z = np.take(temp_z, sampler)

                v_offset = np.min(temp_z)

                c_off = 1.0
                _c = (temp_z - v_offset)/2 + c_off

            #     print("max c", np.max(_c))

                _c[np.argmax(temp_z)] = 7 + c_off   # set a nail to stretch the cmap to the look that we want
                _c[np.argmin(temp_z)] = 0   # set a lower one too

                ax.scatter(temp_x,
                           temp_y,
                           temp_z,
                           zdir='z',
                           cmap=_colors[idx],
                           c=_c,
                           s=sizes[idx],
                           alpha=alphas[idx])

                ax.set_title(class_names[idx])
                # ax.set_xlim(-3.0, 3.0)
                # ax.set_ylim(-4, 3)
                # ax.set_zlim(-3, 3)
            #     ax.w_xaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
            #     ax.w_yaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
            #     ax.w_zaxis.set_pane_color((0.98, 0.98, 0.98, 1.0))
                ax.set_axis_off()
                ax.autoscale_view(tight=True)

        #         ax.set_facecolor("black")     # for spacing debugging

                ax.view_init(15 + 15 * np.sin(ang * np.pi / 180), ang)

            plt.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)

            w, h = fig.canvas.get_width_height()
            fig.canvas.draw()

            new_buffer = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).astype(np.int16).reshape(h, w, -1)

            self.ffm.add_frame(new_buffer)
            plt.close()

        self.ffm.close()


if __name__ == "__main__":
    time.sleep(2)

    # iters = [300, 500, 700, 900, 1100, 1300, 1500, 1700, 2000]
    lrs = [1200, 1400, 1600, 1800, 2000]
    for lr in lrs:
        tessa = Tesseract(
            tfidf_feature_count=25000,
            tsne_feature_count=1000,
            tsne_iter=4000,
            tsne_lr=lr)

        tessa.run_me()

        time.sleep(2)

        del tessa

        time.sleep(2)
