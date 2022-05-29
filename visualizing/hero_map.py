import collections
import logging
import math
import random

import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from tools.metadata import get_hero_dict, get_last_patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_index = 0


def _build_vocabulary(words, vocabulary_size):
    """ Создает словарь, и подсчитывает появление каждого слова.
     В этом контексте каждое слово представляет собой индекс героя, приведенный к строке, например. Антимаг -> "1"

    Args:
        words: list of strings representing the corpus
        vocabulary_size: number of words to be evaluated (the vocabulary will contain only this
            number of words, even if there are more unique words in the corpus)
    Returns:
        data: list of indices obtained by mapping the words' indices to the corpus
        count: list of [word, appearances] for the corpus
        dictionary: the vocabulary (the key is the word, and the value is the appearances)
        reverse_dictionary: the reversed vocabulary (the key are the appearances and the values
            are the words)
    """

    # create dictionary with the most common heroes
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            # the word is unknown
            index = 0
            unk_count = unk_count + 1
        data.append(index)

    count[0][1] = unk_count

    # save the dictionary's reversed version for later usage
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def _generate_batch(data, batch_size, num_skips, window_size):
    """ Генерирует пакет данных для использования в обучении, используя разновидность word2vec skip-gram.

    Args:
        data: list of indices obtained by mapping the words' indices to the corpus
        batch_size: number of samples to be used in a batch
        num_skips: number of skips hyperparameter of word2vec
        window_size: window size hyperparameter of word2vec
    Returns:
        batch: batch of data to be used in training
        labels: labels of each sample in the batch
    """

    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * window_size + 1
    buffer = collections.deque(maxlen=span)

    for _ in list(range(span)):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in list(range(batch_size // num_skips)):
        # target label at the center of the buffer
        target = window_size
        targets_to_avoid = [window_size]

        for j in list(range(num_skips)):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[window_size]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


def _train_word2vec(data, batch_size, vocabulary_size, embedding_size, neg_samples, window_size, num_steps,
                    reverse_dictionary, heroes_dict):
    """ Учитывая входные данные и гиперпараметры, обучение набором данных игр, используя word2vec с особенностями skip-gram.

     Args:
        data: list of indices obtained by mapping the words' indices to the corpus
        batch_size: number of samples to be used in a batch
        vocabulary_size: number of words to be evaluated (the vocabulary will contain only this
            number of words, even if there are more unique words in the corpus)
        embedding_size: number of dimensions when creating word embeddings
        neg_samples: word2vec negative samples hyperparameter
        window_size: word2vec window size hyperparameter
        num_steps: number of steps to train for at (at least 10k should be fine)
        window_size: window size hyperparameter of
        reverse_dictionary: the reversed vocabulary (the key are the appearances and the values
            are the words)
        heroes_dict: dictionary that maps the hero's ID to its name
    Returns:
        final_embeddings: np.array of (samples, embedding_size) dimension corresponding to each
            hero's embeddings
    """

    valid_size = 15
    valid_examples = np.random.randint(0, vocabulary_size, valid_size)
    err_skip = 0
    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):
        train_dataset = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        embeddings = tf.Variable(tf.random.uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(tf.random.truncated_normal([vocabulary_size, embedding_size],
                                                          stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, train_labels, embed, neg_samples, vocabulary_size))

        optimizer = tf.compat.v1.train.AdagradOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.compat.v1.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    with tf.compat.v1.Session(graph=graph) as session:
        session.run(tf.compat.v1.global_variables_initializer())
        logger.info('Initialized graph')
        average_loss = 0

        for step in list(range(num_steps)):
            batch_data, batch_labels = _generate_batch(data,
                                                       batch_size,
                                                       2 * window_size,
                                                       window_size)

            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, new_loss = session.run([optimizer, loss], feed_dict=feed_dict)

            average_loss += new_loss

            # print the loss every 2k steps
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                logger.info('Средняя потеря на шаге %d: %f', step, average_loss)
                average_loss = 0

            # print a sample of similarities between heroes every 10k steps
            if step % 10000 == 0:
                sim = similarity.eval()

                for i in range(valid_size):
                    try:
                        valid_word = reverse_dictionary[valid_examples[i]]
                        # ignore unknown and padding tokens
                        if valid_word != 'UNK' and valid_word != 'PAD':
                            valid_word = heroes_dict[int(reverse_dictionary[valid_examples[i]])]

                        top_k = 8  # number of nearest neighbors to print
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Ближайший к %s:' % valid_word

                        for k in range(top_k):
                            try:
                                index = reverse_dictionary[nearest[k]]

                                if index != 'UNK' and index != 'PAD':
                                    close_word = heroes_dict[int(index)]
                                else:
                                    close_word = index
                                log = '%s %s,' % (log, close_word)
                            except:
                                err_skip+=1

                        logger.info(log)
                    except:
                        err_skip+=1
        final_embeddings = normalized_embeddings.eval()

    return final_embeddings


def _plot_similarities(embeddings, heroes_dict, reverse_dictionary, perplexity=20):
    """ Построение полученных ролей героев с использованием алгоритма TSNE в 2D-пространстве.
     Есть 4 предполагаемые роли: мид, керри, оффлейнер, поддержка, каждая категория содержит
     репрезентативный жестко закодированный герой, чтобы правильно определить роль каждого кластера.

    Args:
        embeddings: hero embeddings obtained after training
        heroes_dict: dictionary that maps the hero's ID to its name
        reverse_dictionary: the reversed vocabulary (the key are the appearances and the values
            are the words)
        perplexity: hyperparameter of TSNE (15-30 seems to work best)
    """

    # Reduce the embeddings to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    two_d_embeddings = tsne.fit_transform(embeddings)

    # Apply KMeans on the data in order to clusterize by role
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tsne.embedding_)

    labels = kmeans.labels_
    labels = labels[2:]

    x_vals = two_d_embeddings[:, 0]
    y_vals = two_d_embeddings[:, 1]

    number_of_heroes = len(heroes_dict.keys())
    names = number_of_heroes * ['']
    for i in list(range(number_of_heroes)):
        names[i] = heroes_dict[int(reverse_dictionary[i + 2])]

    x_vals = list(x_vals)
    y_vals = list(y_vals)

    # delete 'UNK' and 'PAD' when plotting
    del x_vals[1]
    del x_vals[0]
    del y_vals[1]
    del y_vals[0]

    traces = []
    for cluster in list(range(max(labels) + 1)):
        indices = []
        for i in list(range(len(labels))):
            if labels[i] == cluster and i < 123:
                indices.append(i)

        cluster_text = 'Mixed'

        heroes_in_cluster = [names[i] for i in indices]
        if 'Terrorblade' in heroes_in_cluster:
            cluster_text = 'Carry'
        elif 'Shadow Fiend' in heroes_in_cluster:
            cluster_text = 'Mid'
        elif 'Batrider' in heroes_in_cluster:
            cluster_text = 'Offlane'
        elif 'Dazzle' in heroes_in_cluster:
            cluster_text = 'Support'

        trace = go.Scatter(
            x=[x_vals[i] for i in indices],
            y=[y_vals[i] for i in indices],
            mode='markers+text',
            text=[names[i] for i in indices],
            name=cluster_text,
            textposition='top center'
        )

        traces.append(trace)

    layout = go.Layout(
        title='Hero map test',
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            dtick=True,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            dtick=True,
            ticks='',
            showticklabels=False
        )
    )

    data = traces
    figure = go.Figure(data=data, layout=layout)

    py.plot(figure, filename='heromap')


def plot_hero_map(csv_path, batch_size=128, embedding_size=25, window_size=2, neg_samples=64, num_steps=30001,
                  low_mmr=0, high_mmr=9000):
    """ Создает 2D-график героев на основе их сходства, полученного с помощью word2vec. Результат загружается в график.

    Args:
        csv_path: path to the training dataset csv
        batch_size: size of the batch to be used in training
        embedding_size: number of dimensions when creating word embeddings
        window_size: word2vec window size hyperparameter
        neg_samples: word2vec negative samples hyperparameter
        num_steps: number of steps to train for at (at least 10k should be fine)
        low_mmr: lower bound of the MMRs filtered for plotting
        high_mmr: upper bound of the MMRs filtered for plotting
    """
    patch = get_last_patch()
    heroes_dict = get_hero_dict()
    dataset = pd.read_csv(csv_path)

    # filter the games by MMR and transform the dataset to a numpy array
    dataset = dataset[(dataset.avg_mmr > low_mmr) & (dataset.avg_mmr < high_mmr)].values

    vocabulary_size = patch['heroes_released'] + 1
    words = []

    # create corpus by separating each team and adding padding
    for match in dataset:
        radiant_list = match[2].split(',')
        dire_list = match[3].split(',')

        words.extend(radiant_list)
        words.append('PAD')
        words.append('PAD')
        words.extend(dire_list)
        words.append('PAD')
        words.append('PAD')

    # create vocabulary using the corpus
    data, count, dictionary, reverse_dictionary = _build_vocabulary(words, vocabulary_size)

    logger.info('Самые распространенные герои (+UNK): %s', count[:5])
    logger.info('Образец данных: %s', data[:10])

    # free unused memory
    del words

    final_embeddings = _train_word2vec(data,
                                       batch_size,
                                       vocabulary_size,
                                       embedding_size,
                                       neg_samples,
                                       window_size,
                                       num_steps,
                                       reverse_dictionary,
                                       heroes_dict)

    _plot_similarities(final_embeddings, heroes_dict, reverse_dictionary)
