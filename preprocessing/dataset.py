import logging
import numpy as np
import pandas as pd

from preprocessing.advantages import compute_advantages
from preprocessing.augmenter import augment_with_advantages
from tools.metadata import get_last_patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _dataset_to_features(dataset_df, advantages=None):
    """ Преобразует добытый DataFrame pandas в матрицу признаков. Этот метод предполагает следующий формат DataFrame:
    columns: [match_id,radiant_win,radiant_team,dire_team,avg_mmr,num_mmr,game_mode,lobby_type]

    Args:
        dataset_df: pandas DataFrame to be transformed
        advantages: if given, the synergy and counters matrix are used to compute synergy and
            counter rating for every game which are appended to the feature matrix
    Returns:
        [X, y], where X is feature matrix, y are outputs
    """
    last_patch_info = get_last_patch()
    heroes_released = last_patch_info['heroes_released']
    synergy_matrix, counter_matrix = None, None

    if advantages:
        x_matrix = np.zeros((dataset_df.shape[0], 2 * heroes_released + 3))
        [synergy_matrix, counter_matrix] = advantages
    else:
        x_matrix = np.zeros((dataset_df.shape[0], 2 * heroes_released))

    y_matrix = np.zeros(dataset_df.shape[0])

    dataset_np = dataset_df.values

    for i, row in enumerate(dataset_np):
        radiant_win = row[1]
        radiant_heroes = list(map(int, row[2].split(',')))
        dire_heroes = list(map(int, row[3].split(',')))

        if len(radiant_heroes) == 5 & len(dire_heroes) == 5:
            for j in range(5):
                x_matrix[i, radiant_heroes[j] - 1] = 1
                x_matrix[i, dire_heroes[j] - 1 + heroes_released] = 1

                if advantages:
                    x_matrix[i, -3:] = augment_with_advantages(synergy_matrix,
                                                               counter_matrix,
                                                               radiant_heroes,
                                                               dire_heroes)

        y_matrix[i] = 1 if radiant_win else 0

    return [x_matrix, y_matrix]


def read_dataset(csv_path,
                 low_mmr=None,
                 high_mmr=None,
                 advantages=False):
    """ Читает Pandas DataFrame из csv_path, фильтрует игры между low_mmr и high_mmr, если указано,
    и добавляет функции синергии и счетчика

    Args:
        csv_path: path to read pandas DataFrame from
        low_mmr: lower MMR bound
        high_mmr: higher MMR bound
        advantages: if True, advantages are recalculated and saved to files, else it is read from
            they are read from files
    Returns:
        [feature_matrix, [synergy_matrix, counter_matrix]]
    """
    global logger
    dataset_df = pd.read_csv(csv_path)

    if low_mmr:
        dataset_df = dataset_df[dataset_df.avg_mmr > low_mmr]

    if high_mmr:
        dataset_df = dataset_df[dataset_df.avg_mmr < high_mmr]

    logger.info("Набор данных содержит %d игр", len(dataset_df))

    if advantages:
        logger.info("Вычисление преимуществ...")
        advantages_list = compute_advantages(dataset_df)
    else:
        logger.info("Загрузка преимуществ из файлов...")
        synergies = np.loadtxt('pretrained/synergies_all.csv')
        counters = np.loadtxt('pretrained/counters_all.csv')
        advantages_list = [synergies, counters]

    logger.info("Преобразование фрейма данных в карту объектов...")
    feature_map = _dataset_to_features(dataset_df, advantages=advantages_list)

    return [feature_map, advantages_list]
