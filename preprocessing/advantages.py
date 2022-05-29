""" Модуль, отвечающий за расчет преимуществ по набору данных игр. """
import numpy as np

from tools.metadata import get_last_patch


def _update_dicts(game, synergy, counter):
    """ Обновляет синергию и встречные игры, учитывая игру, указанную в качестве входных данных.
    Args:
        game: row of a mined pandas DataFrame
        synergy: synergy matrix
        counter: counter matrix
    Returns:
        None
    """
    radiant_win, radiant_heroes, dire_heroes = game[1], game[2], game[3]

    radiant_heroes = list(map(int, radiant_heroes.split(',')))
    dire_heroes = list(map(int, dire_heroes.split(',')))
    if len (radiant_heroes) == 5 & len (dire_heroes) == 5:
        for i in list(range(5)):
            for j in list(range(5)):
                if i != j:
                    synergy['games'][radiant_heroes[i] - 1, radiant_heroes[j] - 1] += 1
                    synergy['games'][dire_heroes[i] - 1, dire_heroes[j] - 1] += 1

                    if radiant_win:
                        synergy['wins'][radiant_heroes[i] - 1, radiant_heroes[j] - 1] += 1
                    else:
                        synergy['wins'][dire_heroes[i] - 1, dire_heroes[j] - 1] += 1

                counter['games'][radiant_heroes[i] - 1, dire_heroes[j] - 1] += 1
                counter['games'][dire_heroes[i] - 1, radiant_heroes[j] - 1] += 1

                if radiant_win:
                    counter['wins'][radiant_heroes[i] - 1, dire_heroes[j] - 1] += 1
                else:
                    counter['wins'][dire_heroes[i] - 1, radiant_heroes[j] - 1] += 1


def _compute_winrates(synergy, counter, heroes_released):
    """ Рассчитывает винрейт каждой комбинации выпущенных героев с точки зрения синергии и контратаки.
    Результаты сохраняются в словарях синергии и счетчиков с помощью ключа 'winrate'.
    Args:
        synergy: synergy matrix
        counter: counter matrix
        heroes_released: number of heroes released until current patch
    Returns:
        None
    """
    for i in list(range(heroes_released)):
        for j in list(range(heroes_released)):
            if i != j and i != 23 and j != 23:
                if synergy['games'][i, j] != 0:
                    synergy['winrate'][i, j] = synergy['wins'][i, j] / \
                                               float(synergy['games'][i, j])

                if counter['games'][i, j] != 0:
                    counter['winrate'][i, j] = counter['wins'][i, j] / \
                                               float(counter['games'][i, j])


def _adv_synergy(winrate_together, winrate_hero1, winrate_hero2):
    """ Учитывая винрейт двух героев, сыгранных по отдельности и вместе, возвращает счет,
    отражающий преимущество героев сыгранных вместе.
    Было предпринято много попыток рассчитать это преимущество,
    но простой винрейт при совместной игре работает лучше всего.
    Args:
        winrate_together: winrate when both heroes are played in the same team
        winrate_hero1: general winrate of hero1
        winrate_hero2: general winrate of hero2
    Returns:
        advantage computed using the winrates given as input
    """
    return winrate_together


def _adv_counter(winrate_together, winrate_hero1, winrate_hero2):
    """ Учитывая винрейт одного героя при игре против другого героя и их отдельные винрейты, возвращает счет,
    представляющий преимущество, когда герой1 выбран против героя2.
    Было предпринято много попыток рассчитать этот показатель преимущества,
    но простой винрейт при игре друг против друга, кажется, работает лучше всего..
    Args:
        winrate_together: winrate when hero1 is picked against hero2
        winrate_hero1: general winrate of hero1
        winrate_hero2: general winrate of hero2
    Returns:
        advantage computed using the winrates given as input
    """
    return winrate_together


def _calculate_advantages(synergy, counter, heroes_released):
    """ Рассчет базового процента побед для каждого героя и использование его для расчета преимуществ
    Args:
        synergy: synergy matrix
        counter: counter matrix
        heroes_released: number of heroes released in the current patch
    Returns:
        synergy matrix, counter_matrix using advantages
    """
    synergies = np.zeros((heroes_released, heroes_released))
    counters = np.zeros((heroes_released, heroes_released))

    base_winrate = np.zeros(heroes_released)

    for i in list(range(heroes_released)):
        if i != 23:
            base_winrate[i] = np.sum(synergy['wins'][i]) / np.sum(synergy['games'][i])

    for i in list(range(heroes_released)):
        for j in list(range(heroes_released)):
            if i != j and i != 23 and j != 23:
                if synergy['games'][i, j] > 0:
                    synergies[i, j] = _adv_synergy(synergy['winrate'][i, j],
                                                   base_winrate[i],
                                                   base_winrate[j])
                else:
                    synergies[i, j] = 0

                if counter['games'][i, j] > 0:
                    counters[i, j] = _adv_counter(counter['winrate'][i, j],
                                                  base_winrate[i],
                                                  base_winrate[j])
                else:
                    counters[i, j] = 0

    return synergies, counters


def compute_advantages(dataset_df):
    """ Получение в качестве входных данных DataFrame данных pandas,
    рассчет преимуществ и сохранение их в словарях синергии и противодействия.
    Результаты сохраняются в файлах для более удобного последующего использования.
    Args:
        dataset_df: pandas DataFrame containing the games to be analyzed
    Returns:
        synergy matrix and counter matrix using advantages
    """

    last_patch_info = get_last_patch()
    heroes_released = last_patch_info['heroes_released']

    synergy = dict()
    synergy['wins'] = np.zeros((heroes_released, heroes_released))
    synergy['games'] = np.zeros((heroes_released, heroes_released))
    synergy['winrate'] = np.zeros((heroes_released, heroes_released))

    counter = dict()
    counter['wins'] = np.zeros((heroes_released, heroes_released))
    counter['games'] = np.zeros((heroes_released, heroes_released))
    counter['winrate'] = np.zeros((heroes_released, heroes_released))

    dataset_np = dataset_df.values

    for row in dataset_np:
        _update_dicts(row, synergy, counter)

    _compute_winrates(synergy, counter, heroes_released)

    synergy_matrix, counter_matrix = _calculate_advantages(synergy, counter, heroes_released)

    # uncomment only for overwriting precomputed advantages - NOT RECOMMENDED
    # np.savetxt('pretrained/synergies_all.csv', synergy_matrix)
    # np.savetxt('pretrained/counters_all.csv', counter_matrix)

    return [synergy_matrix, counter_matrix]