""" Модуль, отвечающий за парсинг метаданных """

import json
import logging

METADATA_JSON_PATH = 'metadata.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_metadata():
    """ Загружает метаданные JSON
    Returns:
        JSON containing the metadata
    """
    global METADATA_JSON_PATH

    with open(METADATA_JSON_PATH, 'r') as metadata_file:
        metadata_json = json.load(metadata_file)

    return metadata_json


def get_last_patch():
    """ Получает информацию о последнем патче
    Returns:
        dictionary containing info of the last patch
    """
    metadata_json = _load_metadata()
    return metadata_json['patches'][0]


def get_patch(patch_name):
    """ Получает информацию о патче с именем patch_name
    Args:
        patch_name: patch identifier
    Returns:
        dictionary containing info of the wanted patch
    """
    global logger
    metadata_json = _load_metadata()

    for entry in metadata_json['patches']:
        if entry['patch_name'] == patch_name:
            logger.info('Found patch %s', patch_name)
            return entry

    logger.error('Could not find patch %s', patch_name)


def get_hero_dict():
    """ Возвращает словарь, где ключом является идентификатор героя, а значением — имя героя.
    Returns:
        dictionary of (hero_ID, name)
    """
    metadata_json = _load_metadata()
    hero_dict = dict()

    for entry in metadata_json['heroes']:
        hero_dict[entry['id']] = entry['name']

    return hero_dict
