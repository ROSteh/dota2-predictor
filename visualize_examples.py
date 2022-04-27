""" Module for showing examples of dota predictor's API usage """
import logging

from plotly.offline import init_notebook_mode

from preprocessing.dataset import read_dataset
from tools.metadata import get_last_patch, get_patch
from tools.miner import mine_data
from training.cross_validation import evaluate
from training.query import query
from visualizing.dataset_stats import pick_statistics, winrate_statistics, mmr_distribution
from visualizing.hero_combinations import plot_synergies, plot_counters
from visualizing.hero_map import plot_hero_map
from visualizing.learning_curve import plot_learning_curve


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_data_example1():
    # in order to use plotly (for most of these examples), you need to create an account and
    # configure your credentials; the plots will be saved to your online account
    # see https://plot.ly/python/getting-started/

    # plot learning curve for a loaded dataset with either matplotlib or plotly
    # subsets represents the number of points where the accuracies are evaluated
    # cv represents the number of folds for each point of the evaluation
    features, _ = read_dataset('706e_train_dataset.csv', low_mmr=3000, high_mmr=3500)
    #plot_learning_curve(features[0], features[1], subsets=20, cv=3, mmr=3250, tool='matplotlib')

    # the rest of the plots were implemented only for plotly because of their size

    # plot win rate statistics
    #winrate_statistics(features, '3000 - 3500')

    # plot pick rate statistics
   # pick_statistics(features, '3000 - 3500')

    # plot mmr distribution
   # mmr_distribution('706e_train_dataset.csv')

    # plot synergies and counters for hero combinations
    # they are loaded from the pretrained folder
  #  plot_synergies()
 #   plot_counters()

    # plot hero map containing the heroes grouped by the similarity of their role
    # the heroes are clustered by roles: support, offlane, mid, carry
    plot_hero_map('706e_train_dataset.csv')


def main():
    visualize_data_example1()


if __name__ == '__main__':
    main()