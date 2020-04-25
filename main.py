"""
https://www.kaggle.com/rounakbanik/pokemon

Постройте классификатор, отвечающий на вопрос 'является ли покемон легендарным?'.

Наберите команду из N покемонов, максимизирующую причиняемый урон.
"""

import pandas
import sklearn

# Load our data
data = pandas.read_csv("pokemon.csv", delimeter=",")
