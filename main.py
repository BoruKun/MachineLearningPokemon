"""
https://www.kaggle.com/rounakbanik/pokemon

Постройте классификатор, отвечающий на вопрос 'является ли покемон легендарным?'.

Наберите команду из N покемонов, максимизирующую причиняемый урон.
"""

import pandas

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    # Load our data
    data = pandas.read_csv("pokemon.csv")

    # Organize our data
    X = data.drop(columns = ["is_legendary"])
    y = data["is_legendary"]

    # Split our data to test and train samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3, random_state=42)

    # Handle features
    integer_values = X.select_dtypes(include=['integer']).columns.to_list()

    # Initialize our classifier
    gnb = GaussianNB()

    # Train
    model = gnb.fit(X_train[integer_values], y_train)

    # Predict
    predictions = gnb.predict(X_test[integer_values])
    print(predictions)


if __name__ == "__main__":
    main()
