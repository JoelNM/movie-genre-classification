# Movie Genre Classification

## Project Overview
This project uses machine learning to predict the genre of a movie based on its plot description. It uses TF-IDF to convert text into numerical features and Logistic Regression to classify the movie genre.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## Dataset
The project uses the IMDb Genre Classification dataset from Kaggle.

Dataset link: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

## How It Works
1. Load the movie dataset.
2. Clean and prepare the plot descriptions.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression model.
5. Evaluate the model using accuracy and a classification report.
6. Enter a new movie plot to predict its genre.

## Model Performance
The model achieved **58.2% accuracy** on the test split used during this experiment.

## How to Run
1. Install Python.
2. Install the required libraries:
   `pip install pandas scikit-learn`
3. Download and extract the dataset from Kaggle.
4. 4. Extract the downloaded dataset and place the `Genre Classification Dataset` folder in the same directory as `movie_genre_classifier.py`. Make sure `train_data.txt` is inside that folder.
5. Run:
   `python movie_genre_classifier.py`

## Project Status
The model is working, and new movie plot prediction is implemented.

## Author
JoelNM
