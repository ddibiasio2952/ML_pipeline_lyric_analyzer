# Spotify Song & Lyrics Analysis Pipeline

This project explores a dataset of Spotify songs and lyrics using natural language processing techniques as a Rhode Island College Natural Language Processing Capstone Project. 

It applies sentiment and emotion analysis to examine the emotional characteristics of music across different genres and artists.

The project also includes a playlist generator that recommends songs based on a selected emotion, genre, or combination of both.

## Project Overview

The repository contains a Google Colab notebook covering the complete analysis workflow, including:

* Loading and cleaning the dataset
* Preprocessing and filtering lyrics
* Sentiment analysis with NLTK VADER
* Sentiment classification with TF-IDF and Logistic Regression
* Emotion classification with a pretrained DistilBERT model
* Genre- and artist-level analysis
* Data visualization with Matplotlib and Seaborn
* Emotion- and genre-based playlist generation
* Playlist export to CSV

## Dataset

This project uses a subset of the **Spotify Lyrics Dataset**, published by Kaggle user **Eva Botoș**.

* **Dataset:** [Spotify Lyrics Dataset](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset)
* **Platform:** Kaggle
* **Local project file:** `lyrics_10k.csv`
* **Project sample size:** Approximately 10,000 songs

The dataset contains Spotify songs and associated information such as:

* Song title
* Song ID
* Artist
* Artist ID
* Genre
* Explicit-content indicator
* Lyrics

Songs with missing or insufficient lyrical content are removed during preprocessing.

> The dataset is not necessarily included in this repository. Download it from the [Spotify Lyrics Dataset page on Kaggle](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset) and review its license and usage terms before using or redistributing it.

## Project Files

- [View the Analysis Notebook](https://colab.research.google.com/drive/1maxIHJwWVfwWQPamCIL_hsDvUWZqHYvP)
- [Download the original dataset from Kaggle](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset)

## Methodology

### 1. Data Loading and Preprocessing

The dataset is loaded into a Pandas DataFrame and prepared for analysis by:

* Removing unnecessary columns
* Handling missing values
* Standardizing artist and genre entries
* Removing malformed artist values
* Calculating lyric token counts
* Filtering songs with limited lyrical content
* Truncating lyrics when required by the transformer model

These steps improve data consistency and help ensure that the lyrics contain enough information for meaningful analysis.

### 2. Sentiment Analysis with NLTK VADER

The project uses NLTK's VADER (**Valence Aware Dictionary and Sentiment Reasoner**) to calculate sentiment scores for each song.

VADER produces four scores:

* Negative
* Neutral
* Positive
* Compound

The compound score represents the overall sentiment of the lyrics. It is converted into a categorical sentiment label such as:

* `positive`
* `neutral`
* `negative`

### 3. Sentiment Classification

A supervised machine-learning model is trained to predict the VADER-generated sentiment labels.

The classification workflow includes:

1. Converting lyrics into numerical features with TF-IDF vectorization
2. Splitting the data into training and testing sets
3. Training a Logistic Regression classifier
4. Evaluating the model with accuracy, precision, recall, F1-score, and a confusion matrix

The Logistic Regression model achieved approximately **81% accuracy** when predicting sentiment labels.

### 4. Emotion Analysis with DistilBERT

For more detailed emotional classification, the project uses the pretrained [`DistilBERT-base-uncased-emotion`](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) transformer model.

The model predicts a dominant emotion for each song, including:

* Joy
* Anger
* Sadness
* Love
* Fear
* Surprise

Because transformer models have a maximum input length, longer lyrics are truncated before classification.

### 5. Exploratory Data Analysis and Visualization

Matplotlib and Seaborn are used to visualize patterns in the dataset.

The notebook includes visualizations for:

* Overall sentiment distribution
* Overall emotion distribution
* Average sentiment by genre
* Emotion frequency by genre
* Artists with the highest average sentiment
* Artists with the lowest average sentiment
* Genre and emotion relationships
* Sentiment-classification performance

Stacked bar charts are also used to compare the relative frequency of emotions across selected genres.

### 6. Spotify Playlist Generator

The project includes a Python function that generates a 20-song playlist based on user-selected criteria.

Playlists can be filtered by:

* Emotion
* Genre
* Both emotion and genre

A second function exports the generated playlist to a CSV file. The exported file can then be used with a compatible playlist-import service to create a Spotify playlist.

Example:

```python
playlist = generate_playlist(
    dataframe=df,
    emotion="joy",
    genre="pop",
    playlist_size=20
)
```

Export the playlist:

```python
export_playlist(
    playlist,
    filename="joy_pop_playlist.csv"
)
```

## Key Findings

* The Logistic Regression model achieved approximately **81% accuracy** when classifying lyrics by sentiment.
* The model performed especially well when identifying positive lyrics, achieving approximately **91% recall** for the positive class.
* The emotion model provided more specific emotional categories than VADER's general positive, negative, and neutral labels.
* Sentiment and emotion distributions varied across genres, demonstrating that different genres can have distinct lyrical profiles.
* Artist-level analysis revealed differences in average lyrical sentiment, suggesting that some artists maintain recognizable emotional patterns across their songs.

## How to Use

### 1. Download the Dataset

Download the source data from the [Spotify Lyrics Dataset on Kaggle](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset).

A Kaggle account may be required to download the dataset.

Prepare or select the approximately 10,000-song subset used by the notebook and save it as:

```text
lyrics_10k.csv
```

### 2. Open the Notebook

Download the repository and open the `.ipynb` notebook in [Google Colab](https://colab.research.google.com/).

You can also upload the notebook directly to Colab.

### 3. Upload the Dataset

Upload `lyrics_10k.csv` to the Colab environment.

The file can typically be placed at:

```text
/content/lyrics_10k.csv
```

If the dataset is stored in Google Drive, mount your Drive and update the notebook's file path accordingly.

### 4. Install the Dependencies

Run the following command in a Colab cell:

```bash
pip install pandas nltk seaborn scikit-learn transformers matplotlib torch
```

The Python `re` module is part of the standard library and does not need to be installed separately.

### 5. Run the Notebook

Execute the notebook cells sequentially.

The notebook will:

1. Load and clean the data
2. Analyze lyrical sentiment
3. Train and evaluate the sentiment classifier
4. Predict dominant emotions
5. Generate visualizations
6. Create and export custom playlists

### 6. Generate a Playlist

Run the playlist-generator section and provide an emotion, genre, or both.

The resulting playlist can be exported as a CSV file for use outside the notebook.

## Dependencies

The project uses the following Python libraries:

* [Pandas](https://pandas.pydata.org/)
* [NLTK](https://www.nltk.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

It also uses Python's built-in `re` module for regular-expression-based text cleaning.

## Technologies Used

* Python
* Google Colab
* Pandas
* NLTK VADER
* TF-IDF
* Logistic Regression
* Hugging Face Transformers
* DistilBERT
* Matplotlib
* Seaborn

## Limitations

* Song lyrics may contain incomplete, duplicated, or incorrectly formatted entries.
* VADER was primarily designed for general-purpose text and may not fully capture figurative or poetic language.
* Transformer input-length limits require longer lyrics to be truncated.
* The predicted emotion represents the model's interpretation of the lyrics and may not match the listener's experience.
* Genre labels are not always standardized, and individual songs may belong to multiple genres.
* The generated playlists are exported as CSV files rather than being added directly to a Spotify account through the Spotify Web API.

## Future Improvements

Possible future additions include:

* Integrating the Spotify Web API to create playlists automatically
* Supporting multi-label emotion classification
* Comparing additional transformer models
* Incorporating audio features such as tempo, energy, and valence
* Building an interactive playlist-generation interface
* Expanding the analysis to a larger dataset
* Deploying the playlist generator as a web application

## Data Attribution

The lyrical data used by this project was obtained from:

> Eva Botoș. *Spotify Lyrics Dataset*. Kaggle.
> https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset

This project is not affiliated with or endorsed by Spotify or Kaggle.

## License

The source code and analysis in this repository are intended for educational and research purposes.

The dataset remains subject to the license and usage terms provided by its original publisher. Review the [Kaggle dataset page](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset) before redistributing the dataset or using it outside this project.

## Author
Daniel DiBiasio

[GitHub](https://github.com/ddibiasio2952/)
