Python project that uses web scraping and machine learning techniques to classify song lyrics to their respective artists. By providing links to the web pages of two different artists and a specific song, the classifier predicts which artist is more likely to have written the song.

### Features

- Web Scraping: The project utilizes the `requests` library and BeautifulSoup to scrape the lyrics from the provided web pages.
- Text Processing: The lyrics are processed and cleaned using various techniques, including tokenization and stop word removal.
- Feature Extraction: The `CountVectorizer` and `TfidfTransformer` from the scikit-learn library are used to convert the lyrics into numerical feature vectors.
- Machine Learning: The classifier employs the Multinomial Naive Bayes algorithm from scikit-learn to train a model based on the extracted features.
- Prediction: Given a new song's lyrics, the model predicts the artist who is more likely to have written the song and provides the prediction probability.

### Usage

To use the Lyrics Artist Classifier, follow these steps:

1. Provide the URLs of the web pages for two different artists and the URL of a specific song.
2. The project will scrape the lyrics for the provided artists and song, clean and process the text, and train a machine learning model.
3. Finally, the model will predict the artist who is more likely to have written the song and display the prediction along with the prediction probability.

Example usage:

```python
artist(
    'https://www.lyrics.com/artist/Sex-Pistols/2137844906', 
    'https://www.lyrics.com/artist/The-Clash/3913', 
    'https://www.lyrics.com/lyric/6293626/The+Clash/I%27m+So+Bored+with+the+U.S.A.'
)
```

```python
artist(
    'https://www.lyrics.com/artist/Metallica/4906', 
    'https://www.lyrics.com/artist/Slayer/5453', 
    'https://www.lyrics.com/lyric/27233912/Slayer/God+Send+Death'
)
```

### Requirements

The following Python libraries are required to run the project:

- `pandas`
- `requests`
- `beautifulsoup4`
- `scikit-learn`

Please make sure to install these dependencies before running the code.

### Note

This project is for educational purposes only and serves as a demonstration of web scraping and text classification techniques. The accuracy of the predictions may vary depending on the quality and diversity of the training data.

Feel free to explore and modify the code according to your needs!

If you have any questions or suggestions, feel free to reach out.

Enjoy classifying song lyrics to their respective artists with the Lyrics Artist Classifier!
