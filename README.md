# text_classification
Scrape through lyric.com and predict artist based on lyrics.

1. The `save_links(url)` function takes a URL of an artist's web page, scrapes the page, extracts the artist's name and the links to their lyrics, and saves the links as a text file.

2. The `lyrics_list(links_list)` function takes a list of links to lyrics, scrapes the lyrics from the links, cleans them, and saves them as a text file.

3. The `artist_finder(list_1, list_2, link_1, link_2, song)` function takes two lists of lyrics from different artists, transforms them into a dataframe using CountVectorizer and TfidfTransformer, extracts the artists' names from their web pages, splits the dataset into features (lyrics) and target (artist names), trains a Multinomial Naive Bayes model on the dataset, and finally predicts the artist of a given song and prints the prediction and prediction probabilities.

4. The `artist(artist_1, artist_2, song)` function takes the URLs of two artists' web pages and a URL of a song, and calls the `artist_finder` function with the scraped lyrics from the two artists and the given song.

The code then provides a few examples of calling the `artist` function with different artists and songs and prints the predicted artist and prediction probabilities.

Please note that to run this code, you would need to have the necessary libraries (such as `requests`, `beautifulsoup4`, and `scikit-learn`) installed and provide valid URLs for the artists and songs you want to compare.
