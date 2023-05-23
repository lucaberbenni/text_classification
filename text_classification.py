import pandas as pd

import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB

def save_links(url):
    """

    import artist web page 

    """
    headers = {'User-agent': 'Mozilla/5.0 (X11; Linux i686; rv:2.0b10) Gecko/20100101 Firefox/4.0b10'}
    response = requests.get(url = url, headers=headers)
    html = response.text
    """

    convert page in html, extract artist name and links

    """
    artist_page = BeautifulSoup(html, 'html.parser')
    artist_name = artist_page.h1.get_text()
    links_list = artist_page.find(class_='tdata-ext').find_all('a', href=True)
    links_list = [link['href'] for link in links_list if link['href'].startswith('/lyric')]
    """

    save links list as a .txt file

    """
    with open(f'data/{artist_name}_links.txt', 'w') as f:
        for link in links_list:
            f.write("%s\n" % link)
    """

    return links list

    """
    return links_list

def lyrics_list(links_list):
    headers = {'User-agent': 'Mozilla/5.0 (X11; Linux i686; rv:2.0b10) Gecko/20100101 Firefox/4.0b10'}
    """
    
    extract, clean and save in a .txt file 100 lyrics

    """
    lyrics_list = []
    count = 0
    for link in links_list:
        try:
            lyric_raw = requests.get(url = 'https://www.lyrics.com' + link, headers=headers).text
            lyric_html = BeautifulSoup(lyric_raw, 'html.parser')

            artist = lyric_html.find('h3', class_ = 'lyric-artist').get_text()
            lyric = lyric_html.find('pre', id = 'lyric-body-text').get_text()

            lyrics_list.append(lyric)
            with open(f'data/{artist}_lyrics.txt', 'w')as f:
                for lyric in lyrics_list:
                    f.write("%s\n" % lyric)

            count += 1
            if count == 100:
                break

        except:
            (AttributeError, requests.exceptions.RequestException, FileNotFoundError)
    """
    
    return lyrics list
    
    """
    return lyrics_list
    
def artist_finder(list_1, 
                  list_2, 
                  link_1, 
                  link_2, 
                  song):
    """
    
    create and transform a dataframe with 2 artist lyrics list
    
    """
    lists = list_1 + list_2
    vectorizer = CountVectorizer(stop_words = 'english')
    transformer = make_pipeline(vectorizer, TfidfTransformer())
    df = transformer.fit_transform(lists)
    df_trans = pd.DataFrame(df.todense(), columns= vectorizer.get_feature_names_out())
    """
    
    extract the artists names
    
    """
    headers = {'User-agent': 'Mozilla/5.0 (X11; Linux i686; rv:2.0b10) Gecko/20100101 Firefox/4.0b10'}
    
    response_1 = requests.get(url = link_1, headers=headers)
    html_1 = response_1.text
    artist_page_1 = BeautifulSoup(html_1, 'html.parser')
    artist_name_1 = artist_page_1.h1.get_text()
    
    response_2 = requests.get(url = link_2, headers=headers)
    html_2 = response_2.text
    artist_page_2 = BeautifulSoup(html_2, 'html.parser')
    artist_name_2 = artist_page_2.h1.get_text()
    """
    
    split the dataset and train a model
    
    """
    X = df_trans.values
    y = [artist_name_1] * 100 + [artist_name_2] * 100

    m = MultinomialNB(force_alpha=True, alpha=0.6)
    m.fit(X, y)
    """
    
    extract another song lyrics
    
    """
    response_song = requests.get(url = song, headers = headers)
    song_html = response_song.text

    song_text = BeautifulSoup(song_html, 'html.parser')
    song_text_clean = song_text.find('pre', id = 'lyric-body-text').get_text()

    song_clean = []
    song_clean.append(song_text_clean)
    song_trans = transformer.transform(song_clean)
    """
    
    calculate and print prediction and prediction probability
    
    """
    song_predict = m.predict(song_trans)
    song_proba = m.predict_proba(song_trans)

    print(song_predict)
    print(song_proba)

def artist(artist_1, 
           artist_2, 
           song):
    """
    
    function tree
    
    """
    artist_finder(lyrics_list(save_links(artist_1)), 
                  lyrics_list(save_links(artist_2)), 
                  link_1 = artist_1, 
                  link_2 = artist_2, 
                  song = song)

print(artist(
    'https://www.lyrics.com/artist/Sex-Pistols/2137844906', 
    'https://www.lyrics.com/artist/The-Clash/3913', 
    'https://www.lyrics.com/lyric/6293626/The+Clash/I%27m+So+Bored+with+the+U.S.A.'
))

print(artist(
    'https://www.lyrics.com/artist/Metallica/4906', 
    'https://www.lyrics.com/artist/Slayer/5453', 
    'https://www.lyrics.com/lyric/27233912/Slayer/God+Send+Death'
))

print(artist(
    'https://www.lyrics.com/artist/Sonic-Youth/5474', 
    'https://www.lyrics.com/artist/Joy-Division/71273', 
    'https://www.lyrics.com/lyric/37320104/Joy+Division/Only+Mistake+%5BLive%3B+Previously+Unreleased+Track%5D'
))

print(artist(
    'https://www.lyrics.com/artist/2Pac/50051', 
    'https://www.lyrics.com/artist/The-Notorious-B.I.G./44889', 
    'https://www.lyrics.com/lyric/27133816/2Pac/World+Wide+Mob+Figgaz'
))