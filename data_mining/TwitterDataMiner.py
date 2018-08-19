import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.distance import vincenty
from twitter import *

# try:
#     to_unicode = to_unicode
# except NameError:
#     to_unicode = str

############## Control Parameters ################

get_search_location_data = False  # Turn on/off getting geolocation for data
update_tweet_database = True  # Turn on/off querying Twitter for more tweets 

############## Search Parameters #################

search_terms = ['canada', 'toronto', 'justintrudeau',
                'ontario', 'trudeau', 'canpoli', 'canadian',
                'OhCanada']
search_type = 'mixed'  # 'popular', 'recent' OR 'mixed'
include_retweets = True  # Include retweets in search
sleep_duration = 16*60  # 16*60 # seconds
search_radius = 250  # Radius from search location to include

##################################################

# Start geolocater
geolocater = Nominatim()

# Access Twitter API
ACCESS_TOKEN = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
ACCESS_SECRET = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
CONSUMER_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
CONSUMER_SECRET = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

twitter = Twitter(auth=oauth)


def list_to_str(terms_list):
    """ [list] -> str
    Converts a list into a query string
    :param terms_list: list
    :return search_string: str
    """
    search_string = ''
    first_term = True

    for term in terms_list:
        if first_term:
            if include_retweets:
                search_string = term
            else:
                search_string = "-RT OR " + term
            first_term = False
        else:
            search_string += " OR " + term

    return search_string


def unigrams(s):
    """ str -> [list]
    Converts a str into a list of unigrams
    :param s: str
    :return return_list: list
    """
    return_list = ''

    if len(s) == 0:
        return return_list

    s = s.lower()

    for i in range(len(s)):
        if 'a' <= s[i] <= 'z':  # Keep letters
            return_list += s[i]
        elif '0' <= s[i] <= '9':  # Keep numbers
            return_list += s[i]
        elif s[i] == ",":  # Replace commas with spaces
            return_list += ' '
        elif s[i] == " ":  # Keep spaces
            return_list += s[i]
        elif s[i] == '#':  # replace hashtags with spaces
            return_list += ' '

    return return_list.split()


def bigrams(s):
    """ str -> [list]
    Converts a string into a list of bigrams
    :param s: str
    :return bigram_list: list
    """
    str_list = unigrams(s)
    bigram_list = []
    for i in range(len(str_list) - 1):
        bigram_list.append(str_list[i] + " " + str_list[i + 1])
    return bigram_list


def get_location_data(df):
    """ pd.DataFrame -> pd.DataFrame
    This function takes in the data from the table acquired from the website:
    http://www.citymayors.com/statistics/largest-cities-population-125.html
    and gets the full location and geo location in latitude and longitude use GeoPy
    :param df: pd.DataFrame
    :return df: pd.DataFrame
    """
    df['City'] = df['City'].apply(lambda x: x.lower())
    df['Country'] = df['Country'].apply(lambda x: x.lower())
    df['Full Location'] = df['City'] + "," + df['Country']

    latitudes = np.zeros(len(locationsDF))
    longitudes = np.zeros(len(locationsDF))

    for i in range(len(locationsDF)):
        try:
            location = geolocater.geocode(df['Full Location'][i])
            print(location.address)
            latitudes[i] = location.latitude
            longitudes[i] = location.longitude
        except:
            latitudes[i] = 0
            longitudes[i] = 0

    df['Longitude'] = pd.Series(longitudes)
    df['Latitude'] = pd.Series(latitudes)
    df = df[df['Latitude'] != 0]
    df = df.reset_index(drop=True)

    return df


def clean_location(df, user_location, place, latitude, longitude):
    """
    This function cleans the location to grab a city and country if possible.
    It will first try to grab the closest location based on the latitude, longitude. If that fails, it will concatenate
    user_location and place, and split it into bigrams and unigrams. It will try matching bigrams and unigrams to find
    locations within the locations database. 
    :param df: pd.DataFrame
    :param user_location: str
    :param place: str
    :param latitude: double
    :param longitude: double
    :return: cleaned_location: str
    """

    if latitude != '' and longitude != '':
        geo_location = (latitude, longitude)

        distances = np.zeros(len(df))

        for i in range(len(df)):
            city_geo_location = (df['Latitude'][i], df['Longitude'][i])

            try:
                distances[i] = vincenty(geo_location, city_geo_location).kilometers
            except ValueError:
                distances[i] = 1000000

            if distances[i] <= 50:
                return df['City'][i] + ',' + df['Country'][i]

        return df['City'][distances.argmin()] + ',' + df['Country'][distances.argmin()]

    location = user_location + " " + place
    cleaned_location = ''

    if len(location) == 0:
        return cleaned_location

    cleaned_unigrams = unigrams(location)
    cleaned_bigrams = bigrams(location)

    for city in cleaned_bigrams:
        if city in set(df['City']):
            cleaned_location = df[df['City'] == city]['City'].to_string().split()[1] + ' ' \
                               + df[df['City'] == city]['City'].to_string().split()[2] + ',' \
                               + df[df['City'] == city]['Country'].to_string().split()[1]
            return cleaned_location

    for city in cleaned_unigrams:
        if city in set(df['City']):

            cleaned_location = df[df['City'] == city]['City'].to_string().split()[1] + ',' \
                               + df[df['City'] == city]['Country'].to_string().split()[1]
            return cleaned_location

    for country in cleaned_unigrams:
        if country in set(df['Country']):
            cleaned_location = "," + country

    for country in cleaned_bigrams:
        if country in set(df['Country']):
            cleaned_location = "," + country

    return cleaned_location


def update_tweets_df(search_string, places_df, tweets_df, tweets_per_location=100, locations_per_day=180,
                     days=10, results_type=search_type, max_range=search_radius, sleep_time=sleep_duration):
    """
    This function updates the twitter database by mining more tweets from twitter given the parameters
    
    :param search_string: str
    :param places_df: pd.DataFrame
    :param tweets_df: pd.DataFrame
    :param tweets_per_location: int
    :param locations_per_day: int
    :param days: int
    :param results_type: str
    :param max_range: int
    :param sleep_time: int
    :return: tweetDF: pd.DataFrame
    """
    # Correct inputs
    if locations_per_day > len(places_df):
        locations_per_day = len(places_df)

    if days > 10:
        days = 10

    queries_per_location = int(tweets_per_location/100)
    if tweets_per_location/100 > int(tweets_per_location/100):
        queries_per_location += 1

    # Create new dataframe to store tweets
    index = ['tweet_id', 'created_at', 'screen_name', 'searched_location', 'population', 'cleaned_location',
             'user_location', 'place', 'latitude', 'longitude',
             'tweet', 'hashtags', 'fav_count', 'retweet_count']

    new_tweets_df = pd.DataFrame(data=np.random.randn(days * locations_per_day * tweets_per_location, len(index)),
                                 columns=index)
    max_ids = np.zeros(locations_per_day)
    df_indx = 0

    # print
    queries_completed = 0
    query_limit_exceeded = False
    for date_itr in range(days):
        date = (datetime.today() - timedelta(days=date_itr)).date()

        for location_itr in range(locations_per_day):
            latitude = places_df['Latitude'][location_itr]
            longitude = places_df['Longitude'][location_itr]

            for query_itr in range(queries_per_location):
                results_requested = 100
                if query_itr == queries_per_location - 1:
                    results_requested = tweets_per_location % 100

                if max_ids[location_itr] == 0:
                    try:
                        query = twitter.search.tweets(q=search_string,
                                                      geocode="%f,%f,%dkm" % (latitude, longitude, max_range),
                                                      result_type=results_type, until=date,
                                                      lang='en', count=results_requested)
                    except:
                        print('Query Limited Exceeded')
                        query_limit_exceeded = True
                        break
                else:
                    try:
                        query = twitter.search.tweets(q=search_string,
                                                      geocode="%f,%f,%dkm" % (latitude, longitude, max_range),
                                                      result_type=results_type, until=date,
                                                      max_id=max_ids[location_itr],
                                                      lang='en', count=results_requested)
                    except:
                        print('Query Limited Exceeded')
                        query_limit_exceeded = True
                        break

                queries_completed += 1
                print('Progress: ', queries_completed / queries_per_location / locations_per_day / days * 100.0, '%')
                old_df_indx = df_indx

                for tweet in query['statuses']:
                    max_ids[location_itr] = tweet['id'] - 1
                    new_tweets_df.loc[df_indx, 'tweet_id'] = tweet['id']
                    new_tweets_df.loc[df_indx, 'created_at'] = tweet['created_at']
                    new_tweets_df.loc[df_indx, 'screen_name'] = tweet['user']['screen_name'].replace('\n', ' ')\
                        .replace('\t', ' ')
                    new_tweets_df.loc[df_indx, 'tweet'] = tweet['text'].replace('\n', ' ').replace('\t', ' ')
                    new_tweets_df.loc[df_indx, 'fav_count'] = tweet['favorite_count']
                    new_tweets_df.loc[df_indx, 'retweet_count'] = tweet['retweet_count']
                    new_tweets_df.loc[df_indx, 'user_location'] = tweet['user']['location'].replace('\n', ' ')\
                        .replace('\t', ' ')

                    new_tweets_df.loc[df_indx, 'latitude'] = ''
                    new_tweets_df.loc[df_indx, 'longitude'] = ''
                    new_tweets_df.loc[df_indx, 'hashtags'] = '[]'
                    new_tweets_df.loc[df_indx, 'place'] = ''

                    if tweet['coordinates']:
                        new_tweets_df.loc[df_indx, 'longitude'] = tweet['coordinates']['coordinates'][0]
                        new_tweets_df.loc[df_indx, 'latitude'] = tweet['coordinates']['coordinates'][1]

                    if tweet['entities']['hashtags']:
                        hashtag_list = []
                        for hashtag in tweet['entities']['hashtags']:
                            hashtag_list.append(hashtag['text'])
                        new_tweets_df.loc[df_indx, 'hashtags'] = '[' + ','.join(hashtag_list) + ']'

                    if tweet['place']:
                        new_tweets_df.loc[df_indx, 'place'] = tweet['place']['full_name'].replace('\n', ' ')\
                                                              .replace('\t', ' ') \
                                                              + ' ' + tweet['place']['country'].replace('\n', ' ')\
                                                              .replace('\t', ' ')

                    new_tweets_df.loc[df_indx, 'cleaned_location'] = clean_location(places_df,
                                                                                    new_tweets_df.loc[df_indx,
                                                                                                      'user_location'],
                                                                                    new_tweets_df.loc[df_indx,
                                                                                                      'place'],
                                                                                    new_tweets_df.loc[df_indx,
                                                                                                      'latitude'],
                                                                                    new_tweets_df.loc[df_indx,
                                                                                                      'longitude'])
                    df_indx += 1

                for indx_itr in range(old_df_indx, df_indx):
                    new_tweets_df.loc[indx_itr, 'searched_location'] = places_df['City'][location_itr] + ',' \
                                                                       + places_df['Country'][location_itr]
                    new_tweets_df.loc[indx_itr, 'population'] = places_df['Population'][location_itr]

                if queries_completed % 180 == 0:
                    time.sleep(sleep_time)

            if query_limit_exceeded:
                break

        if query_limit_exceeded:
            break

    new_tweets_df = new_tweets_df[0:df_indx]
    tweets_df = pd.concat([tweets_df, new_tweets_df])
    tweets_df = tweets_df.drop_duplicates(subset='tweet_id', keep='last').reset_index(drop=True)
    tweets_df.to_csv('TweetDatabase.tsv', index=False, sep='\t')

    return tweets_df


if __name__ == '__main__':

    if get_search_location_data:
        locationsDF = pd.read_csv('RawWorldCities.csv', sep=',')
        locationsDF = get_location_data(locationsDF)
    else:
        locationsDF = pd.read_csv('WorldLocations.tsv', sep='\t')

    tweetsDF = pd.read_csv('TweetDatabase.tsv', sep='\t')

    if update_tweet_database:
        locationsDF = locationsDF.sample(frac=1).reset_index(drop=True)
        old_size = len(tweetsDF)
        tweetsDF = update_tweets_df(list_to_str(search_terms), locationsDF, tweetsDF,
                                    tweets_per_location=100, locations_per_day=5, days=2)
        print('Tweets Added:', len(tweetsDF)-old_size)

    print(tweetsDF.head())

    print(tweetsDF['tweet_id'].count())