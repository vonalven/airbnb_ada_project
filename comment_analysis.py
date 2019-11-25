import string
import re
import nltk
import matplotlib as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from langdetect import detect

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


def get_sentiment(comment):
    '''
    takes a comment as an argument, sent_tokenize() it to seperate the sentences
    then computes the scores of negativity, neutrality, positivity and compound for each sentence
    and finally compute the mean of these scores for the whole comment
    
    '''
    # tokenize comment to get seperated sentences
    comment_tk = nltk.sent_tokenize(comment)
    
    # compute sentiment scores for each sentence
    scores = list()
    for c in comment_tk:
        scores.append(sid.polarity_scores(c))
    
    # compute mean scores for the comment
    neg = 0
    neu = 0
    pos = 0
    comp = 0
    for i in range(len(comment_tk)):
        neg += scores[i]['neg']
        neu += scores[i]['neu']
        pos += scores[i]['pos']
        comp += scores[i]['compound']
    
    mean_neg = neg/len(comment_tk)
    mean_neu = neu/len(comment_tk)
    mean_pos = pos/len(comment_tk)
    mean_comp = comp/len(comment_tk)
    
    return [mean_neg, mean_neu, mean_pos, mean_comp]


def analyze_comments(comments):
    print('This data contains', comments.shape[0], 'lines.')
    
    # remove NaN
    comments = comments.dropna(how='any',axis=0)
    
    # remove comments only filled with whitespaces
    comments['isSpace'] = comments['comments'].apply(lambda x: x.isspace())
    comments = comments[comments.isSpace == False]
    comments = comments.drop(columns=['isSpace'])
    
    # remove all non-alphabetical characters to allow detect() to work
    regex = re.compile('[^A-Za-zÀ-ÿ]')      
    comments['rm_comments'] = comments['comments'].apply(lambda x: regex.sub(' ', x))
    
    # again, remove comments only filled with whitespaces
    comments['isSpace'] = comments['rm_comments'].apply(lambda x: x.isspace())
    comments = comments[comments.isSpace == False]
    comments = comments.drop(columns=['isSpace'])
    
    # detect the language of each comment
    comments['language'] = comments['rm_comments'].apply(lambda x: detect(x))
    
    # as the previous step takes some time, the result is saved and can be loaded for further use
    comments.to_pickle("./comments_languages.pkl")
    
    # keep only english comments
    comments_en = comments[comments.language == 'en']
    
    # non-alphabetical characters are removed but the ponctuation in the comments is kept
    regex2 = re.compile('[^A-Za-zÀ-ÿ?!.,:;]')     
    comments_en['ap_comments'] = comments_en['comments'].apply(lambda x: regex2.sub(' ', x))
    
    # remove unnecessary columns for next steps
    comments_en = comments_en.drop(columns=['comments', 'rm_comments', 'language'])
    
    # get sentiment
    comments_en['sentiment'] = comments_en['ap_comments'].apply(lambda x: get_sentiment(x))
    
    # get a column for each different score (neg, pos, neu, comp) of each comment
    comments_en[['negativity','neutrality', 'positivity', 'compound']] = pd.DataFrame(comments_en.sentiment.values.tolist(), index = comments_en.index)

    # remove unnecessary column
    comments_en = comments_en.drop(columns=['sentiment'])
    
    # work on a copy as we remove columns that could be useful afterwards
    comments_en_copy = comments_en.copy()
    comments_en_copy = comments_en_copy.drop(columns=['id', 'date', 'reviewer_id', 'reviewer_name', 'ap_comments'])
    
    # compute the mean of each score (neg, pos, neu, comp) for a given housing (-> using listing_id to groupby)
    comments_en_copy = comments_en_copy.groupby('listing_id').mean()
    print('There are', comments_en_copy.shape[0], 'different housings in this city.')
    
    # plot 
    #comments_en_copy.hist(bins=100)
    
    return comments, comments_en, comments_en_copy

