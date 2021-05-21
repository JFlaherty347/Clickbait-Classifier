import pandas as pd
import numpy as np
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from trainAndTestNN import test_simple_fc_nn

# function from https://towardsdatascience.com/setting-up-text-preprocessing-pipeline-using-scikit-learn-and-spacy-e09b9b76758f
def pipelinize(function, active=True):
  def list_comprehend_a_function(list_or_series, active=True):
    if active:
      return [function(i) for i in list_or_series]
    else: # if it's not active, just pass it right back
      return list_or_series
  return FunctionTransformer(list_comprehend_a_function, validate=False, kw_args={'active':active})

def stemmer(text):
  stemmer = LancasterStemmer()

  return stemmer.stem(text)

def lemmatizer(text):
  lemmatizer = WordNetLemmatizer()
  return lemmatizer.lemmatize(text)

def split_and_countvectorize(df_headline:pd.DataFrame, label:pd.DataFrame):
  label = df['clickbait']
  df_headline = df.drop(columns=['clickbait'])
  #print(df_headline)

  x_train, x_test, y_train, y_test = train_test_split(df_headline, label, test_size=0.3, random_state = 123)

  # nltk.download('wordnet')

  preprocessing = Pipeline([('stemmer', pipelinize(stemmer)), ('vectorizer', CountVectorizer(lowercase=True)), 
   ('term_frequency', TfidfTransformer(use_idf=False))])
  # preprocessing = Pipeline([('lemmatizer', pipelinize(lemmatizer)), ('vectorizer', CountVectorizer(lowercase=True)), 
  #  ('term_frequency', TfidfTransformer(use_idf=False))])

  x_train = preprocessing.fit_transform(x_train['headline'])
  x_test = preprocessing.transform(x_test['headline'])

  return x_train, x_test, y_train, y_test

def test_multinomial_nb(x_train, x_test, y_train, y_test):
  model = MultinomialNB(alpha = 0.2).fit(x_train, y_train)
  # model = SVC().fit(x_train, y_train)

  prediction = model.predict(x_test)

  class_report = classification_report(y_test, prediction)
  print("Report: \n", class_report)
  conf_matrix = confusion_matrix(y_test, prediction)
  print("\nConfusion Matrix:\n", conf_matrix)
  f1 = f1_score(y_test, prediction)
  print("\nF1-Score: ", f1)



df = pd.read_csv('./train1.csv')

label = df['clickbait']
df_headline = df.drop(columns=['clickbait'])
print(df_headline)

x_train, x_test, y_train, y_test = split_and_countvectorize(df_headline, label)

print(x_train)
print(x_test)

print("Multinomial Naive Bayes Test:")
test_multinomial_nb(x_train, x_test, y_train, y_test)


print("Simple NN:")
test_simple_fc_nn(x_test=x_test.todense(), y_test=y_test.to_numpy())

# NB with previous setup: F1-Score:  0.9732808616404309
# NB with lancaster stemming: F1-Score:  0.9732863946986954

# Simple NN: F1-Score: 0.9734832414085702

