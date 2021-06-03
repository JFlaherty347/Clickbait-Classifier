# Clickbait-Classifier

## Problem Statement

In the digital age, countless websites and articles are fighting for our attention in order to gain traction for their websites. This has manifested itself in the presentation of news headlines, where the titles go from intriguing to inaccurate. Clickbait, as it is commonly known, has become increasingly common. Whether the article's headline is a half-truth or a full lie, the reader is typically left disappointed by the unkept promises of the title. The author, on the other hand, has succeeded nonetheless in getting people to look at their webpage and in turn, their advertisements.

Clickbait doesn't just affect those unfortunate enough to be tricked into reading the articles. Their mere presence dilutes the actual news for anyone scrolling through a news feed. Even if you can recognize all clickbait, it still ends up wasting your time because you have to sift through it all to get to the news you actually want to read. Clickbait titles can also make readers suspicious of legitimate news with impressive titles, making real news harder to digest.

Given the headache of clickbait, can a machine learning model determine whether or not a headline is clickbait? How well can clickbait be separated from honest headlines? This project aims to create a machine learning model capable of delineating between clickbait and honest headlines.

## Dataset

The dataset used is the "News Clickbait Dataset" which is [available here](https://www.kaggle.com/vikassingh1996/news-clickbait-dataset).

The data contains two subsets. We chose to start with only the first because the second subset was unbalanced. The first subset contained a balanced 32,000 samples so we were confident that it would be sufficient for a competent agent. The data is formatted in a very simple fashion. It contains only two fields: a headline given as a string and a clickbait status given as a 0 or 1. As a result, the model only uses the title of the article to make its decision. Other information such as the contents of the article, the author, the website of publication, and associated images are not considered.

## Applied Techniques

The machine learning techniques we evaluated are:

- Naive Bayes
- Vector Machines 
- MLP Neural Networks
 
Our goal with our different model types was to encompass a range of varying complexity to allow us to use the simplest sufficient model.
  
The data pre-processing techniques we evaluated are:

- lowercasing text
- removing punctuation
- tokenization
- term counting
- normalizing
- term frequency representation
- stop list
- stemming
- lemmatization

The data pre-processing techniques were chosen in alignment with the most common Natural Language Processing Techniques. 

## Evaluation Measures

The data evaluation measures we employed are:

- F1-Score
- Confusion Matrix
- Accuracy
- Precision
- Recall

Of these evaluation measures, we valued F1-Score and confusion matrices the most. F1-Scores gave us a robust representation of how well a particular model was doing overall. Confusion matrices helped us to diagnose if a model was incorrectly classifying too many articles as clickbait or vice versa. The other metrics were also useful statistics but didn't drive our process as much as F1-Score or confusion matrices. 

## Results

Some of our most notable results for how our model performed on unseen test data are as follows:

- Naive Bayes F-1 Score: 0.97328
- SVM F-1 Score: 0.97024
- Neural Network F-1 Score: 0.97348
- NN With Preprocessing F1-Score: 0.96846
    
## Conclusion

Ultimately, we found that effective clickbait classifiers can be created solely based on the article's title. Another key takeaway from this project is that even a simple model such as Naive Bayes can perform extremely well. While near-perfect accuracy is still just out of reach, increased model complexity and more features would likely be enough to break past the 0.97 F1-Score barrier.

We feel that our models perform exceptionally well and would function properly in a production environment. That being said, if we wanted to strive for near-perfect accuracy, there are some improvements that we could still make. First, incorporating the second subset of data into the data we used could help make the model more robust. Second, we feel that collecting more features in the dataset would also create a way to understand more challenging examples. A feature like the author of the article could help rule out any articles written by established journalists. The images used alongside the article could also provide some insight, with clickbait articles perhaps leaning towards more shocking images.

Other changes that may help make improvements would involve our pre-processing pipeline. Various other techniques are possible although we feel that these changes alone are unlikely to bring about near-perfect accuracy. Having already tried a variety of methods and different approaches to many pre-processing techniques, it seems as though we have attempted many of the changes that would bring about the most positive impact. That being several incremental improvements could still help the model perform even better.
