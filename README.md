# Week4-NLP-Disaster-Tweets-Kaggle-Mini-Project


**Problem and Data Description**

In this project I will pull data we will construct a couple RNNs to classify binary text data as either "disaster" or "not disaster". The data come from a (Kaggle "getting started" competition)[https://www.kaggle.com/competitions/nlp-getting-started/overview] designed to be an excellent introduction to Natural Language Processing. The goal of NLP is to input text data and output something. NLP could output a "mood" for sentiment analysis, a predicted stock price from ticker mentions, more text in the frame of language conversion, and countless other outputs in a variety of applications. I will utilize TensorFlow's Keras API to build a recurrent nueral network to perform binary classification. A recurrent nueral network is suited for data with a fixed series, for example: time series data like stock price, code translation from one language to machine language, next-frame generation from videos and/or gifs, and as I perform in the following project text classification in buckets - disaster vs. not disaster.

The data in this project are tweets. There are three meaningful features: the full text of the tweet, the location from which the tweet was made, and a keyword the authors tagged as potentially indicating an accident. Note that not all keyword inclusions are the result of disasters.


**Cleaning and Exploratory Data Analysis**

There are 7613 rows in the dataset total, of which id, text, and target are fully populated. Location is recorded for 5080 records, and there are keywords for all but 61 observations. We need to do some initial cleaning because the data types for keyword, location, and text should be strings, but they are "object".

Here are the two questions I asked myself immediately: (1) Is keyword a useful predictor of target? (2) What about location? If variance in the features do not correlate with large variance in the binary response, we can say that the location / keyword can be ignored as a feature. Otherwise, we should include the feature. I'll make a histogram of counts ver proportion of tweets with each label for both keywords and targets, grouped by feature, for groups with at least 30 observations. We want to ensure we only include groups with at least 30 observations to let the central limit theorem do it's work. We should get a distribution that is approaching Gaussian if the feature is "nice". This happens to be easier to do in R using tidyverse, so I'll do it there and import the graphs I made. I will also provide a copy of the .rmd in my GitHub repo for this project for the sake of reproducability. Let's start with location. 

![location_observation](https://github.com/user-attachments/assets/a2a15946-56e8-4fc0-8539-0cba76b0e3a6)

What can we learn from this? The locations are super high-variance entities, and don't resemble a nice probability distribution. Although nueral networks are universal function approximators and so could reproduce the location distribution for the training data, the locations are so high variance that we are unlikely to gain much advantage from feeding the RNN location information. Some locations are very rare, so we could drastically overfit based on location. Even if we don't (by heavily controlling for validation accuracy), the computation cost vs. marginal benefit to binary crossentropy seems not worthwhile.

As further evidence against usage of location as a feature, let's look at a histogram of the marginal pdf for location.

![PDF_Locations](https://github.com/user-attachments/assets/2a553e0f-3039-4fc9-b60a-55ceee7ce3d2)

Now let's look at keyword.

![keyword_mentions](https://github.com/user-attachments/assets/c447c408-9a1b-4647-be55-60ad726c97e7)


Keyword is a fabulous feature in contrast to location. The pdf per target value is reasonably continuous and ranges over a wide swath of the domain [0,1]. One thing though - inclusion of keyword doesn't pass a sanity check. The reason the keyword exists is because it is a standout word in the tweet that might indicate the presence of a disaster. It doesn't make sense to manually weight any given keyword more than any other, because the whole point of using a nueral network on NLP is to let it find the optimal weights for each word / character. So, in summary: we have reason to completely exclude both location and keyword from our nueral network. This has the added benefit of allowing us to use the entire training dataset without pruning missing values, because every tweet has text. ID is also not a useful feature - it seems like vestigial as a primary key from the original dataset before splitting into test and train and hosting on Kaggle.

Here's the pdf for keyword:

![PDF_Keywords](https://github.com/user-attachments/assets/f34cf918-542e-4339-86be-ebc9bad01718)

**Transformation with TF-IDF:**

TF-IDF stands for term frequency - inverse data frequency. TF-IDF constructs a skewed matrix, which then can be normalized into a form that is convenient to factor. Fortunately sklearn takes care of this normalization for us. In TF-IDF each word is given a weight which balances it's frequency with its rarity, the underlying idea being that globally rare words are useful as categorical identifiers, and the more frequent such "buzzwords" are used the more clear indication of a category it is. Here's the math:

The frequency of a term in a document (i.e. a word or phrase, but in our implementation it will just be a word due to processing time concerns) is calculated as:

TF(i,j)=Frequency of term i in document jTotal word count of document j 

The inverse document frequency of a term over all documents is calculated as:

IDF(i)=log2Total number of documentsNumber of documents with term i 

By multiplying these quantities, we obtain weights for word "usefulness" in identifying categories. We can run this over all articles in a desired dataframe (either test or train) to construct a matrix of dimension (number of documents, number of unique terms).

One issue is that stopwords like "the", "I", "we", "he", etc. are the most prevalent in text, and do not indicate whether a disaster has taken place. As such the weights will be very slow to update in our GAN. We therefore need to cleanse the stopwords. There are some fortunately very good libraries for this!

Citation:

Medium article 1: https://www.google.com/url?q=https%3A%2F%2Fmedium.com%2Fswlh%2Ftext-classification-using-tf-idf-7404e75565b8

Medium article 2: https://www.google.com/url?q=https%3A%2F%2Fmedium.com%2F%40imamun%2Fcreating-a-tf-idf-in-python-e43f05e4d424

**Model Architecture, Results, and Analysis**

The tensorflow team supplies a tutorial for using RNN for text classification which I will be following and adapting for this project. The tutorial uses an auto-encoder, and so we will actually change tack and not use TF-IDF with the RNN and just do an ANN with TF-IDF. We will then compare against the RNN as an architecture optimization.

**Model Commentary:**

The best model started with  η=5×10−5 . I used learning rate scheduling to help prevent overfitting. The graphs show that the optimal number of epochs to train the model was either 10 or 11, because the validation loss did not decrease nor the accuracy meaningfully increase after that. Let's go ahead and make predictions on the test set, submit to kaggle, then compare against the RNN model. For my RNN I will use the Long Short Term Memory model from the Keras API.

**Results and Conclusion**

In this project I compared two different model architectures that attempted to classify tweets as either "disaster" or "not disaster". The first model I tried was an artificial nueral network that took as an input a TF-IDF matrix. I trained the TF-IDF features on only the training data - we could have artificially increased the test accuracy by including the test tweets in the input features (TF-IDF is unsupervised, so the lack of labels would not have been an impedement). This is borderline unethical - the function of test data is to determine how well the model would perform on data it has never seen before. We didn't do that. The ANN with TF-IDF yielded a test set accuracy of 0.781.

The second model I used was a RNN with a bi-directional LSTM layer. The LSTM layer used an auto-encoder as input, which was probably suboptimal. In the future I would like to find a way to make the LSTM layer take as input my TF-IDF matrix to (1) prevent duplication of effort, and (2) potentially increase test set accuracy. I couldn't manage to align the shapes in this project to make that work, but I just need more experience with Keras. That would be a good follow up project. As is, the RNN returned a test set accuracy of 0.789, which is slightly better than the ANN.

I learned a lot in this project, and am eager to use my new NLP tools in Keras to perhaps do sentiment analysis. For a future data mining project I want to predict youtube views based on a variety of inputs. These NLP techniques will help include video title in those inputs. Thanks for taking the time to review my project!
