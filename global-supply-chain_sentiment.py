#!/usr/bin/env python
# coding: utf-8

# In[2]:


from GoogleNews import GoogleNews
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk import word_tokenize
from nltk import pos_tag
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud
import re
import string


# In[3]:


googlenews = GoogleNews(start='03/01/2024',end='03/16/2024')
googlenews.get_news('latest supply chain', 'el nino supply chain')
googlenews.search('supply chain')


# In[4]:


result_0=googlenews.result()
desc_1 = googlenews.get_texts()
link_1 = googlenews.get_links()

for i in list(range(2, 70)):

    result = googlenews.page_at(i)
    desc = googlenews.get_texts()
    link = googlenews.get_links()

    desc_1 = desc_1 + desc
    link_1 = link_1 + link
column_names = ["title", 'link']
data = pd.DataFrame(columns = column_names)

data['title'] = desc_1
data['link'] = link_1
display(data)


# In[5]:


data.to_csv('google_news.csv', index = False)


# In[6]:


data.info()


# In[8]:


#Checking for missing values
np.sum(data.isnull().any(axis=1))


# In[9]:


data['title_processed'] = [title.lower() for title in data['title']]


# In[10]:


def unnecessary_words(title_processed):
  return re.sub(r'\b\w{1,2}\b', '',title_processed)
data['title_processed'] = data['title_processed'].apply(lambda x: unnecessary_words(x))
data['title_processed'].head()


# In[11]:


def cleaning_repeating_char(title_processed):
  return re.sub(r'(.)1+', r'1', title_processed)
data['title_processed'] = data['title_processed'].apply(lambda x: cleaning_repeating_char(x))
data['title_processed'].head()


# In[12]:


def cleaning_URLs(title_processed):
  return re.sub(r'http\S+', ' ', title_processed)
  #return re.sub('((www.[^s]+)|(https?://[^s]+))'," ",tweet_processed)
data['title_processed'] = data['title_processed'].apply(lambda x: cleaning_URLs(x))
data['title_processed'].head()


# In[13]:


#removing HTML tags
def remove_html(title_processed):
  html=re.compile(r'<.*?>')
  return html.sub(r'', title_processed)
data['title_processed'] = data['title_processed'].apply(remove_html)


# In[14]:


def remove_com(title_processed):
  return re.sub(r"\ [A-Za-z]*\.com", " ", title_processed)
data['title_processed'] = data['title_processed'].apply(lambda x: remove_com(x))
data['title_processed'].head()


# In[15]:


#cleaning and removing numeric numbers
def cleaning_numbers(title_processed):
  return re.sub('[0-9]+', '', title_processed)
data['title_processed'] = data['title_processed'].apply(lambda x: cleaning_numbers(x))
data['title_processed'].head()


# In[16]:


def stripping_extra_spaces(title_processed):
  return re.sub(r' +', ' ', title_processed)
data['title_processed'] = data['title_processed'].apply(lambda x: stripping_extra_spaces(x))
data['title_processed'].head()


# In[18]:


add_words = ['â€˜', ' ðŸšš ðŸš', 'Visit','say','give','would','demo', 'center', 'Itâ€™s', 'â‡ï', 'ðŸŒŽ','scm', 's', 'get','take','help', 'read', 'more', 'new', 'one', 'supply','webinar','republic','latin']
stop_words.extend(add_words)

def cleaning_stopwords(title_processed):
  return " ".join([word for word in str(title_processed).split() if word not in stop_words])
data['title_processed'] = data['title_processed'].apply(lambda title_processed: cleaning_stopwords(title_processed))
data['title_processed'].head()


# In[19]:


def clean(text):
    # Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text
data['title_processed'] = data['title_processed'].apply(clean)
data['title_processed']


# In[20]:


mydata = pd.DataFrame(data[['title','title_processed']])


# In[21]:


pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
def token_stop_pos(title_processed):
  tags = pos_tag(word_tokenize(title_processed))
  newlist = []
  for word, tag in tags:
    if word.lower() not in set(stopwords.words('english')):
      newlist.append(tuple([word, pos_dict.get(tag[0])]))
  return newlist
mydata['pos tagged'] = mydata['title_processed'].apply(token_stop_pos)
mydata.head()


# In[22]:


wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

mydata['Lemmatized_sent'] = mydata['pos tagged'].apply(lemmatize)
mydata.head()


# In[23]:


from textblob import TextBlob

# function to calculate subjectivity
def getSubjectivity(title_processed):
    return TextBlob(title_processed).sentiment.subjectivity

# function to calculate polarity
def getPolarity(title_processed):
    return TextBlob(title_processed).sentiment.polarity

# function to analyze the reviews
def analysis(score):
    if score <= 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


# In[24]:


mydata['Polarity'] = mydata['Lemmatized_sent'].apply(getPolarity)
mydata['Analysis'] = mydata['Polarity'].apply(analysis)
mydata.head()


# In[25]:


tb_counts = mydata.Analysis.value_counts()
tb_counts


# In[26]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

tb_count= mydata.Analysis.value_counts()
plt.figure(figsize=(10, 7))
plt.pie(tb_counts.values, labels = tb_counts.index, autopct='%1.1f%%', shadow=False)
# plt.legend()


# In[173]:


get_ipython().system('pip install vaderSentiment')


# In[27]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
# function to calculate vader sentiment
def vadersentimentanalysis(title_processed):
    vs = analyzer.polarity_scores(title_processed)
    return vs['compound']
mydata['Vader Sentiment'] = mydata['Lemmatized_sent'].apply(vadersentimentanalysis)


# In[28]:


# function to analyse
def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5 :
        return 'Negative'
    else:
        return 'Neutral'
mydata['Vader_Analysis'] = mydata['Vader Sentiment'].apply(vader_analysis)
mydata.head()


# In[29]:


#count of sentiments with Neutral sentiment being considered
vader_counts = mydata['Vader_Analysis'].value_counts()
vader_counts


# In[30]:


vader_counts= mydata['Vader_Analysis'].value_counts()
plt.figure(figsize=(10, 7))
plt.rcParams['font.size'] = '13'
plt.pie(vader_counts.values, labels = vader_counts.index, explode = (0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.title("Global Supply Chain Sentiment in March 2024 ")
plt.legend(fontsize = 11)


# In[31]:


from wordcloud import WordCloud, ImageColorGenerator


# In[32]:


def create_wordcloud(text):
  wc = WordCloud(background_color = 'white', max_words=3000, repeat=False)
  wc.generate(str(text))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis('off')
  plt.show()
create_wordcloud(mydata['Lemmatized_sent'].values)


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer
countVectorizer = CountVectorizer()
countVector = countVectorizer.fit_transform(mydata['Lemmatized_sent'])
print('{} Number of tweets have {} words'.format(countVector.shape[0], countVector.shape[1]))

count_vect_df = pd.DataFrame(countVector.toarray(), columns = countVectorizer.get_feature_names_out())
count_vect_df


# In[34]:


counts = pd.DataFrame(count_vect_df.sum())
count_df = counts.sort_values(0, ascending = False).head(20)
count_df


# In[35]:


ind = count_df.index
val = [item for sublist in count_df.values for item in sublist]
plt.bar(ind, val)
plt.xticks(rotation=90)
plt.title('20 Most frequently used words - Supply Chain, March 2024')


# In[36]:


word_cloud_df = mydata.loc[mydata['Vader_Analysis'] == 'Negative', :]
all_words = ' '.join([text for text in word_cloud_df['Lemmatized_sent']])

wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(all_words)

plt.figure(figsize=(8,8), facecolor=None)
plt.rcParams['font.size'] = '16'
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.title('NEGATIVE SENTIMENTS - Supply Chain, March 2024')
plt.show()


# In[37]:


word_cloud_df1 = mydata.loc[mydata['Vader_Analysis'] == 'Positive', :]
all_words = ' '.join([text for text in word_cloud_df1['Lemmatized_sent']])

wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(all_words)

plt.figure(figsize=(8,8), facecolor=None)
plt.rcParams['font.size'] = '16'
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.title('POSITIVE SENTIMENTS - Supply Chain, March 202')
plt.show()


# In[38]:


from nltk.util import ngrams
#tokens = mydata['Lemmatized_sent'].str.split()
#sequences = [tokens[i:] for i in range(3)]
#bigrams = zip(*sequences)
n_grams = ngrams(mydata['Lemmatized_sent'].str.split(), 3)
for grams in n_grams:
    print(grams)

#output = list(ngrams(mydata['Lemmatized_sent'], 5))
#print(output)


# In[39]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[40]:


tfidf = TfidfVectorizer(max_features=2500)
X = tfidf.fit_transform(mydata.Lemmatized_sent).toarray()
y = mydata.Vader_Analysis.map({'Positive': 1, 'Neutral': 0, 'Negative':-1}).values
featureNames = tfidf.get_feature_names_out()

print("Number of features:", len(featureNames))
print("Features:", featureNames)


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)


# In[42]:


X_train.shape, X_test.shape


# In[43]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

## Testing the model on test set
y_pred = classifier.predict(X_test)
y_pred


# In[44]:


from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("The model accuracy is", accuracy )


# In[46]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth = 15)
dt.fit(X_train,y_train)
dt.score(X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




