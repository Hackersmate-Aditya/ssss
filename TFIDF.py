import nltk

paragraph =  """finishing called the one minute sentence we mincemeat 
tool that allows us students to summarise their understanding of a 
lesson it can be used at the end of the lesson or may be the beginning 
of the next day have them summarizing the lesson of the previous day or
 maybe homeward the previous night the one minute sentence you tasteless
 as planned or the activity assign the homework assignment and then the
 first step is to give them whatever their writing on indecisive with
 the instructions with a question or what they are summarizing then said 
 the time or for one minute and have them right a sentence or two that's 
 really concise and to the point that as summarizing what they learned
 from the lesson or maybe the previous homework assignment so member when
 i say go you'll have one minute to write your sentence explaining the
 benefits of sexual reproduction compared to a sexual reproduction then
 you'll share with your group pick the best one and then you'll share
 the best one with the class we ready said no i like that it is quick
 it only has one minute you can start the time one minute and you're
 done i think the students like the fact that there receiveth also
 like that at short and hadn't have to write a whole paragraph they
 don't have to write five sentences they can describe one or two in
 the women a time period it allows him very requires them to pick 
 the most important ideas they're not worth for word repeating the
 same thing that they heard the same thing in her notes but they
 have to get it down to justine concise sentence it also shows their
 understanding their comprehension how is deeper comprehension is
 whether they are distributing your weather they are putting it 
 into their own words and so it gives a teacher a very quick
 assessment of their understanding and the level of understanding"""
               
               
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()