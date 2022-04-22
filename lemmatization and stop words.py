import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """finishing called the one minute sentence we mincemeat 
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
               
               
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   
