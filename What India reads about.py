#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import spacy
from wordcloud import WordCloud


# In[2]:


import os
os.chdir('C:\\Analytics\\Text mining\\India reads about news')


# In[3]:


data = pd.read_csv('india-news-headlines.csv')
data.head()


# In[4]:


data = data[['publish_date', 'headline_text']].drop_duplicates()
data['publish_date'] = pd.to_datetime(data['publish_date'], format='%Y%M%d')
data['year'] = data['publish_date'].dt.year
import spacy
nlp = spacy.load("en_core_web_sm")


# In[5]:


data.head()


# In[7]:



import sklearn.feature_extraction.text as text
def get_imp(bow,mf,ngram):
    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)
### Global trends
bow=data['headline_text'].tolist()
total_data=get_imp(bow,mf=5000,ngram=1)
total_data_bigram=get_imp(bow=bow,mf=5000,ngram=2)
total_data_trigram=get_imp(bow=bow,mf=5000,ngram=3)
### Yearly trends
imp_terms_unigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_unigram[y]=get_imp(bow,mf=5000,ngram=1)
imp_terms_bigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_bigram[y]=get_imp(bow,mf=5000,ngram=2)
imp_terms_trigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_trigram[y]=get_imp(bow,mf=5000,ngram=3)


# In[11]:


common_unigram={}
for y in np.arange(2001,2017,1):
    if y==2001:       
        common_unigram[y]=set(imp_terms_unigram[y].index).intersection(set(imp_terms_unigram[y+1].index))
    else:
        common_unigram[y]=common_unigram[y-1].intersection(set(imp_terms_unigram[y+1].index))
### Common bigrams across all the years
common_bigram={}
for y in np.arange(2001,2017,1):
    if y==2001:
         common_bigram[y]=set(imp_terms_bigram[y].index).intersection(set(imp_terms_bigram[y+1].index))
    else:
        common_bigram[y]=common_bigram[y-1].intersection(set(imp_terms_bigram[y+1].index))
### Common trigrams, 1 year window
common_trigram_1yr={}
for y in np.arange(2001,2017,1):
    common_trigram_1yr[str(y)+"-"+str(y+1)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index))
### Commin trigrams, 2 year window
common_trigram_2yr={}
for y in np.arange(2001,2015,3):
    if y==2001:
        common_trigram_2yr[str(y)+"-"+str(y+1)+"-"+str(y+2)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index)).intersection(set(imp_terms_trigram[y+2].index))
    else:
        common_trigram_2yr[str(y)+"-"+str(y+1)+"-"+str(y+2)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index)).intersection(set(imp_terms_trigram[y+2].index))


# Count of top 20 unigrams, bigrams and trigrams

# In[13]:


import matplotlib.pyplot as plt
plt.subplot(1,3,1)
total_data.head(20).plot(kind='bar', figsize=(25,10),colormap='Set2')
plt.title('Unigrams', fontsize=30)
plt.yticks([])
plt.xticks(size=20)
plt.subplot(1,3,2)
total_data_bigram.head(20).plot(kind='bar', figsize=(25,10),colormap='Set2')
plt.title('Bigrams', fontsize=30)
plt.yticks([])
plt.xticks(size=20)
plt.subplot(1,3,3)
total_data_trigram.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')
plt.title("Trigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)


# Bigrams and Trigrams across years

# In[14]:


for i in range(1,18,1):
    plt.subplot(9,2,i)
    imp_terms_bigram[2000+i].head(5).plot(kind='barh',figsize=(20,25),colormap='Set2')
    plt.title(2000+i, fontsize=20)
    plt.xticks([])
    plt.yticks(size=20, rotation=5)


# Top 5 Trigrams across years

# In[15]:


for i in range(1,18,1):
    plt.subplot(9,2,i)
    imp_terms_trigram[2000+i].head(5).plot(kind='barh', figsize=(20,25),colormap='Set2')
    plt.title(2000+i, fontsize=20)
    plt.xticks([])
    plt.yticks(size=15, rotation=5)


# If you look at the trigrams and bigrams closely, you will realize, that reporting of crime, sports (cricket in particular) and Shah Rukh Khan is persistent!!!

# In[16]:


## Count of common tokens across the years
count_common_bi = {}
for year in range(2001,2017,1):
    count_common_bi[year] = pd.Series()
    for word in common_bigram[year]:
        if year == 2001:
            count_common_bi[year][word] = imp_terms_bigram[year][word] + imp_terms_bigram[year+1][word]
        else:
            count_common_bi[year][word] = count_common_bi[year-1][word] + imp_terms_bigram[year+1][word]


# Which bigrams have been consistently reported over years ?

# Top 10 bigrams common across years

# In[18]:


for i in range(1,17,1):
    plt.subplot(9,2,i)
    count_common_bi[2000+i].sort_values(ascending=False).head(10).plot(kind='barh', figsize=(20,35),colormap='Set2')
    
    if (2000+i) == 2001:
        plt.title(str(2000+i) + "-" + str(2000+i+1), fontsize=30)
    else:
        plt.title('upto-' + str(2000+i+1),fontsize=30)
        plt.xticks([])
        plt.yticks(size=20, rotation=1)


# In[20]:


index = data['headline_text'].str.match(r'(?=.*\byear\b)(?=.*\bold\b).*$')
texts = data['headline_text'].loc[index].tolist()
noun = []
verb = []
for doc in nlp.pipe(texts, n_threads=16, batch_size=10000):
    try:
        for c in doc:
            if c.pos_ == 'NOUN':
                noun.append(c.text)
            elif c.pos_ == 'VERB':
                verb.append(c.text)
    except:
        noun.append("")
        verb.append("")
        


# In[22]:


plt.subplot(1,2,1)
pd.Series(noun).value_counts().head(10).plot(kind='bar',figsize=(20,5),colormap='Set2')
plt.title("Top 10 Nouns in context of 'Year Old'",fontsize=30)
plt.xticks(size=20,rotation=80)
plt.yticks([])
plt.subplot(1,2,2)
pd.Series(verb).value_counts().head(10).plot(kind="bar",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs in context of 'Year Old'",fontsize=30)
plt.xticks(size=20,rotation=80)
plt.yticks([])


# In[23]:


data['headline_text'].loc[index].tolist()[0:20]


# Are sucides on a rise in India?

# In[24]:


index_s = data['headline_text'].str.match(r'(?=.*\bcommits\b)(?=.*\bsuicide\b).*$')
text_s = data['headline_text'].loc[index].tolist()
noun_s = []
for doc in nlp.pipe(text_s, n_threads=16, batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_s.append(c.text)
    except:
        for c in doc:
            noun_s.append("")


# In[25]:


pd.Series(noun_s).value_counts().head(20).plot('bar',figsize=(15,5),colormap='Set2')
plt.xticks(fontsize=20)
plt.yticks([])
plt.ylabel('Frequency')
plt.title("Frequency of Nouns in the context of 'Commits Suicide'", fontsize=30)


# In[26]:


index_s = data['headline_text'].str.match(r'(?=.*\bcommits\b)(?=.*\bsuicide\b).*$')
index_farmer = data.loc[index_s]['headline_text'].str.match(r'farmer',case=False)
index_stu = data.loc[index_s]['headline_text'].str.match(r'student',case=False)


# In[27]:


print('Approximately {} percent of suicides reported were student related'.format(round(np.sum(index_stu)/np.sum(index_s),2)*100))


# In[28]:


print("Approximately {} percent of suicides reported were farmer related".format(round(np.sum(index_farmer)/np.sum(index_s),2)*100))


# In[29]:


ind_farmer = data['headline_text'].str.match(r'farmer|farmers',case=False)


# In[30]:


text_f = data.loc[ind_farmer]['headline_text'].tolist()
noun_f = []
verb_f = []
for doc in nlp.pipe(text_f, n_threads=16, batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_f.append(c.text)
            elif c.pos_=='VERB':
                verb_f.append(c.text)
                
    except:
        for c in doc:
            noun_f.append("")
            verb_f.append("")
            


# In[33]:


plt.subplot(1,2,1)
pd.Series(noun_f).value_counts()[2:].head(10).plot(kind='bar',figsize=(20,5),colormap='Set2')
plt.title("Top 10 Nouns in the context of 'Farmer(s)'", fontsize=25)
plt.xticks(size=20, rotation=80)
plt.yticks([])
plt.subplot(1,2,2)
pd.Series(verb_f).value_counts().head(10).plot(kind="bar",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs in the context of 'Farmer(s)'",fontsize=25)
plt.xticks(size=20,rotation=80)
plt.yticks([])


# # Indian Political Scene: BJP vs Congress

# In[34]:


index_bjp = data['headline_text'].str.match(r"bjp.*$",case=False)
index_con = data['headline_text'].str.match(r"congress.*$",case=False)
print('BJP was mentioned {} times'.format(np.sum(index_bjp)))
print('Congress was mentioned {} times'.format(np.sum(index_con)))
print('BJP was mentioned {} times more than congress'. format(np.round(np.sum(index_bjp)/np.sum(index_con),2)))


# In[36]:


import textblob
data_bjp=data.loc[index_bjp].copy()
data_bjp['polarity']=data_bjp['headline_text'].map(lambda x: textblob.TextBlob(x).sentiment.polarity)
pos=" ".join(data_bjp.query("polarity>0")['headline_text'].tolist())
neg=" ".join(data_bjp.query("polarity<0")['headline_text'].tolist())
text=" ".join(data_bjp['headline_text'].tolist())


# In[37]:


from wordcloud import WordCloud,STOPWORDS
import PIL
bjp_mask = np.array(PIL.Image.open('bjp.png'))
wc = WordCloud(max_words=500, mask=bjp_mask,width=5000,height=2500,
              background_color='white', stopwords=STOPWORDS).generate(text)
plt.figure(figsize=(30,15))
plt.imshow(wc)
plt.yticks([])
plt.xticks([])
plt.axis('off')


# In[38]:


# top trigrams
from sklearn.feature_extraction import text
def get_imp(bow, mf, ngram):
    tfidf = text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix = tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)
bow = data_bjp['headline_text'].tolist()
bjp_trigrams = get_imp(bow,mf=5000,ngram=3)


# In[39]:


text_bjp = data_bjp['headline_text'].tolist()
noun_bjp = []
verb_bjp = []
for doc in nlp.pipe(text_bjp, n_threads=16, batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_bjp.append(c.text)
            elif c.pos_=='VERB':
                verb_bjp.append(c.text)
                
    except:
        for c in doc:
            noun_bjp.append("")
            verb_bjp.append("")


# In[40]:


plt.subplot(1,3,1)
bjp_trigrams.head(10).plot(kind='barh', figsize=(20,5),colormap='Set2')
plt.title("Top 10 Trigrams (BJP)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)

pd.Series(noun_bjp).value_counts().head(10).plot(kind='barh',figsize=(20,5),colormap='Set2')
plt.title('Top 10 Nouns (BJP)', fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_bjp).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (BJP)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# What did positive and negative headlines about BJP contain?

# In[45]:


thumbs_up = np.array(PIL.Image.open('thumbsup.jpg'))
wc= WordCloud(max_words=500, mask=thumbs_up,width=5000,height=2500,background_color='white', stopwords=STOPWORDS).generate(pos)
thumbs_dn = np.array(PIL.Image.open('thumbs down.jpg'))
wc1= WordCloud(max_words=500, mask=thumbs_dn,width=5000,height=2500,background_color='white', stopwords=STOPWORDS).generate(neg)

fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(1,2,1)
ax.imshow(wc)
ax.axis('off')
ax.set_title('Positive headlines',fontdict={'fontsize':20})

ax=fig.add_subplot(1,2,2)
ax.imshow(wc1)
ax.axis('off')
ax.set_title("Negative Headlines",fontdict={'fontsize':20})


# In[46]:


bow = data_bjp.query('polarity>0')['headline_text'].tolist()
bjp_trigrams_pos = get_imp(bow,mf=5000,ngram=3)
bow = data_bjp.query('polarity<0')['headline_text'].tolist()
bjp_trigrams_neg = get_imp(bow,mf=5000,ngram=3)


# In[47]:


text_bjp_pos = data_bjp.query('polarity>0')['headline_text'].tolist()
noun_bjp_pos = []
verb_bjp_pos = []
for doc in nlp.pipe(text_bjp_pos, n_threads=16, batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_bjp_pos.append(c.text)
            elif c.pos_=='VERB':
                verb_bjp_pos.append(c.text)
    except:
        for c in doc:
            noun_bjp_pos.append("")
            verb_bjp_pos.append("")


# In[48]:


text_bjp_neg = data_bjp.query('polarity<0')['headline_text'].tolist()
noun_bjp_neg = []
verb_bjp_neg = []

for doc in nlp.pipe(text_bjp_neg,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_bjp_neg.append(c.text)
            elif c.pos_=='VERB':
                verb_bjp_neg.append(c.text)
                
    except:
        for c in doc:
            noun_bjp_neg.append("")
            verb_bjp_neg.append("")


# In[49]:


plt.subplot(1,3,1)
bjp_trigrams_pos.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (BJP+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_bjp_pos).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (BJP+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_bjp_pos).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (BJP+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# What were the headlines about Congress ?

# In[50]:


data_con = data.loc[index_con].copy()
data_con['polarity'] = data_con['headline_text'].map(lambda x:textblob.TextBlob(x).sentiment.polarity)
pos = " ".join(data_con.query('polarity>0')['headline_text'].tolist())
neg = " ".join(data_con.query('polarity<0')['headline_text'].tolist())

text = " ".join(data_con['headline_text'].tolist())


# In[53]:


con_mask = np.array(PIL.Image.open('congress.png'))
wc = WordCloud(max_words=500, mask=con_mask,width=5000,height=2500,background_color='white', stopwords=STOPWORDS).generate(text)
plt.figure( figsize=(30,15))
plt.imshow(wc)
plt.yticks([])
plt.xticks([])
plt.axis("off")


# In[54]:


from sklearn.feature_extraction import text
bow = data_con['headline_text'].tolist()
con_trigrams = get_imp(bow,mf=5000,ngram=3)


# In[55]:


text_con=data_con['headline_text'].tolist()
noun_con=[]
verb_con=[]
for doc in nlp.pipe(text_con,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_con.append(c.text)
            elif c.pos_=="VERB":
                verb_con.append(c.text)
    except:
        for c in doc:
            noun_con.append("") 
            verb_con.append("")


# In[56]:


plt.subplot(1,3,1)
con_trigrams.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (Congress)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_con).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (Congress)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_con).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (Congress)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# In[57]:


wc = WordCloud(max_words=500, mask=thumbs_up,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(pos)
wc1=WordCloud(max_words=500, mask=thumbs_dn,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(neg)
fig=plt.figure(figsize=(30,15))
ax=fig.add_subplot(1,2,1)
ax.imshow(wc)
ax.axis('off')
ax.set_title("Positive Headlines",fontdict={'fontsize':20})
ax=fig.add_subplot(1,2,2)
ax.imshow(wc1)
ax.axis('off')
ax.set_title("Negative Headlines",fontdict={'fontsize':20})


# In[58]:


bow=data_con.query("polarity>0")['headline_text'].tolist()
con_trigrams_pos=get_imp(bow,mf=5000,ngram=3)
bow=data_con.query("polarity<0")['headline_text'].tolist()
con_trigrams_neg=get_imp(bow,mf=5000,ngram=3)


# In[59]:


text_con_pos=data_con.query("polarity>0")['headline_text'].tolist()
noun_con_pos=[]
verb_con_pos=[]
for doc in nlp.pipe(text_con_pos,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_con_pos.append(c.text)
            elif c.pos_=="VERB":
                verb_con_pos.append(c.text)
    except:
        for c in doc:
            noun_con_pos.append("") 
            verb_con_pos.append("")


# In[60]:


text_con_neg=data_con.query("polarity<0")['headline_text'].tolist()
noun_con_neg=[]
verb_con_neg=[]
for doc in nlp.pipe(text_con_neg,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_con_neg.append(c.text)
            elif c.pos_=="VERB":
                verb_con_neg.append(c.text)
    except:
        for c in doc:
            noun_con_neg.append("") 
            verb_con_neg.append("")


# In[61]:


plt.subplot(1,3,1)
con_trigrams_pos.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (Congress+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_con_pos).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (Congress+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_con_pos).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (Congress+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# In[62]:


plt.subplot(1,3,1)
con_trigrams_neg.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (Congress-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_con_neg).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (Congress-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_con_neg).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (Congress-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# # Modi Vs Rahul Gandhi

# In[64]:


index_nm=data['headline_text'].str.match("narendra modi",case=False)
index_rahul=data['headline_text'].str.match("rahul gandhi",case=False)
print("Modi has been mentioned {} times".format(np.sum(index_nm)))
print("Rahul Gandhi has been mentioned {} times".format(np.sum(index_rahul)))


# In[65]:


nm = pd.DataFrame(data['year'].loc[index_nm].value_counts())
r = pd.DataFrame(data['year'].loc[index_rahul].value_counts())
n_r  = pd.concat([nm,r],axis=1)
n_r.columns = ['Modi','Rahul']


# In[66]:


n_r.plot(figsize=(10,10))
plt.title("Mentions of Modi and Rahul over time",fontsize=20)


# Headlines shah rukh apperaed

# In[67]:


index_shah=data['headline_text'].str.match(r'(?=.*\bshah\b)(?=.*\brukh\b).*$',case=False)
data_shah=data.loc[index_shah].copy()
data_shah['polarity']=data_shah['headline_text'].map(lambda x: textblob.TextBlob(x).sentiment.polarity)


# In[68]:


pos=data_shah.query("polarity>0")['headline_text']
neg=data_shah.query("polarity<0")['headline_text']
print("The number of positve headlines were {} times the negative headlines".format(round(len(pos)/len(neg),2)))


# In[70]:


plt.figure(figsize=(8,8))
plt.bar(["Positive","Negative"],[len(pos),len(neg)])
plt.title("Frequency of Positive and Negative News about Shah Rukh",fontsize=20)


# In[71]:


bow=data_shah['headline_text'].str.replace(r'shah|rukh|khan',"",case=False).tolist()
shah_uni=get_imp(bow,mf=5000,ngram=1)
shah_bi=get_imp(bow,mf=5000,ngram=2)
shah_tri=get_imp(bow,mf=5000,ngram=3)


# In[72]:


shah_bi.head(10).plot(kind="barh",figsize=(8,8),colormap="Set2")
plt.title("Most Frequent Bigrams in the Context of Shah Rukh",fontsize=15)


# In[74]:


shah_text=" ".join(bow)
con_mask=np.array(PIL.Image.open('shahrukh.jpg'))
wc = WordCloud(max_words=500, mask=con_mask,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(shah_text)
plt.figure( figsize=(30,15))
plt.imshow(wc)
plt.axis("off")
plt.yticks([])
plt.xticks([])
plt.savefig('./srk.png', dpi=50)
plt.show()


# In[ ]:




