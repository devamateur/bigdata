""" 케글의 fake and real news data를 이용한 가짜뉴스 탐지 
https://www.kaggle.com/code/madz2000/nlp-using-glove-embeddings-99-87-accuracy/notebook """

## 필요한 라이브러리 임포트
import numpy as np
import pandas as pd

# 시각화
import seaborn as sns 
import matplotlib.pyplot as plt

# 자연어처리
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence

# 딥러닝 모델 및 훈련, 검증 라이브러리
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

true = pd.read_csv("True.csv")
false = pd.read_csv("Fake.csv")


# true 데이터에 로이터 통신(Reuters)이 기재되어 있으므로 이를 제거해보자....!
import re

#true['text'] = true['text'].apply(lambda x : re.sub(r'[(Reuters)]', '', x))
true['text'] = true['text'].str.replace('\(Reuters\)', '')

# 지역 이름도 제거하고 싶은데 어떻게 할 지 잘 모르겠음 ... 

false.head()

# 두 데이터에 category 컬럼 만들기(진짜/가짜 여부)
true['category'] = 1
false['category'] = 0

# 둘을 합친 df 데이터프레임
df = pd.concat([true, false])

# counterplot으로 category의 0/1 개수 표시
sns.set_style("darkgrid")
sns.countplot(df.category)

df.head()
df.isna().sum()  # nan인 데이터 개수 확인
df.title.count()
df.subject.value_counts()  # 뉴스의 subject 확인

# 뉴스의 subject별로 가짜/진짜 뉴스 개수 표시
plt.figure(figsize = (12,8))
sns.set(style = "whitegrid",font_scale = 1.2)
chart = sns.countplot(x = "subject", hue = "category" , data = df)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)

# 문자열 데이터를 한 컬럼으로 합치기
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']

# 불용어 설정
stop = set(stopwords.words('english'))   # nltk.corpus의 stopwords
punctuation = list(string.punctuation)   # string.punctuation: 구두점(특수문자)
stop.update(punctuation)   # 불용어에 구두점도 포함


# 데이터 cleaning - df['text']에서 필요없는 단어, 문자 등 제거
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets 괄호 제거
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text 불용어 제거
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)
#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text
#Apply function on review column
df['text']=df['text'].apply(denoise_text)



# 워드클라우드
# - 진짜 뉴스 데이터
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , 
               stopwords = STOPWORDS).generate(" ".join(df[df.category == 1].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.axis("off")

# - 가짜 뉴스 데이터
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , 
               stopwords = STOPWORDS).generate(" ".join(df[df.category == 0].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.axis("off")


""" 단어 개수 시각화
# 진짜/가짜 뉴스 데이터에서 단어 개수 - 가짜 뉴스가 더 많은 단어 사용
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=df[df['category']==1]['text'].str.len()
ax1.hist(text_len,color='red')
ax1.set_title('Original text')
text_len=df[df['category']==0]['text'].str.len()
ax2.hist(text_len,color='green')
ax2.set_title('Fake text')
fig.suptitle('Characters in texts')
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=df[df['category']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='red')
ax1.set_title('Original text')
text_len=df[df['category']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='green')
ax2.set_title('Fake text')
fig.suptitle('Words in texts')
plt.show()

# 평균 단어 길이
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
word=df[df['category']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('Original text')
word=df[df['category']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
ax2.set_title('Fake text')
fig.suptitle('Average word length in each text')어
"""

# get_corpus(): 단어에서 코퍼스를 가져옴
def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words
corpus = get_corpus(df.text)
corpus[:5]

# Counter 이용
from collections import Counter 
counter = Counter(corpus)
most_common = counter.most_common(10)  # 가장 흔한 상위 10개 단어
most_common = dict(most_common)
most_common

# CountVectorizer 이용 
# - 이 코드를 참고해서 우리는 이 부분을 TfidfVectorizer로 수정하면 좋을듯
from sklearn.feature_extraction.text import CountVectorizer
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# 가장 흔한 상위 10개 단어 시각화
"""plt.figure(figsize = (16,9))
most_common_uni = get_top_text_ngrams(df.text,10,1)
most_common_uni = dict(most_common_uni)
sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))

plt.figure(figsize = (16,9))
most_common_bi = get_top_text_ngrams(df.text,10,2)
most_common_bi = dict(most_common_bi)
sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))

plt.figure(figsize = (16,9))
most_common_tri = get_top_text_ngrams(df.text,10,3)
most_common_tri = dict(most_common_tri)
sns.barplot(x=list(most_common_tri.values()),y=list(most_common_tri.keys()))
"""

# train/test 데이터 분리
x_train,x_test,y_train,y_test = train_test_split(df.text,df.category,random_state = 0)

max_features = 10000  # feature: 예측에 영향을 미치는 변수, 여기서는 텍스트
maxlen = 300

# 각 단어를 tokenizer의 인덱스로 매핑?
tokenizer = text.Tokenizer(num_words=max_features)  # keras의 text.Tokenizer
tokenizer.fit_on_texts(x_train)  # x_train으로 훈련
tokenized_train = tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(x_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

# Glove는 생략.. 
# - word2vec 등으로 대체 가능

# 딥러닝 모델 설정
batch_size = 256  # 한 번의 훈련 당 batch_size
epochs = 10   # 훈련 횟수
embed_size = 100
# keras.callback의 ReduceLROnPlateau()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

# 딥러닝 모델
model = Sequential()
#Non-trainable embeddidng layer
model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen, trainable=False))
#LSTM 
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 훈련
history = model.fit(x_train, y_train, batch_size = batch_size , validation_data = (X_test,y_test) , 
                    epochs = epochs , callbacks = [learning_rate_reduction])

# 모델 검증
# - 모델의 정확도
print("Accuracy of the model on Training Data is - " , model.evaluate(x_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")

# - train/test 데이터의 정확도, loss를 시각화
epochs = [i for i in range(10)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

# - 샘플로 5개 예측값
pred = model.predict_classes(X_test)
pred[:5]

# - 모델 평가지표 한번에 보기
print(classification_report(y_test, pred, target_names = ['Fake','Not Fake']))

# 혼동행렬 그리기
cm = confusion_matrix(y_test,pred)
cm

cm = pd.DataFrame(cm , index = ['Fake','Original'] , columns = ['Fake','Original'])
plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Fake','Original'] , yticklabels = ['Fake','Original'])
plt.xlabel("Predicted")
plt.ylabel("Actual")