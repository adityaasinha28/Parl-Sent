# Parl-Sent
Sentiment Analysis of Parliamentary Debates Using Deep Learning

Dataset
----
We are going to use the [HanDeSeT](https://data.mendeley.com/datasets/xsvp45cbt4/2) dataset which is a dataset of annotated Hansard speeches.

The dataset contains texts (1251 instances) as well as metadata in the form of motion/govt-opposition votes, party ID and manual votes/speech classification labels. 

Preprocessing: 
---
Tokenisation, Stopword Removal (including additional parliamentary stopwords from this [PhD](https://livrepository.liverpool.ac.uk/19793/)), Lemmatisation, removal of numbers, converting to lower case. 


Word Embeddings
---
We experiment with three common word embeddings known for sentiment analysis 

1. `Glove.6B.100 dimensional` 
2. `Glove.42B.300 dimensional`
3. `fast-text crawl.300d.2M`

Glove word embeddings are available [here](https://nlp.stanford.edu/projects/glove/) and fast-text word embeddings are available [here](https://fasttext.cc/docs/en/english-vectors.html)

Glove word embeddings show better results than fasttext and hence the uploaded notebooks use architectures with Glove embeddings. 

Dependencies include:
----
1. `numpy`
2. `keras`
3. `tensorflow`

Architecture of the Network
---
We will test out multiple different architectures for sentiment analysis as described below along with their results on the dataset. All results are using `10-fold-cross-validation`. Best results are reported for a particular architecture with a variation of embeddings and metadata - `id+motion+party_id`

MLP based Dense Network
---

Since the standard paper has the highest accuracy and F1 score with an MLP classifier, a dense network with 3 layers and 100 units each with BatchNormalisation and Dropout came closest to the highest accuracy. 
The model is as follows: 
```
	model.add(BatchNormalization(input_shape=tuple([X_train.shape[1]])))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(BatchNormalization())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(BatchNormalization())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(BatchNormalization())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(1, activation='sigmoid'))
	#model.add(Dropout(0.5))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

```
This model gave an accuracy of 0.941 and an F1 score of 0.943

3-layer GRU architecture
---
In terms of the Deep Learning architectures, this architecture performed the best. 
The model is as 
```
	model.add(Embedding(200, 128,input_length = (X_new_train_data.toarray()).shape[1]))
    model.add(GRU(units=16, name = "gru_1",return_sequences=True))
    model.add(GRU(units=8, name = "gru_2" ,return_sequences=True))
    model.add(GRU(units=4, name= "gru_3"))
    model.add(Dense(1, activation='sigmoid',name="dense_1"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
This gave an accuracy of 89.21% and an F1 score of 0.89

Stacked Bidirectional LSTM model
---

This model gave a best accuracy of 70.4% with an F1 score of 0.69

```
model1.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1],weights=[embedding_matrix],trainable=True))
model1.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model1.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model1.add(Dense(num_classes,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
```

1DCNN and Multilayer CNN-GRU model
---

The better of the two models was the CNN-GRU model which gave an accuracy of 82.4% with an F1 score of 0.828

The multilayer models is as follows
```
inp = Input((X_train.shape[1],))
x =  Embedding(max_features, embed_dim,input_length = X_train.shape[1])(inp)
x1 = x
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(GRU(256, return_sequences=True))(x)
x = Conv1D(256, 7, strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
    
x2 = x
x = SpatialDropout1D(0.)(x)
x = Bidirectional(GRU(128, return_sequences=True))(x)
x = Conv1D(128, 7, strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
    
x3 = x
x = SpatialDropout1D(0.)(x)
x = Bidirectional(GRU(64, return_sequences=True))(x)
x = Conv1D(64, 7, strides=2)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
    
x_o1 = GlobalMaxPooling1D()(x)
x_o2 = Bidirectional(GRU(embed_dim, return_sequences=False))(x1)
x_o3 = Bidirectional(GRU(256, return_sequences=False))(x2)
x_o4 = Bidirectional(GRU(128, return_sequences=False))(x3)
x_o5 = Bidirectional(GRU(64, return_sequences=False))(x)
    
x = Concatenate()([x_o1, x_o2, x_o3, x_o4, x_o5])
x = Dense(128, activation='tanh')(x)
out = Dense(1, activation='sigmoid')(x)
    
model = Model(inputs=inp, outputs=out)
```

Bidirectional LSTM with Attention
---

LSTM with attention is known to give good results with scarce data. Since our data is close to 1200 instances, typical Deep Learning Architectures wouldnt perform that well. 

The Attention model flatlined after a few epochs and gave a best accuracy of 68.7% with an F1 score of 0.71

Since Attention does not come in-house with keras, an Attention function is written separately inspired from [this kaggle kernel](https://www.kaggle.com/takuok/bidirectional-lstm-and-attention-lb-0-043). The Attention layer is then included in the model as follows:

```
inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, input_length=maxlen)(inp)
    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(x)
    x = Attention(maxlen)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
```




