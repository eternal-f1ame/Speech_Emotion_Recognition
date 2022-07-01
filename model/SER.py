# %% [markdown]
# # Speech Emotion Recognition
# ## PRML Course Project

# %% [markdown]
# ### Loading the Important libraries and Modules

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import os
import glob
import pickle

from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier,LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score,classification_report

from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Add,LSTM,Dense,Conv1D,InputLayer,Activation,InputLayer,BatchNormalization,Flatten,Dropout,Conv1D,MaxPooling1D
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
tf.config.run_functions_eagerly(True)

sc = StandardScaler()
enc = LabelEncoder()
mms = MinMaxScaler()
oh = OneHotEncoder()
sns.set_style('darkgrid')

emotions_encoder = {
    '01': 'Neutral',
    '02': 'Calm',
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fearful',
    '07': 'Disgust',
    '08': 'Surprised'
}

observed_emotions = [x for x in emotions_encoder.values()]

# %% [markdown]
# ## The created Pipeline <code>Feature()</code> 
# ### Takes in the series of Paths of the audio files in the device, <br>loads them into time series data in the <code>extract_features()</code> through <code>engineer()</code>function, and further extracts the relevant features out of the time series data by :-
# * Frequency Sampling
# * Fourier Analysis
# * Slope Change for the wave
# ### All of which is facilitated by <code>extract_feature()</code> and its input parameters.

# %%
class Feature():

    def __init__(self, mfcc=True, chroma=True, mel=True, zcr=True, spread=True, mean=True):
        self.mfcc = mfcc
        self.chroma = chroma
        self.mel = mel
        self.zcr = zcr
        self.spread = spread
        self.mean = mean

    def extract_feature(self,file_name,augmented = str()):
        data = []    
        X,sample_rate = librosa.load(path=file_name,duration=2.5,offset=0.5)
        if augmented.lower() == "noised":
            noise_amp = 0.05*np.random.uniform()*np.amax(X)
            X = X + noise_amp*np.random.normal(size=X.shape[0])

        elif augmented.lower() == "stretched":
            X = librosa.effects.time_stretch(X, 0.8)

        elif augmented.lower() == "rolled":
            shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
            X = np.roll(X, shift_range)

        elif augmented.lower() == "pitchedd":
            sampling_rate = self.sampling_rate
            X = librosa.effects.pitch_shift(X, sampling_rate, self.pitch_factor)

        data.append(X)
        self.sample_rate = sample_rate
        result=np.array([])

        if self.mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))

        if self.chroma:
            stft=np.abs(librosa.stft(X))
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate,n_chroma=32).T,axis=0)
            result=np.hstack((result, chroma))

        if self.mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate,n_fft=2048).T,axis=0)
            result=np.hstack((result, mel))

        if self.zcr:
            Z = np.mean(librosa.feature.zero_crossing_rate(y=X),axis=1)
            result=np.hstack((result, Z))

        if self.spread:
            var = np.var(X)
            result=np.hstack((result, var))

        if self.mean:
            mean = np.mean(X)
            result=np.hstack((result, mean))

        return result

    def engineer(self,Paths=[],augmented = str()):
        X = []        
        self.Paths = Paths
        for file in Paths:
            feature=self.extract_feature(file)
            X.append(feature)
        self.features = X

        return np.array(X)



# %% [markdown]
# ### Loading the paths in a pandas series and encoding labels to usable formats.

# %%
X,Y=[],[]

for file in glob.glob("Data/Actor_*/*.wav"):
    file_name=os.path.basename(file)
    emotion=emotions_encoder[file_name.split("-")[2]]
    X.append(file)
    Y.append(emotion)
Y_ = LabelEncoder().fit_transform(Y)
Y_oh = OneHotEncoder().fit_transform(Y_.reshape(-1,1)).toarray()



# %% [markdown]
# #### Class Frequencies

# %%
plt.figure(figsize=(16,6))
sns.countplot(x = Y)
plt.ylabel("Frequency of occurence",fontdict={'fontsize':15})
plt.xlabel("Emotion in the Audio",fontdict={'fontsize':15})
plt.tight_layout()
plt.show()

# %% [markdown]
# # Loading Data and Models

# %% [markdown]
# ## ALL Data ( MFCC , Chroma , MEL )

# %% [markdown]
# ### Loading all the relevant features using the pipeline to train supervised models on them
# 

# %%
Feat_all = Feature(spread=False,zcr=False,mean=False)
x_train,x_test,y_train,y_test = train_test_split(X,Y_,train_size=0.8)
Non_Augmented = Feat_all.engineer(x_train)
alltest = Feat_all.engineer(x_test)


# %% [markdown]
# ### Augmented versions of the Data are loaded too

# %%
noised = Feat_all.engineer(x_train,augmented="noised")
stretched = Feat_all.engineer(x_train,augmented='stretched')
rolled = Feat_all.engineer(x_train,augmented='rolled')
pitched = Feat_all.engineer(x_train,augmented='pitched')

# %% [markdown]
# ## Shallow MLP

# %%
def ShallowMLP(Xtrain,Ytrain,Xtest,Ytest):
    mod1=MLPClassifier(hidden_layer_sizes=(256,128,64,32),validation_fraction=0.2,early_stopping=True,learning_rate_init=0.001)
    mod1.fit((sc.fit_transform(Xtrain)),Ytrain)
    print(classification_report(mod1.predict(sc.transform(Xtest)),Ytest))

# %% [markdown]
# ## SVM (RBF Kernel)

# %%
def SVM_Model(Xtrain,Ytrain,Xtest,Ytest):
    svm = SVC(C = 100)
    svm.fit((sc.fit_transform(Xtrain)),Ytrain)
    print(classification_report(svm.predict(sc.transform(Xtest)),Ytest))


# %% [markdown]
# ## Logistic Regression

# %%
def Log_Regressor(Xtrain,Ytrain,Xtest,Ytest):
    LR = LogisticRegression(max_iter=10000)
    LR.fit(sc.fit_transform(Xtrain),Ytrain)
    print(classification_report(LR.predict(sc.transform(Xtest)),Ytest))

# %% [markdown]
# ## Passive Aggressive Classifier 

# %%
def PAClassifier(Xtrain,Ytrain,Xtest,Ytest):
    pac = PassiveAggressiveClassifier(max_iter=10000)
    pac.fit(sc.fit_transform(Xtrain),Ytrain)
    print(classification_report(pac.predict(sc.transform(Xtest)),Ytest))

# %% [markdown]
# ## K Nearest Neighbour

# %%
from sklearn.neighbors import KNeighborsClassifier

def KNN(Xtrain,Ytrain,Xtest,Ytest):
    r = [1]
    acc = []
    for k in r:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit((sc.fit_transform(Xtrain)),Ytrain)
        print(classification_report(knn.predict(sc.transform(Xtest)),Ytest))   

# %% [markdown]
# ## One Dimensional CNN

# %%
def One_D_CNN(Xtrain,Ytrain,Xtest,Ytest):

    model = Sequential()
    model.add(InputLayer(input_shape=(200,1)))
    model.add(Conv1D(32,3,padding="same",activation=relu))
    model.add(Conv1D(64,3,padding="same",activation=relu))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4,strides=2))
    model.add(Conv1D(128,3,padding="same",activation=relu))
    model.add(Flatten())
    model.add(Dense(8,activation=softmax))
    model.compile(loss=categorical_crossentropy,optimizer=Adam(),metrics="accuracy")
    print(model.summary())

    model.fit(np.expand_dims(sc.fit_transform(Xtrain), -1),to_categorical(Ytrain),
            validation_data=(np.expand_dims(sc.fit_transform(Xtest), -1),to_categorical(Ytest)),
            epochs=40, batch_size=128,)
    y_pred = model.predict(np.expand_dims(sc.fit_transform(Xtest), -1))
    print(classification_report(np.argmax(y_pred, -1), Ytest))


# %% [markdown]
# ## XGBoost

# %%
from xgboost import XGBClassifier
def XGBC(Xtrain,Ytrain,Xtest,Ytest):
    xgb = XGBClassifier()
    xgb.fit(Xtrain,Ytrain)
    Ypred = xgb.predict(Xtest)
    print(classification_report(Ypred,Ytest))



# %% [markdown]
# ## LSTM Model

# %%
def LSTM_model(Xtrain,Ytrain,Xtest,Ytest):
    def create_model_LSTM():
        model = Sequential()
        model.add(LSTM(128, return_sequences=False, input_shape=(Xtrain.shape[1], 1)))
        model.add(Dense(64))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Dropout(0.4))
        model.add(Activation('relu'))
        model.add(Dense(8,activation=relu))
        model.add(Dense(1))
        model.add(Activation('softmax'))
        model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
        return model

    model = create_model_LSTM()
    model.fit(np.expand_dims(Xtrain,-1),Ytrain,epochs=50)
    ypred = model.predict(np.expand_dims(Xtest,-1))
    print(classification_report(ypred,Ytest))



# %% [markdown]
# ## SVM on Original Data

# %%
rbf_acc_train = []
rbf_acc_test = []
max_const = 50
test = Feat_all.engineer(x_test)
for gen_const in range(0,max_const,2):
    rbf_model = SVC(C=(gen_const+1),kernel='rbf')
    rbf_model.fit(sc.fit_transform(Non_Augmented),y_train)
    rbf_results_train = rbf_model.predict(sc.transform(Non_Augmented))
    rbf_results_test = rbf_model.predict(sc.transform(test))
    rbf_acc_train.append(accuracy_score(rbf_results_train,y_train))
    rbf_acc_test.append(accuracy_score(rbf_results_test,y_test))
rbfm = np.argmax(rbf_acc_test)

# %%
plt.figure('Accuracy plot RBF',figsize=(24,8))

sns.set_style('darkgrid')
plt.subplot(1,2,1)
sns.lineplot(x=[i+1 for i in range(0,max_const,2)],y=rbf_acc_train)
plt.xticks(ticks=[i+1 for i in range(0,max_const,2)])
plt.xlabel("Generalization constant C",fontdict={'fontsize':15})
plt.ylabel("Accuracy Scores",fontdict={'fontsize':15})
plt.ylim([0.35, 1])
plt.title('Training Accuracies',fontdict={'fontsize':20})
plt.tight_layout()

plt.subplot(1,2,2)
sns.lineplot(x=[i+1 for i in range(0,max_const,2)],y=rbf_acc_test)
plt.xticks(ticks=[i+1 for i in range(0,max_const,2)])
plt.xlabel("Generalization constant C",fontdict={'fontsize':15})
plt.ylabel("Accuracy Scores",fontdict={'fontsize':15})
plt.ylim([0.35, 1])
plt.title('Testing Accuracies',fontdict={'fontsize':20})
plt.tight_layout()
plt.show()


# %% [markdown]
# # In the following code the models are trained in the order
# * Shallow MLP
# * SVM
# * Logistic Regresseor
# * Passive Aggressive Classifier 
# * KNN
# * One D CNN
# * XGBoost
# * LSTM
# 

# %% [markdown]
# ## MFCC
# 

# %% [markdown]
# ### original Data

# %%
Feat_mf = Feature(spread=False,zcr=False,mean=False,chroma=False,mel=False)
mfx_train,mfx_test,mfy_train,mfy_test = train_test_split(X,Y_,train_size=0.8)
mfNon_Augmented = Feat_mf.engineer(mfx_train)
mftest = Feat_mf.engineer(mfx_test)

# %%
ShallowMLP(mfNon_Augmented,mfy_train,mftest,mfy_test)

# %%
SVM_Model(mfNon_Augmented,mfy_train,mftest,mfy_test)

# %%
Log_Regressor(mfNon_Augmented,mfy_train,mftest,mfy_test)

# %%
PAClassifier(mfNon_Augmented,mfy_train,mftest,mfy_test)

# %%
KNN(mfNon_Augmented,mfy_train,mftest,mfy_test)

# %%
One_D_CNN(mfNon_Augmented,mfy_train.reshape(-1,1),mftest,mfy_test.reshape(-1,1))

# %%
XGBC(mfNon_Augmented,mfy_train,mftest,mfy_test)

# %%

LSTM_model(mfNon_Augmented,mfy_train,mftest,mfy_test)

# %% [markdown]
# ## MEL Coefficients

# %% [markdown]
# ### Original Data

# %%
Feat_ml = Feature(spread=False,zcr=False,mean=False,chroma=False,mfcc=False)
mlx_train,mlx_test,mly_train,mly_test = train_test_split(X,Y_,train_size=0.8)
mlNon_Augmented = Feat_ml.engineer(mlx_train)
mltest = Feat_ml.engineer(mlx_test)

# %%
ShallowMLP(mlNon_Augmented,mly_train,mltest,mly_test)

# %%
SVM_Model(mlNon_Augmented,mly_train,mltest,mly_test)

# %%
Log_Regressor(mlNon_Augmented,mly_train,mltest,mly_test)

# %%
PAClassifier(mlNon_Augmented,mly_train,mltest,mly_test)

# %%
KNN(mlNon_Augmented,mly_train,mltest,mly_test)

# %%

One_D_CNN(mlNon_Augmented,mly_train,mltest,mly_test)

# %%
XGBC(mlNon_Augmented,mly_train,mltest,mly_test)

# %%
LSTM_model(mlNon_Augmented,mly_train,mltest,mly_test)

# %% [markdown]
# ## Chroma Vectors

# %% [markdown]
# ### Original Data

# %%
Feat_cv = Feature(spread=False,zcr=False,mean=False,mel=False,mfcc=False)
cvx_train,cvx_test,cvy_train,cvy_test = train_test_split(X,Y_,train_size=0.8)
cvNon_Augmented = Feat_cv.engineer(cvx_train)
cvtest = Feat_cv.engineer(cvx_test)

# %%
ShallowMLP(cvNon_Augmented,cvy_train,cvtest,cvy_test)

# %%
SVM_Model(cvNon_Augmented,cvy_train,cvtest,cvy_test)

# %%
Log_Regressor(cvNon_Augmented,cvy_train,cvtest,cvy_test)

# %%
PAClassifier(cvNon_Augmented,cvy_train,cvtest,cvy_test)

# %%
KNN(cvNon_Augmented,cvy_train,cvtest,cvy_test)

# %%
One_D_CNN(cvNon_Augmented,cvy_train,cvtest,cvy_test)

# %%
XGBC(cvNon_Augmented,cvy_train,cvtest,cvy_test)

# %%
LSTM_model(cvNon_Augmented,cvy_train,cvtest,cvy_test)

# %% [markdown]
# # Training on Comined Data

# %% [markdown]
# ## Original

# %%
ShallowMLP(Non_Augmented,y_train,alltest,y_test)

# %%
SVM_Model(Non_Augmented,y_train,alltest,y_test)

# %%
Log_Regressor(Non_Augmented,y_train,alltest,y_test)

# %%
PAClassifier(Non_Augmented,y_train,alltest,y_test)

# %%
KNN(Non_Augmented,y_train,alltest,y_test)

# %%
One_D_CNN(Non_Augmented,y_train,alltest,y_test)

# %%
XGBC(Non_Augmented,y_train,alltest,y_test)

# %%
LSTM_model(Non_Augmented,y_train,alltest,y_test)

# %% [markdown]
# ## Augmented

# %%
ShallowMLP(np.r_[Non_Augmented,noised,rolled,stretched,pitched],np.r_[y_train,y_train,y_train,y_train,y_train],alltest,y_test)

# %%
SVM_Model(np.r_[Non_Augmented,noised,rolled,stretched,pitched],np.r_[y_train,y_train,y_train,y_train,y_train],alltest,y_test)

# %%
Log_Regressor(np.r_[Non_Augmented,noised,rolled,stretched,pitched],np.r_[y_train,y_train,y_train,y_train,y_train],alltest,y_test)

# %%
PAClassifier(np.r_[Non_Augmented,noised,rolled,stretched,pitched],np.r_[y_train,y_train,y_train,y_train,y_train],alltest,y_test)

# %%
KNN(np.r_[Non_Augmented,noised,rolled,stretched,pitched],np.r_[y_train,y_train,y_train,y_train,y_train],alltest,y_test)

# %%
One_D_CNN(np.r_[Non_Augmented,noised,rolled,stretched,pitched],np.r_[y_train,y_train,y_train,y_train,y_train],alltest,y_test)

# %%
XGBC(np.r_[Non_Augmented,noised,rolled,stretched,pitched],np.r_[y_train,y_train,y_train,y_train,y_train],alltest,y_test)

# %% [markdown]
# ## END


