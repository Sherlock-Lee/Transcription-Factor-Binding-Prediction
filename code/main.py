import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.metrics import binary_accuracy
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import datetime



def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # plot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # plot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()

###neural network model
def nn_model(train_x,train_y,test_x):
	model = Sequential()
	model.add(Dense(28,input_shape = (train_x.shape[-1],),activation = 'relu'))
	model.add(Dropout(0.4))
	model.add(Dense(4,activation = 'relu'))
	model.add(Dense(2,activation='sigmoid'))

	model.summary()
	earlystop = EarlyStopping(monitor = 'loss',patience = 3)
	model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

	hist = model.fit(train_x,train_y,
		# validation_data = (val_x,val_y),
		validation_split = 0.1,
		batch_size = 8,
		callbacks = [earlystop],
		epochs = 200)
	# visualize the procedure
	training_vis(hist)
	y_pre = model.predict(test_x)
	y_pre = np.argmax(y_pre,1)
	# print(y_pre[:3])
	return y_pre


def main():
	train_dir = "../data/train.csv"
	test_dir = "../data/test.csv"
	df_train = pd.read_csv(train_dir)
	df_test = pd.read_csv(test_dir)
	sub = pd.DataFrame()
	sub['id'] = df_test['id']
	df = pd.concat([df_train.iloc[:,1:-1],df_test.iloc[:,1:]])
	y = df_train['label']
	trian_row = df_train.shape[0]
	print('columns ',df.columns)
	print("df head")
	print(df.head())
	gene_map = {"A":0,"T":1,"C":2,"G":3}
	df['sequence'] = df['sequence'].map(list)
	df['sequence'] = df['sequence'].map(lambda x : [gene_map[i] for i in x])
	for i in range(14):
		df['sequence' + str(i)] = list(map(lambda x:x[i],df['sequence']))
	del df['sequence']
	print("after pre ")
	print(df.head())
	train = df.iloc[:trian_row:,:]
	test = df.iloc[trian_row:,:]
	y = to_categorical(y,2)
	print('train shape is',train.shape)
	print('test shape is',test.shape)
	sub["prediction"] = nn_model(train,y,test)
	sub['prediction'] = [1 if i > 0.5 else 0 for i in sub['prediction']]
	sub_name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	sub.to_csv("../result/" + sub_name +".csv",index = None)
	print(sub.head())


if __name__ == '__main__':
	main()