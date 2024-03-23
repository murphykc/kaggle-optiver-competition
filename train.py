import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from src.feature_engineering import run
from src.model import SAINT


def fit_model(X_train, X_val, y_train, y_val):
    model = Sequential([
            Input(shape = (X_train.shape[1]),batch_size=256),
            SAINT(1, df['stock_id'].max()+1, df['date_id'].max()+1, 8, 4, 4),
            Dense(1 ,activation='linear')
        ])
    trainds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    trainds = trainds.batch(256, drop_remainder = True)
    
    valds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    valds = valds.batch(256, drop_remainder = True)
    
    model.compile(tf.keras.optimizers.Adam(0.001), 'mae',metrics=['mae'])
    model.summary()
    model.fit(trainds,epochs=20,batch_size=256,validation_data=valds)
    return model
    

if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    df = run(df)

    df.dropna(subset = 'target', inplace=True)
    df.fillna(0, inplace=True)

    y= df['target'].values
    x = df.drop(['target','time_id','row_id'],axis=1).values

    X_train, X_val, y_train, y_val = train_test_split(x,y, test_size=0.2, random_state=42)
    model = fit_model(X_train, X_val, y_train, y_val)