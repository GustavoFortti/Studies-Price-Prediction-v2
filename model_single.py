import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class model_s():
    
    def __init__(self, df, csv):

        aux = []
        for i in df.values:
            aux.append(str(i).replace("['", "").replace("']", "").split("\\t"))
        
        # aux = [ j[:5] if '.' in j else j for j in [i for i in aux]]
        data = pd.DataFrame(aux, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        data = data.set_index('Date')
        for i in data:
            data[i] = data[i].astype(float)


        data['Close_Tomorrow'] = data['Close'].shift(-1)
        data['Return'] = data['Close_Tomorrow'] - data['Close']
        
        scaler = StandardScaler()
        for c in data.columns:
            data[c+'_Norm'] = data[c].to_numpy().reshape(-1, 1)
        data = data.dropna()
        
        train = data.iloc[:40000, :]
        train_x = train[['Open_Norm','High_Norm','Low_Norm','Close_Norm','Volume_Norm']].to_numpy()
        train_y = train[['Return_Norm']].to_numpy()
        val = data.iloc[40000:48000, :]
        val_x = val[['Open_Norm','High_Norm','Low_Norm','Close_Norm','Volume_Norm']].to_numpy()
        val_y = val[['Return_Norm']].to_numpy()
        test = data.iloc[48000:, :]
        test_x = test[['Open_Norm','High_Norm','Low_Norm','Close_Norm','Volume_Norm']]
        test_y = test[['Return_Norm', 'Return']]

        model = models.Sequential()
        model.add(layers.Dense(30, input_dim = train_x.shape[1], activation = "relu"))
        model.add(layers.Dense(30, activation ="relu"))
        model.add(layers.Dense(30, activation ="relu"))
        model.add(layers.Dense(1))  

        model.summary()
        model.compile(
            optimizer = 'adam',
            loss = 'mean_squared_error', #mean_squared_error
        )
        results = model.fit(
            train_x, train_y,
            epochs= 3,
            batch_size = 128,
            validation_data = (val_x, val_y),
            # verbose = 0
        )

        # plt.plot(results.history["loss"], label="loss")
        # plt.plot(results.history["val_loss"], label="val_loss")
        # plt.legend()
        # plt.show()

        pred = model.predict(test_x)
        # pred = scaler.inverse_transform(pred)

        pred = [float(str(i)[1:8]) for i in pred]
        test_y['Return'] = [float(str(i)[0:7]) for i in test_y['Return']]
        test_y['Return_Predicted'] = pred
        test_y = test_y.drop(columns=['Return_Norm'])

        test_y['Movement_Predicted'] = [ 'Up' if p > 0 else 'Down' for p in pred]
        test_y['Movement_Real'] = [ 'Up' if m > 0 else 'Down' for m in test_y.loc[:,'Return']]
        
        test_y['Hit'] = [ 1 if m[1]['Movement_Real'] == m[1]['Movement_Predicted'] else 0 for m in test_y.iterrows()]
        test_y['Investiment_Return'] = [m[1]['Return'] if m[1]['Movement_Predicted'] == 'Up' else 0 for m in test_y.iterrows()]
        
        self.csv = csv
        self.test_y = test_y
        self.acc = test_y.loc[:, 'Hit'].mean()

        print(test_y)
        print(data)
        print(pred)
        # plt.plot(data.iloc[-10:, 0:1].values)
        # plt.plot(pred[-10:])
        # plt.legend()
        # plt.show()

        # print('Acc: ',test_y.loc[:, 'Hit'].mean())
        # print('Strategy Return:  R$',test_y.loc[:, 'Investiment_Return'].sum())
        
        # print(' Buy & Hold: R$',test_y['Return'].sum())
        # print('\nInvestiment: R$\n', data.iloc[45000:, :].head(1)['Open'])
       
    def out(self):
        self.test_y.to_csv('out_' + self.csv +  '_' + str(self.acc)[:5] + '_.csv')  