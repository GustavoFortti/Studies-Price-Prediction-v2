
        # with tf.device('/device:GPU:0'):

        # train_x, train_y = np.array(train_x), np.array(train_y)
        # test_x = np.array(test_x)
        # test_x = np.reshape(test_x, (test_x.shape[0], 1, 1))
        # train_x = np.reshape(train_x, (train_x.shape[0], 1, 1))
        
        # # train_y = np.reshape(train_y, (train_y.shape[0], 1, 1))
        # # test_y = np.reshape(test_y, (test_y.shape[0], 1, 1))

        # # val_x, val_y = np.array(val_x), np.array(val_y)
        # # val_x = np.reshape(val_x, (val_x.shape[0], 1, 1))
        # # val_y = np.reshape(val_y, (val_y.shape[0], 1, 1))

        # print(train_x.shape)
        # print(test_x.shape)
        # print(train_y.shape)
        # print(test_y.shape)

        # n_features = train_x.shape[1]
        # model = models.Sequential()
        # model.add(layers.LSTM(100,activation='relu',input_shape=(1,1)))
        # model.add(layers.Dense(n_features))

        # #Model summary
        # model.summary()
        # #Compiling
        # model.compile(optimizer='adam', loss = 'mse')

        # #Training
        # model.fit(train_x, train_y, epochs = 5, batch_size=30)
        # # scaler.scale_

        # y_pred = model.predict(test_x)
        # y_pred = scaler.inverse_transform(test_y)
        # print(y_pred[:10])

        # test_y = np.array(test_y).reshape(-1,1)
        # test_y = scaler.inverse_transform(test_y)
        # print(test_y[:10])

        # plt.figure(figsize=(10,5))
        # plt.title('Foreign Exchange Rate of India')
        # plt.plot(test_y , label = 'Actual', color = 'g')
        # plt.plot(y_pred , label = 'Predicted', color = 'r')
        # plt.legend()

        # mean_squared_error(test_y, y_pred)


        # regresor = models.Sequential()

        # regresor.add(layers.LSTM(units=50, return_sequences=True,  input_shape=(train_x.shape[1], 1)))
        # regresor.add(layers.Dropout(0.2))

        # regresor.add(layers.LSTM(units=50, return_sequences=True))
        # regresor.add(layers.Dropout(0.2))

        # regresor.add(layers.Dense(units=1))

        # regresor.compile(optimizer='adam', loss='mse')

        # results = regresor.fit(train_x, train_y, epochs=1000, batch_size=32, validation_data = (val_x, val_y),)

        ###################
        # model = models.Sequential()

        # model.add(layers.LSTM(76, input_shape=(train_x.shape[1], 1), return_sequences = True))
        # model.add(layers.Dropout(0.2))
        # model.add(Dense(1))
        # model.compile(loss="mse", optimizer='Adam')
        ###################