from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
import tensorflow as tf

class hysterisis_curve:
    def __init__(self, x_tr, y_tr, x_te, y_te):
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.strategy = tf.distribute.MirroredStrategy()  # multi-GPU strategy
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def lstm_model(self):
        lstm_model = Sequential()
        lstm_model.add(LSTM(128, input_shape=(2, 1), return_sequences=True))
        lstm_model.add(LSTM(64))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return lstm_model
    
    def lstm_fit(self, epochs=100, batch_size=1):
        with self.strategy.scope():
            self.lstm_model_instance = self.lstm_model()
            self.lstm_model_instance.fit(self.x_tr, self.y_tr, epochs = epochs, batch_size = batch_size)
        return self.lstm_model_instance
    
    def lstm_predict(self):
        y_pred = self.lstm_model_instance.predict(self.x_te)
        return y_pred
    
    def dnn_model(self):
        dnn_model = Sequential()
        dnn_model.add(Dense(128, input_dim=2, activation='relu'))
        dnn_model.add(Dense(64, activation='relu'))
        dnn_model.add(Dense(1))
        dnn_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return dnn_model
    
    def dnn_fit(self, epochs=100, batch_size=1):
        self.dnn_model_instance = self.dnn_model()
        self.dnn_model_instance.fit(self.x_tr, self.y_tr, epochs = epochs, batch_size = batch_size)
        return self.dnn_model_instance
    
    def dnn_predict(self):
        y_pred = self.dnn_model_instance.predict(self.x_te)
        return y_pred


class PINN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=512, activation=tf.nn.tanh, dtype = tf.float32)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=256, activation=tf.nn.tanh, dtype = tf.float32)
        self.output_layer = tf.keras.layers.Dense(units=output_dim)

    def call(self, inputs):
        x = self.hidden_layer_1(inputs)
        x = self.hidden_layer_2(x)
        output = self.output_layer(x)
        return output


    @tf.function
    def compute_df_du(self, u, v):
        with tf.GradientTape() as tape:
            tape.watch(u)  
            f = -2*self.beta*u + tf.exp(self.beta*u) - tf.exp(-self.beta*u) + self.kb*u - tf.sign(v)*((self.ka-self.kb)/self.alfa)*(tf.exp(-self.alfa*(tf.sign(v)*(u-self.uj)+2*self.u0))-tf.exp(-2*self.alfa*self.u0)) + tf.sign(v)*self.f0

        df_du = tape.gradient(f, u)
        df_du = tf.cast(df_du, dtype=tf.float32)

        return df_du
    
    def train_step(self):
        with tf.GradientTape() as tape:
            y_pred = self.model_pinn(self.x_tr, training=True)
            loss_value = self.custom_loss(self.y_tr, y_pred)
        gradients = tape.gradient(loss_value, self.model_pinn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_pinn.trainable_variables))
        return loss_value
    

    def pinn_fit(self, epochs=500):
        self.model_pinn = self.PINN(input_dim=2, output_dim=1)
        for epoch in range(epochs):
            loss_value = self.train_step()
            if epoch % 100 == 0:  # print loss every 100 epochs
                print(f'Epoch {epoch}, Loss: {loss_value}')
        return self.model_pinn



class PINN_BoucWen(tf.keras.Model):
    def __init__(self, output_dim):
        super().__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.output_layer_z = tf.keras.layers.Dense(units=1)  # To predict z
        self.output_layer_f = tf.keras.layers.Dense(units=output_dim)  # To predict F(t)

    def call(self, inputs):
        x = self.hidden_layer_1(inputs)
        x = self.hidden_layer_2(x)
        z = self.output_layer_z(x)
        f = self.output_layer_f(x)
        return z, f  # Return both z and F(t)

def compute_boucwen_dynamics(u, z, A, B, G, n):
    # No need to compute gradients here as TensorFlow can automatically compute them during training
    # z = tf.clip_by_value(z, -1e2, 1e2)
    Z_dot = A*u - B*(tf.abs(u)*tf.abs(z)**(n-1)*z) - G*u*tf.abs(z)**n
    return Z_dot

def boucwen_loss(y_true, y_pred_f, u, z_pred, A, B, G, n):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true = tf.expand_dims(y_true, axis=-1)

    Z_dot_pred = compute_boucwen_dynamics(u, z_pred, A, B, G, n)
    
    # Compute physics_loss and data_loss
    physics_loss = tf.reduce_mean(tf.square(Z_dot_pred - z_pred))  # Z_dot_pred should be equal to z_pred according to the Bouc-Wen model
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred_f))  # Predicted F(t) should be close to the true values

    return 0.5 * physics_loss + 0.5 * data_loss  

def model_pinn_boucwen(x, training):
    return model_pinn_boucwen(x, training=training)


def train_step_boucwen(x_tr, y_tr, A, B, G, n):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    u = x_tr[:, 0]
    u = tf.convert_to_tensor(u)

    with tf.GradientTape() as tape:
        z_pred, y_pred_f = model_pinn_boucwen(x_tr, training=True)
        loss_value = boucwen_loss(y_tr, y_pred_f, u, z_pred, A, B, G, n)
    gradients = tape.gradient(loss_value, model_pinn_boucwen.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_pinn_boucwen.trainable_variables))
    return loss_value






def make_data(dt=0.01, total_time = 10):
    # APPLIED DISPLACEMENT TIME HISTORY
    dt = dt
    t = np.arange(0, total_time + dt, dt)
    a0 = 1
    fr = 1
    u = a0 * np.sin(2 * np.pi * fr * t[:len(t)])
    v = 2 * np.pi * fr * a0 * np.cos(2 * np.pi * fr * t[:len(t)])
    n = len(u)
    uj_result = []

    # INITIAL SETTINGS
    # Set the four model parameters
    ka = 5.0
    kb = 0.5
    alfa = 5.0
    beta = 1.0
    # Compute the internal model parameters
    u0 = -(1 / (2 * alfa)) * np.log(10 ** -20 / (ka - kb))
    f0 = ((ka - kb) / (2 * alfa)) * (1 - np.exp(-2 * alfa * u0))
    # Initialize the generalized force vector
    f = np.zeros(n)

    count_reuslt = []
    # CALCULATIONS AT EACH TIME STEP
    for i in range(1, n):
        # Update the history variable
        uj = u[i-1] + 2*u0*np.sign(v[i]) + np.sign(v[i])*(1/alfa)*np.log(np.abs(np.sign(v[i])*(alfa/(ka-kb))*(-2*beta*u[i-1]+np.exp(beta*u[i-1])-np.exp(-beta*u[i-1])+kb*u[i-1]+np.sign(v[i])*((ka-kb)/alfa)*np.exp(-2*alfa*u0)+np.sign(v[i])*f0-f[i-1])))
        # Evaluate the generalized force at time t
        uj_result.append(uj)

        if (np.sign(v[i])*uj-2*u0 < np.sign(v[i])*u[i]) or (np.sign(v[i])*u[i] < np.sign(v[i])*uj):
            count_reuslt.append(i)
            f[i] = -2*beta*u[i] + np.exp(beta*u[i]) - np.exp(-beta*u[i]) + kb*u[i] - np.sign(v[i])*((ka-kb)/alfa)*(np.exp(-alfa*(np.sign(v[i])*(u[i]-uj)+2*u0))-np.exp(-2*alfa*u0)) + np.sign(v[i])*f0
        else:
            f[i] = -2*beta*u[i] + np.exp(beta*u[i]) - np.exp(-beta*u[i]) + kb*u[i] + np.sign(v[i])*f0

    total = pd.DataFrame()  
    total['u'] = u
    total['time'] = t
    total['f'] = f

    return total