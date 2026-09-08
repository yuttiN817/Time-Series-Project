#timegan_utils.py

#importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import acf



#Setting a seed for a reproducibility purpose
def set_seed(seed = 0):
    np.random.seed(seed)
    tf.random.set_seed(seed)


#Allowing to choose the scaler type, either standard (mean 0, std 1) or minmax scaler (range 0 - 1)
def get_scaler(scalerType="standard"):
    if scalerType == "standard":
        return StandardScaler()
    elif scalerType == "minmax":
        return MinMaxScaler()
    #Allow either of those
    else:
        raise ValueError("scalerType must be either 'standard' or 'minmax'.")

#Creating the overlapping windows.
#If original data = [10, 11, 13, 12, 15, 16, 14] and seqLen = 3
# -> [10, 11, 13], [11, 13, 12], [13, 12, 15], [12, 15, 16], [15, 16, 14]
def create_windows(series, seqLen, scalerType = "standard"):
    series = np.asarray(series)

    #Converting data to two-dimentional. Ex.) Shape: (5,) -> (5, 1)
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    scaler = get_scaler(scalerType)
    scaledSeries = scaler.fit_transform(series)

    windows = []

    #Overlapping sequence creation
    for i in range(len(scaledSeries) - seqLen + 1):
        window = scaledSeries[i:i + seqLen]
        windows.append(window)

    return np.array(windows), scaler

#Converting generating windows back to the original scale 
def inverse_transform_series(series, scaler):
    series = np.asarray(series)

    #Removing the batch dimension
    #shape: (1, 2000, 1) -> (2000, 1)
    if series.ndim == 3:
        series = series[0]

    syntheticSeries = scaler.inverse_transform(series)

    if syntheticSeries.shape[1] == 1:
        syntheticSeries = syntheticSeries.flatten()

    return syntheticSeries
    

#Random noise as input for the TimeGAN generator
#batchSize = number of sequence to pass on to generator at a time
#Shape: (number of sequences, sequnce length, noise dimension)
#If (128, 100, 5) -> 128 numbers of sequence of data with 100(time points) by 5(zDim) matrix
def random_noise(batchSize, seqLen, zDim, noiseType = "normal"):
    if noiseType == "normal":
        #Converting to float32 (for tensorflow/Keras), as the default is float64
        return np.random.normal(0, 1, size = [batchSize, seqLen, zDim]).astype(np.float32)

    elif noiseType == "uniform":
        return np.random.uniform(-1, 1, size = [batchSize, seqLen, zDim]).astype(np.float32)

    else:
        raise ValueError("noiseType must be 'normal', 'uniform'.")


#Inheritance from tf.keras.Model
#Embedder is a network which transform the original data into a latent representation
class Embedder(tf.keras.Model):
    #hiddenDim = number of hidden units in GRU
    def __init__(self, hiddenDim, numLayers, latentActivation = "tanh"):
        super().__init__()

        #Number of GRU = numLayers
        self.rnnLayers = []

        #For example, input shape: (128, 100, 1) (batchSize, seqLen, featureDim)
        #This becomes (128, 100, 12) through GRU if hiddenDim = 12. Each time point is represented by 12 hidden values
        for _ in range(numLayers):
            #return_sequence = True allows returning output for all time point
            rnn = layers.GRU(hiddenDim, return_sequences = True)
            self.rnnLayers.append(rnn)

        #Create final latent representation through the dense layer.
        self.dense = layers.Dense(hiddenDim, activation = latentActivation)

    def call(self, x):
        #x is the input data
        h = x

        #h is going through GRU(s)
        for rnn in self.rnnLayers:
            h = rnn(h)

        return self.dense(h)

#Recovery is a network which transform the latent representation back to the original data
class Recovery(tf.keras.Model):
    #hiddenDim = number of hidden units in GRU
    #featureDim = number of features in the original data
    #None allows the output without any range restriction
    def __init__(self, hiddenDim, numLayers, featureDim, recoveryActivation = None):
        super().__init__()

        self.rnnLayers = []

        #For example, input shape: (128, 100, 12) (batchSize, seqLen, hiddenDim)
        #This remains (128, 100, 12) through GRU if hiddenDim = 12
        for _ in range(numLayers):
            rnn = layers.GRU(hiddenDim, return_sequences = True)
            self.rnnLayers.append(rnn)

        #Transform the latent representation back to the original feature dimension through the dense layer
        #For example, (128, 100, 12) becomes (128, 100, 1) if featureDim = 1
        self.dense = layers.Dense(featureDim, activation = recoveryActivation)

    def call(self, h):
        #h is the latent representation
        x_tilde = h

        #x_tilde is going through GRU(s)
        for rnn in self.rnnLayers:
            x_tilde = rnn(x_tilde)

        return self.dense(x_tilde)

#Generator is a network which transform random noise into a synthetic latent representation
class Generator(tf.keras.Model):
    def __init__(self, hiddenDim, numLayers, latentActivation="tanh"):
        super().__init__()

        self.rnnLayers = []

        #For example, input shape: (128, 100, 5) (batchSize, seqLen, zDim)
        #This becomes (128, 100, 12) through GRU if hiddenDim = 12. Each time point is represented by 12 hidden values
        for _ in range(numLayers):
            rnn = layers.GRU(hiddenDim, return_sequences = True)
            self.rnnLayers.append(rnn)

        #Create final synthetic latent representation through the dense layer
        self.dense = layers.Dense(hiddenDim, activation = latentActivation)

    def call(self, z):
        #z is the random noise input
        e_hat = z

        #e_hat is going through GRU(s)
        for rnn in self.rnnLayers:
            e_hat = rnn(e_hat)

        return self.dense(e_hat)

#Supervisor is a network which learns the temporal relationships in the latent representation
class Supervisor(tf.keras.Model):
    def __init__(self, hiddenDim, numLayers, latentActivation="tanh"):
        super().__init__()

        #Number of GRU = numLayers - 1
        #Even if this becomes zero, this network still uses a dense layer - and
        #temporal information is contained in the latent representation h, created by GRU (worked fine for AR(1))
        supervisorLayers = max(numLayers - 1, 0)

        self.rnnLayers = []

        #For example, input shape: (128, 100, 12) (batchSize, seqLen, hiddenDim)
        #This remains (128, 100, 12) through GRU if hiddenDim = 12
        for _ in range(supervisorLayers):
            rnn = layers.GRU(hiddenDim, return_sequences = True)
            self.rnnLayers.append(rnn)

        self.dense = layers.Dense(hiddenDim, activation = latentActivation)

    def call(self, h):
        #h is the latent representation
        s = h

        #s is going through GRU(s)
        for rnn in self.rnnLayers:
            s = rnn(s)

        #Returning the supervised latent representation
        return self.dense(s)

#Discriminator is a network which determines whether the latent representation is real or synthetic
class Discriminator(tf.keras.Model):
    def __init__(self, hiddenDim, numLayers):
        super().__init__()

        self.rnnLayers = []

        #input shape: (128, 100, 12) (batchSize, seqLen, hiddenDim)
        for _ in range(numLayers):
            rnn = layers.GRU(hiddenDim, return_sequences = True)
            self.rnnLayers.append(rnn)

        #Create a real or synthetic score for each time point through the dense layer
        #Final shape: (128, 100, 1)
        #No activation (BinaryCrossentropy uses from_logits = True)
        self.dense = layers.Dense(1)

    def call(self, h):
        y = h

        for rnn in self.rnnLayers:
            y = rnn(y)

        #Returning the real or synthetic
        return self.dense(y)


def build_timegan_models(hiddenDim = 12, numLayers = 1, featureDim = 1, latentActivation = "tanh", recoveryActivation = None):

    embedder = Embedder(hiddenDim = hiddenDim, numLayers = numLayers, latentActivation = latentActivation)

    recovery = Recovery(hiddenDim = hiddenDim, numLayers = numLayers, featureDim = featureDim, recoveryActivation = recoveryActivation)

    generator = Generator(hiddenDim = hiddenDim, numLayers = numLayers, latentActivation = latentActivation)

    supervisor = Supervisor(hiddenDim = hiddenDim, numLayers = numLayers, latentActivation = latentActivation)

    discriminator = Discriminator(hiddenDim = hiddenDim, numLayers = numLayers)

    models = {
        "embedder": embedder,
        "recovery": recovery,
        "generator": generator,
        "supervisor": supervisor,
        "discriminator": discriminator
    }

    return models



#TimeGAN Training
def train_timegan(data, seqLen = 50, hiddenDim = 12, numLayers = 1, featureDim = None, zDim = 5, batchSize = 128, pretrainIterations = 1000,
    supervisorIterations = 1000, jointIterations = 2000, gamma = 1, learningRate = 0.001, supervisedWeight = 10, momentWeight = 1,
    reconstructionWeight = 10, embedderSupervisedWeight = 0.1, latentActivation = "tanh", recoveryActivation = None, noiseType = "normal",
    seed = None, clearSession = False, printEvery = 100):

    if seed is not None:
        set_seed(seed)

    #Memory administration for multiple models
    if clearSession:
        tf.keras.backend.clear_session()

    #Good to use in tensorflow
    data = np.asarray(data).astype(np.float32)

    if data.ndim != 3:
        raise ValueError("data must have shape: number of windows, seqLen, featureDim.")

    if featureDim is None:
        featureDim = data.shape[2]

    #Converting data (numpy array) into tensorflow dataset↓
    #Randomly reordering the training windows for training (STL decomposition for this purpose)↓
    #Dividing the window into batched and dropping the imcomplete final batch (remainder)↓
    #Reapeat the dataset so that it can take the training data over and over again
    dataset = (tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size = len(data)).batch(batchSize, drop_remainder = True).repeat())

    #Creating an iterator to get one batch at a time
    dataIterator = iter(dataset)

    models = build_timegan_models(hiddenDim = hiddenDim, numLayers = numLayers, featureDim = featureDim,
        latentActivation = latentActivation, recoveryActivation = recoveryActivation)

    embedder = models["embedder"]
    recovery = models["recovery"]
    generator = models["generator"]
    supervisor = models["supervisor"]
    discriminator = models["discriminator"]

    #Mean Squared Error for reconstruction / supervised losses
    mse = tf.keras.losses.MeanSquaredError()
    #Binary Cross Entropy loss (for Discriminator)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    #Learning rate is 0.001(fixed using a common setting)
    embedderOptimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)
    supervisorOptimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)
    generatorOptimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)
    discriminatorOptimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)
    #A separate optimizer for Embedder and Recovery during joint training
    embedderJointOptimizer = tf.keras.optimizers.Adam(learning_rate = learningRate)

    #X -> Embedder -> H -> Recovery -> X_tilde for one batch
    @tf.function
    def train_embedder_step(X):
        #GradientTape records the calculuation 
        with tf.GradientTape() as tape:
            H = embedder(X)
            X_tilde = recovery(H)

            reconstructionLoss = mse(X, X_tilde)

        variables = embedder.trainable_variables + recovery.trainable_variables
        #Calculating gradient and considering the change of weights to reduce the reconstruction loss
        gradients = tape.gradient(reconstructionLoss, variables)
        #Updating the weights
        embedderOptimizer.apply_gradients(zip(gradients, variables))

        return reconstructionLoss

    @tf.function
    def train_supervisor_step(X):
        with tf.GradientTape() as tape:
            #stop_gradient - no update on the weights on Embedder, we want to pretrain only supervisor
            #X -> H
            H = tf.stop_gradient(embedder(X))
            #Predicting the next latent representation
            H_hat_supervise = supervisor(H)

            #H[:, 1:, :] -> h2, h3...h100 for all batches and hidden values. -1: removing the last time point
            #Getting corresponding output to predicted h2, h3, ... h100
            #Supervisor output from h1 vs actual h2...Supervisor output from h99 vs actual h100
            supervisorLoss = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

        variables = supervisor.trainable_variables
        gradients = tape.gradient(supervisorLoss, variables)
        #Updating supervisor weights using Adam optimizer
        supervisorOptimizer.apply_gradients(zip(gradients, variables))

        return supervisorLoss

    @tf.function
    #X is the real training data, Z is a random noise
    def train_generator_step(X, Z):
        with tf.GradientTape() as tape:
            #X -> H (Latent representation)
            H = tf.stop_gradient(embedder(X))

            #Z -> E_hat (synthetic latent representation)
            E_hat = generator(Z)
            #Reflects temporal relationships (real / synthetic)
            H_hat = supervisor(E_hat)
            H_hat_supervise = supervisor(H)

            #Convert to original data space
            X_hat = recovery(H_hat)

            #Showing synthetic latent to the Discriminator (before and after going through supervisor)
            Y_fake = discriminator(H_hat)
            Y_fake_e = discriminator(E_hat)

            #Adversarial losses
            G_loss_U = bce(tf.ones_like(Y_fake), Y_fake)
            G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)

            #Supervisor prediction at t vs actual H at t+1
            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

            #Loss of the standard deviation difference between synthetic and real data
            G_loss_V1 = tf.reduce_mean(
                tf.abs(tf.sqrt(tf.math.reduce_variance(X_hat, axis=0) + 1e-6) - tf.sqrt(tf.math.reduce_variance(X, axis=0) + 1e-6))
            )

            #Loss of the mean difference between synthetic and real data
            G_loss_V2 = tf.reduce_mean(
                tf.abs(tf.reduce_mean(X_hat, axis=0) - tf.reduce_mean(X, axis=0))
            )

            G_loss_V = G_loss_V1 + G_loss_V2

            #Final generator loss
            G_loss = (
                G_loss_U + gamma * G_loss_U_e + supervisedWeight * tf.sqrt(G_loss_S + 1e-6) + momentWeight * G_loss_V
            )

        variables = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(G_loss, variables)
        generatorOptimizer.apply_gradients(zip(gradients, variables))

        return G_loss

    @tf.function
    def train_embedder_joint_step(X):
        with tf.GradientTape() as tape:
            H = embedder(X)
            X_tilde = recovery(H)

            H_hat_supervise = supervisor(H)

            E_loss_T0 = mse(X, X_tilde)

            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

            #Total Embedder loss during joint training
            E_loss = (reconstructionWeight * tf.sqrt(E_loss_T0 + 1e-6) + embedderSupervisedWeight * G_loss_S)

        variables = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(E_loss, variables)
        embedderJointOptimizer.apply_gradients(zip(gradients, variables))

        return E_loss

    @tf.function
    def train_discriminator_step(X, Z):
        with tf.GradientTape() as tape:
            H = tf.stop_gradient(embedder(X))

            E_hat = tf.stop_gradient(generator(Z))
            H_hat = tf.stop_gradient(supervisor(E_hat))

            #Discriminator raw scores for real and synthetic latent representations
            Y_real = discriminator(H)
            Y_fake = discriminator(H_hat)
            Y_fake_e = discriminator(E_hat)

            #Real latent representation should be classified as real
            D_loss_real = bce(tf.ones_like(Y_real), Y_real)

            #Synthetic latent representatoins should be classified as fake
            D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake)
            D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e)

            #Total Discriminator loss
            D_loss = (D_loss_real + D_loss_fake + gamma * D_loss_fake_e)

        variables = discriminator.trainable_variables
        gradients = tape.gradient(D_loss, variables)
        discriminatorOptimizer.apply_gradients(zip(gradients, variables))

        return D_loss

    embedderLosses = []

    for iteration in range(pretrainIterations):
        #Taking the next one batch from the iterator
        X = next(dataIterator)

        #Training Embedder and Recovery for one step
        loss = train_embedder_step(X)
        embedderLosses.append(loss.numpy())

        if iteration % printEvery == 0:
            print(f"Embedder iteration {iteration}, loss: {loss.numpy():.6f}")

    supervisorLosses = []

    for iteration in range(supervisorIterations):
        X = next(dataIterator)

        #Training supervisor for one step
        loss = train_supervisor_step(X)
        supervisorLosses.append(loss.numpy())

        if iteration % printEvery == 0:
            print(f"Supervisor iteration {iteration}, loss: {loss.numpy():.6f}")

    generatorLosses = []
    embedderJointLosses = []
    discriminatorLosses = []

    for iteration in range(jointIterations):
        X = next(dataIterator)

        #This is using a function defined above to create random noises for generator
        Z = random_noise(batchSize = batchSize, seqLen = seqLen, zDim = zDim, noiseType = noiseType)

        #Joint training (Generator + Supervisor)
        gLoss = train_generator_step(X, Z)

        #Embedder + Recovery
        eLoss = train_embedder_joint_step(X)

        Z = random_noise(batchSize = batchSize, seqLen = seqLen, zDim = zDim, noiseType = noiseType)

        #Discriminator
        dLoss = train_discriminator_step(X, Z)

        generatorLosses.append(gLoss.numpy())
        embedderJointLosses.append(eLoss.numpy())
        discriminatorLosses.append(dLoss.numpy())

        if iteration % printEvery == 0:
            print(
                f"Joint iteration {iteration}, "
                f"G loss: {gLoss.numpy():.4f}, "
                f"E loss: {eLoss.numpy():.4f}, "
                f"D loss: {dLoss.numpy():.4f}"
            )

    losses = {
        "embedder": embedderLosses,
        "supervisor": supervisorLosses,
        "generator": generatorLosses,
        "embedder_joint": embedderJointLosses,
        "discriminator": discriminatorLosses
    }

    return models, losses

#name represents the series name, such as 'phi = 0.3' in AR(1)
def train_timegan_for_series(name, TimeGANdata, TimeGANscalers = None, outputlength = None, generateExample = True, **kwargs):
    #TimeGANdata is a dictionary - one series can be taken from it
    #Getting training windows for the selected time series
    data = TimeGANdata[name].astype(np.float32)

    #Obtaining the generation settings
    seqLen = kwargs.get("seqLen", data.shape[1])
    zDim = kwargs.get("zDim", 5)
    noiseType = kwargs.get("noiseType", "normal")

    #Setting up the seqLen in kwargs just in case it is not specified
    kwargs["seqLen"] = seqLen

    models, losses = train_timegan(data = data, **kwargs)

    syntheticSeries = None

    #Getting one synthetic series after training
    if generateExample:
        if TimeGANscalers is None:
            raise ValueError("TimeGANscalers must be provided when generateExample = True.")

        #If the length of output is not specified, it automatically computes the original time series length
        if outputlength is None:
            outputlength = data.shape[0] + seqLen - 1

        syntheticSeries = generate_timegan_series(models = models, scaler = TimeGANscalers[name],
            outputlength = outputlength, zDim = zDim, noiseType = noiseType
        )

    return models, losses, syntheticSeries

def generate_timegan_series(models, scaler, outputlength, zDim = 5, noiseType = "normal"):
    generator = models["generator"]
    supervisor = models["supervisor"]
    recovery = models["recovery"]

    #Generating one continuours random noise sequence
    #For example, outputlength = 200 and zDim = 5 -> shape: (1, 2000, 5)
    Z = random_noise(batchSize = 1, seqLen = outputlength, zDim = zDim, noiseType = noiseType)

    #Synthetic latent representation, supervisor(trained temporal relationships), recovery
    E_hat = generator(Z)
    H_hat = supervisor(E_hat)
    X_hat = recovery(H_hat).numpy()

    #Transforming the generated series back to the original scale
    syntheticSeries = inverse_transform_series(series = X_hat, scaler = scaler)

    return syntheticSeries


def generate_timegan_samples(models, scaler, B, outputlength, zDim = 5,
    noiseType="normal"):
    samples = []

    #Create the specified number of samples using the trained model
    for b in range(B):
        sample = generate_timegan_series(models = models, scaler = scaler, outputlength = outputlength,
            zDim = zDim, noiseType = noiseType)

        samples.append(sample)

    return np.array(samples)


#Computing the acf values and errors, but these are also defined in the notebooks
def acf_values(series, maxLag = 50):
    series = np.asarray(series)

    if series.ndim == 2:
        series = series[:, 0]

    return acf(series, nlags=maxLag, fft=True)


def mean_error(original, synthetic):
    return abs(np.mean(original) - np.mean(synthetic))


def variance_error(original, synthetic):
    return abs(np.var(original) - np.var(synthetic))


def acf_error(original, synthetic, maxLag = 50):
    originalACF = acf_values(original, maxLag=maxLag)
    syntheticACF = acf_values(synthetic, maxLag=maxLag)

    return np.sum((originalACF[1:] - syntheticACF[1:]) ** 2)


def timegan_diagnostics(original, synthetic, maxLag = 50):
    originalACF = acf_values(original, maxLag = maxLag)
    syntheticACF = acf_values(synthetic, maxLag = maxLag)

    #Recording results
    diagnostics = {
        "original_mean": np.mean(original),
        "synthetic_mean": np.mean(synthetic),
        "original_std": np.std(original),
        "synthetic_std": np.std(synthetic),
        "original_variance": np.var(original),
        "synthetic_variance": np.var(synthetic),
        "original_min": np.min(original),
        "original_max": np.max(original),
        "synthetic_min": np.min(synthetic),
        "synthetic_max": np.max(synthetic),
        "original_lag1_acf": originalACF[1],
        "synthetic_lag1_acf": syntheticACF[1],
        "acf_error": acf_error(original, synthetic, maxLag=maxLag)
    }

    #Printing out the results
    print("Original mean:", diagnostics["original_mean"])
    print("Synthetic mean:", diagnostics["synthetic_mean"])
    print()
    print("Original std:", diagnostics["original_std"])
    print("Synthetic std:", diagnostics["synthetic_std"])
    print()
    print("Original variance:", diagnostics["original_variance"])
    print("Synthetic variance:", diagnostics["synthetic_variance"])
    print()
    print(
        "Original min/max:",
        diagnostics["original_min"],
        diagnostics["original_max"]
    )
    print(
        "Synthetic min/max:",
        diagnostics["synthetic_min"],
        diagnostics["synthetic_max"]
    )
    print()
    print("Original lag 1 ACF:", diagnostics["original_lag1_acf"])
    print("Synthetic lag 1 ACF:", diagnostics["synthetic_lag1_acf"])
    print("ACF error:", diagnostics["acf_error"])

    return diagnostics
