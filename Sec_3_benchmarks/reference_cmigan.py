#############################################################################################################
# Code is a slightly modified version of the code provided by the authors of the paper:                     #
# C-MI-GAN : Estimation of Conditional Mutual Information using MinMax formulation                          #
# https://github.com/arnabkmondal/C-MI-GAN                                                                  #
#############################################################################################################

import numpy as np
import tensorflow as tf
import os

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

eps = 1e-6
SEED = None
np.random.seed(seed=SEED)


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


class TrainingDataGenerator:
    def __init__(self, X, Y, Z):
        self.xyz_vec = np.concatenate([X, Y, Z], axis=1, dtype=np.float32)
        self.n_samples = self.xyz_vec.shape[0]

    def get_batch(self, bs):
        data_indices = np.random.randint(0, self.n_samples, bs)
        return self.xyz_vec[data_indices]


def build_generator(**params):
    gen_fc_arch = params['gen_fc_arch']
    gen_model = tf.keras.models.Sequential(name='gen-seq-model')
    for layer, neurons in enumerate(gen_fc_arch):
        gen_model.add(tf.keras.layers.Dense(units=neurons,
                                            activation=tf.nn.relu,
                                            name=f'gen-dense-{layer}'))
    gen_model.add(tf.keras.layers.Dense(units=params['final_layer_neuron'],
                                        activation=None, name='gen-dense-final'))

    return gen_model


def build_discriminator(**params):
    disc_fc_arch = params['disc_fc_arch']
    disc_model = tf.keras.models.Sequential(name='disc-seq-model')
    for layer, neurons in enumerate(disc_fc_arch):
        disc_model.add(tf.keras.layers.Dense(units=neurons,
                                             activation=tf.nn.relu,
                                             name=f'disc-dense-{layer}'))
    disc_model.add(tf.keras.layers.Dense(units=1,
                                         activation=None,
                                         name='disc-dense-final'))

    return disc_model


def discriminator_loss(pred_xyz, pred_xz_gen_op):
    real_loss = -tf.reduce_mean(pred_xyz)
    fake_loss = tf.math.log(tf.reduce_mean(tf.math.exp(pred_xz_gen_op)) + eps)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(disc_output):
    return -tf.math.log(tf.reduce_mean(tf.math.exp(disc_output)) + eps)

# def TE_cmigan(var_from, var_to, epochs=5000):
#     assert isinstance(var_from, np.ndarray)
#     assert isinstance(var_to, np.ndarray)
#     assert var_from.shape[0] == var_to.shape[0]

#     x = var_from
#     y = var_to

#     Y = x[:-1]   # X-
#     Z = y[:-1]   # Y-
#     X = y[1:]    # Y0

#     batch_size = X.shape[0]
#     dx = X.shape[1]
#     dy = Y.shape[1]
#     dz = Z.shape[1]

#     params = {'gen_fc_arch': [256, 64], 'disc_fc_arch': [128, 32],
#             'batch_size': batch_size, 'final_layer_neuron': dy}

#     noise_dim = 40

#     generator = build_generator(**params)
#     discriminator = build_discriminator(**params)
#     lr = 1e-3
#     gen_opt = tf.keras.optimizers.RMSprop(lr)
#     disc_opt = tf.keras.optimizers.RMSprop(lr)


#     @tf.function
#     def gen_train_step(noise, xz, lr_decay):
#         with tf.GradientTape() as gen_tape:
#             gen_op = generator(noise)
#             x_gen_op_z = tf.concat([tf.concat([xz[:, 0:dx], gen_op], axis=1), xz[:, dx:]], axis=1)
#             disc_output_for_x_gen_op_z = discriminator(x_gen_op_z)
#             curr_gen_loss = lr_decay * generator_loss(disc_output_for_x_gen_op_z)
#         gradients_of_gen = gen_tape.gradient(curr_gen_loss, generator.trainable_variables)
#         gen_opt.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))

#         return curr_gen_loss / lr_decay


#     @tf.function
#     def disc_train_step(noise, xyz, xz, lr_decay):
#         with tf.GradientTape() as disc_tape:
#             disc_output_for_xyz = discriminator(xyz)
#             gen_op = generator(noise)
#             x_gen_op_z = tf.concat([tf.concat([xz[:, 0:dx], gen_op], axis=1), xz[:, dx:]], axis=1)
#             disc_output_for_x_gen_op_z = discriminator(x_gen_op_z)
#             curr_disc_loss = lr_decay * discriminator_loss(disc_output_for_xyz, disc_output_for_x_gen_op_z)
#         gradients_of_disc = disc_tape.gradient(curr_disc_loss, discriminator.trainable_variables)
#         disc_opt.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

#         return curr_disc_loss / lr_decay


#     # Training loop

#     data_generator = TrainingDataGenerator(X, Y, Z)

#     plot_interval = 100
#     training_steps = epochs
#     est_cmi_buf = []
#     steps_buf = []
#     gen_lr_decay = 1
#     disc_lr_decay = 1
#     disc_training_ratio = 2
#     gen_training_ratio = 1
#     factor = 1.1
#     for step in range(training_steps):
        
#         for _ in range(disc_training_ratio):
#             xyz_batch = data_generator.get_batch(batch_size)
#             xz_batch = np.delete(xyz_batch, np.arange(dx, dx + dy), 1)
#             z_batch = np.delete(xyz_batch, np.arange(0, dx + dy), 1)
#             noise_vec = np.random.normal(0., 1., [batch_size, noise_dim]).astype(np.float32)
#             noise_z = np.concatenate((noise_vec, z_batch), axis=1)
            
#             disc_loss = disc_train_step(noise_z, xyz_batch, xz_batch, disc_lr_decay)
        
#         for _ in range(gen_training_ratio):
#             gen_loss = gen_train_step(noise_z, xz_batch, gen_lr_decay)
        
#         if step > 0 and step % 1000 == 0:
#             gen_lr_decay = gen_lr_decay / (factor)
#             disc_lr_decay = disc_lr_decay / (factor)
#         if step % (plot_interval / 10) == 0:
#             est_cmi_buf.append(-disc_loss.numpy())
#             steps_buf.append(step)
#         if step > 55:
#             est_cmi = np.round(float(np.mean(est_cmi_buf[-5:])),4)
#             print(
#                 f"Current Iteration: {step}, Estimated CMI: {est_cmi}", end='\r')
#         if step > 205:
#             # check if the estimated CMI is decreasing, if so, break the loop
#             if float(np.mean(est_cmi_buf[-10:])) < float(np.mean(est_cmi_buf[-20:-10])):
#                 break
        
#     print()
#     return est_cmi

def TE_cmigan(var_from, var_to, epochs=5000, batch_size=1000):
    assert isinstance(var_from, np.ndarray)
    assert isinstance(var_to, np.ndarray)
    assert var_from.shape[0] == var_to.shape[0]

    x = var_from
    y = var_to

    X = y[1:]    # Y0
    Y = x[:-1]   # X-
    Z = y[:-1]   # Y-
    
    dx = X.shape[1]
    dy = Y.shape[1]
    dz = Z.shape[1]

    params = {'gen_fc_arch': [256, 64], 'disc_fc_arch': [128, 32],
            'batch_size': batch_size, 'final_layer_neuron': dy}

    noise_dim = 40

    generator = build_generator(**params)
    discriminator = build_discriminator(**params)
    lr = 1e-3
    gen_opt = tf.keras.optimizers.RMSprop(lr)
    disc_opt = tf.keras.optimizers.RMSprop(lr)


    @tf.function
    def gen_train_step(noise, xz, lr_decay):
        with tf.GradientTape() as gen_tape:
            gen_op = generator(noise)
            x_gen_op_z = tf.concat([tf.concat([xz[:, 0:dx], gen_op], axis=1), xz[:, dx:]], axis=1)
            disc_output_for_x_gen_op_z = discriminator(x_gen_op_z)
            curr_gen_loss = lr_decay * generator_loss(disc_output_for_x_gen_op_z)
        gradients_of_gen = gen_tape.gradient(curr_gen_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))

        return curr_gen_loss / lr_decay


    @tf.function
    def disc_train_step(noise, xyz, xz, lr_decay):
        with tf.GradientTape() as disc_tape:
            disc_output_for_xyz = discriminator(xyz)
            gen_op = generator(noise)
            x_gen_op_z = tf.concat([tf.concat([xz[:, 0:dx], gen_op], axis=1), xz[:, dx:]], axis=1)
            disc_output_for_x_gen_op_z = discriminator(x_gen_op_z)
            curr_disc_loss = lr_decay * discriminator_loss(disc_output_for_xyz, disc_output_for_x_gen_op_z)
        gradients_of_disc = disc_tape.gradient(curr_disc_loss, discriminator.trainable_variables)
        disc_opt.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

        return curr_disc_loss / lr_decay


    # Training loop

    data_generator = TrainingDataGenerator(X, Y, Z)

    est_cmi_buf = []
    gen_lr_decay = 1
    disc_lr_decay = 1
    disc_training_ratio = 2
    gen_training_ratio = 1
    factor = 1.1
    for step in range(epochs):
        # Training the discriminator
        for _ in range(disc_training_ratio):
            xyz_batch = data_generator.get_batch(batch_size)
            xz_batch = np.delete(xyz_batch, np.arange(dx, dx + dy), 1)
            z_batch = np.delete(xyz_batch, np.arange(0, dx + dy), 1)
            noise_vec = np.random.normal(0., 1., [batch_size, noise_dim]).astype(np.float32)
            noise_z = np.concatenate((noise_vec, z_batch), axis=1)
            
            disc_loss = disc_train_step(noise_z, xyz_batch, xz_batch, disc_lr_decay)
        # Training the generator
        for _ in range(gen_training_ratio):
            gen_loss = gen_train_step(noise_z, xz_batch, gen_lr_decay)
        # Learning rate decay
        if step > 0 and step % 1000 == 0:
            gen_lr_decay = gen_lr_decay / (factor)
            disc_lr_decay = disc_lr_decay / (factor)
        
        if step > 50:
            est_cmi_buf.append(-disc_loss.numpy())
        if step > 60:
            est_cmi = np.round(float(np.mean(est_cmi_buf[-25:])),4)
            print(
                f"Current Iteration: {step+1}, Estimated CMI: {est_cmi}", end='\r')
        
    print()
    return est_cmi

def TE_cmigan_batch(var_from_lst, var_to_lst, epochs=5000, batch_size=1000):

    X_past = []
    Y_past = []
    Y_future = []
    for var_from, var_to in zip(var_from_lst, var_to_lst):
        X_past.append(var_from[:-1])
        Y_past.append(var_to[:-1])
        Y_future.append(var_to[1:])

    X_past = np.concatenate(X_past, axis=0)
    Y_past = np.concatenate(Y_past, axis=0)
    Y_future = np.concatenate(Y_future, axis=0)

    X = Y_future
    Y = X_past
    Z = Y_past
    
    dx = X.shape[1]
    dy = Y.shape[1]
    dz = Z.shape[1]

    params = {'gen_fc_arch': [256, 64], 'disc_fc_arch': [128, 32],
            'batch_size': batch_size, 'final_layer_neuron': dy}

    noise_dim = 40

    generator = build_generator(**params)
    discriminator = build_discriminator(**params)
    lr = 1e-3
    gen_opt = tf.keras.optimizers.RMSprop(lr)
    disc_opt = tf.keras.optimizers.RMSprop(lr)


    @tf.function
    def gen_train_step(noise, xz, lr_decay):
        with tf.GradientTape() as gen_tape:
            gen_op = generator(noise)
            x_gen_op_z = tf.concat([tf.concat([xz[:, 0:dx], gen_op], axis=1), xz[:, dx:]], axis=1)
            disc_output_for_x_gen_op_z = discriminator(x_gen_op_z)
            curr_gen_loss = lr_decay * generator_loss(disc_output_for_x_gen_op_z)
        gradients_of_gen = gen_tape.gradient(curr_gen_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))

        return curr_gen_loss / lr_decay


    @tf.function
    def disc_train_step(noise, xyz, xz, lr_decay):
        with tf.GradientTape() as disc_tape:
            disc_output_for_xyz = discriminator(xyz)
            gen_op = generator(noise)
            x_gen_op_z = tf.concat([tf.concat([xz[:, 0:dx], gen_op], axis=1), xz[:, dx:]], axis=1)
            disc_output_for_x_gen_op_z = discriminator(x_gen_op_z)
            curr_disc_loss = lr_decay * discriminator_loss(disc_output_for_xyz, disc_output_for_x_gen_op_z)
        gradients_of_disc = disc_tape.gradient(curr_disc_loss, discriminator.trainable_variables)
        disc_opt.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

        return curr_disc_loss / lr_decay


    # Training loop

    data_generator = TrainingDataGenerator(X, Y, Z)

    est_cmi_buf = []
    gen_lr_decay = 1
    disc_lr_decay = 1
    disc_training_ratio = 2
    gen_training_ratio = 1
    factor = 1.1
    for step in range(epochs):
        # Training the discriminator
        for _ in range(disc_training_ratio):
            xyz_batch = data_generator.get_batch(batch_size)
            xz_batch = np.delete(xyz_batch, np.arange(dx, dx + dy), 1)
            z_batch = np.delete(xyz_batch, np.arange(0, dx + dy), 1)
            noise_vec = np.random.normal(0., 1., [batch_size, noise_dim]).astype(np.float32)
            noise_z = np.concatenate((noise_vec, z_batch), axis=1)
            
            disc_loss = disc_train_step(noise_z, xyz_batch, xz_batch, disc_lr_decay)
        # Training the generator
        for _ in range(gen_training_ratio):
            gen_loss = gen_train_step(noise_z, xz_batch, gen_lr_decay)
        # Learning rate decay
        if step > 0 and step % 1000 == 0:
            gen_lr_decay = gen_lr_decay / (factor)
            disc_lr_decay = disc_lr_decay / (factor)
        
        if step > 50:
            est_cmi_buf.append(-disc_loss.numpy())
        if step > 60:
            est_cmi = np.round(float(np.mean(est_cmi_buf[-25:])),4)
            print(
                f"Current Iteration: {step+1}, Estimated CMI: {est_cmi}", end='\r')
        
    print()
    return est_cmi