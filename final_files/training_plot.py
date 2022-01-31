import numpy as np
import matplotlib.pyplot as plt
# load the training progress
early_CC = np.load('working_npy/regularize_early_CC_training.npy', allow_pickle='True').item()
early_D = np.load('working_npy/regularize_early_D_training.npy', allow_pickle='True').item()
early_Y = np.load('working_npy/regularize_early_Y_training.npy', allow_pickle='True').item()

dropout_CC = np.load('working_npy/regularize_dropout_CC_training.npy', allow_pickle='True').item()
dropout_D = np.load('working_npy/regularize_dropout_D_training.npy', allow_pickle='True').item()
dropout_Y = np.load('working_npy/regularize_dropout_Y_training.npy', allow_pickle='True').item()

wd1_CC = np.load('working_npy/regularize_wd1_CC_training.npy', allow_pickle='True').item()
wd1_D = np.load('working_npy/regularize_wd1_D_training.npy', allow_pickle='True').item()
wd1_Y = np.load('working_npy/regularize_wd1_Y_training.npy', allow_pickle='True').item()

wd2_CC = np.load('working_npy/regularize_wd2_CC_training.npy', allow_pickle='True').item()
wd2_D = np.load('working_npy/regularize_wd2_D_training.npy', allow_pickle='True').item()
wd2_Y = np.load('working_npy/regularize_wd2_Y_training.npy', allow_pickle='True').item()

plain_CC = np.load('working_npy/plain_CC_training.npy', allow_pickle='True').item()
plain_D = np.load('working_npy/plain_D_training.npy', allow_pickle='True').item()
plain_Y = np.load('working_npy/plain_D_training.npy', allow_pickle='True').item()

da_CC = np.load('data_augmentation_CC_training.npy', allow_pickle='True').item()
da_D = np.load('data_augmentation_D_training.npy', allow_pickle='True').item()
da_Y = np.load('data_augmentation_Y_training.npy', allow_pickle='True').item()

def multiplot(plain, res1, res2, name1, name2, variable_name, filename):
    p_loss_values = plain['loss']
    p_val_loss_values = plain['val_loss']
    p_train_acc = plain['accuracy']
    p_val_acc = plain['val_accuracy']
    epochs_p = range(1, len(p_loss_values) + 1)

    res1_loss_values = res1['loss']
    res1_val_loss_values = res1['val_loss']
    res1_train_acc = res1['accuracy']
    res1_val_acc = res1['val_accuracy']
    epochs_res1 = range(1, len(res1_loss_values) + 1)

    res2_loss_values = res2['loss']
    res2_val_loss_values = res2['val_loss']
    res2_train_acc = res2['accuracy']
    res2_val_acc = res2['val_accuracy']
    epochs_res2 = range(1, len(res2_loss_values) + 1)



    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(epochs_p, p_loss_values, 'bo', label='No regularization - Training loss')
    axs[0].plot(epochs_p, p_val_loss_values, 'b', label='No regularization - Validation loss')
    axs[0].plot(epochs_res1, res1_loss_values, 'ro', label= name1 + ' - Training loss')
    axs[0].plot(epochs_res1, res1_val_loss_values, 'r', label= name1 + ' - Validation loss')
    axs[0].plot(epochs_res2, res2_loss_values, 'go', label= name2 + ' - Training loss')
    axs[0].plot(epochs_res2, res2_val_loss_values, 'g', label= name2 + ' - Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].legend()

    axs[1].plot(epochs_p, p_train_acc, 'bo', label='No regularization - Training accuracy')
    axs[1].plot(epochs_p, p_val_acc, 'b', label='No regularization - Validation accuracy')
    axs[1].plot(epochs_res1, res1_train_acc, 'ro', label=name1 + ' - Training accuracy')
    axs[1].plot(epochs_res1, res1_val_acc, 'r', label=name1 + ' - Validation accuracy')
    axs[1].plot(epochs_res2, res2_train_acc, 'go', label=name2 + ' - Training accuracy')
    axs[1].plot(epochs_res2, res2_val_acc, 'g', label=name2 + ' - Validation accuracy')
    axs[1].set_title('Training and validation accuracy')
    axs[1].legend()
    fig.set_size_inches(8, 5)
    plt.savefig(filename)


def multiplot2(plain, res1, name1, variable_name, filename):

    p_loss_values = plain['loss']
    p_val_loss_values = plain['val_loss']
    p_train_acc = plain['accuracy']
    p_val_acc = plain['val_accuracy']
    epochs_p = range(1, len(p_loss_values) + 1)

    res1_loss_values = res1['loss']
    res1_val_loss_values = res1['val_loss']
    res1_train_acc = res1['accuracy']
    res1_val_acc = res1['val_accuracy']
    epochs_res1 = range(1, len(res1_loss_values) + 1)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(epochs_p, p_loss_values, 'bo', label='No regularization - Training loss')
    axs[0].plot(epochs_p, p_val_loss_values, 'b', label='No regularization - Validation loss')
    axs[0].plot(epochs_res1, res1_loss_values, 'ro', label= name1 + ' - Training loss')
    axs[0].plot(epochs_res1, res1_val_loss_values, 'r', label= name1 + ' - Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].legend()

    axs[1].plot(epochs_p, p_train_acc, 'bo', label='No regularization - Training accuracy')
    axs[1].plot(epochs_p, p_val_acc, 'b', label='No regularization - Validation accuracy')
    axs[1].plot(epochs_res1, res1_train_acc, 'ro', label=name1 + ' - Training accuracy')
    axs[1].plot(epochs_res1, res1_val_acc, 'r', label=name1 + ' - Validation accuracy')
    axs[1].set_title('Training and validation accuracy')
    axs[1].legend()
    fig.set_size_inches(8, 5)
    plt.savefig(filename)
# plots for regularization

multiplot(plain_CC, early_CC, dropout_CC, 'Early Stopping', 'Dropout', 'CC', 'plots/reg_CC_es_do.jpg')
multiplot(plain_D, early_D, dropout_D, 'Early Stopping', 'Dropout', 'D', 'plots/reg_D_es_do.jpg')
multiplot(plain_Y, early_Y, dropout_Y, 'Early Stopping', 'Dropout', 'Y', 'plots/reg_Y_es_do.jpg')

multiplot(plain_CC, wd1_CC, wd2_CC, 'L2, 0.001 ', 'L2, 0.005', 'CC', 'plots/reg_CC_wd1_wd2.jpg')
multiplot(plain_D, wd1_D, wd2_D, 'L2, 0.001 ', 'L2, 0.005', 'D', 'plots/reg_D_wd1_wd2.jpg')
multiplot(plain_Y, wd1_Y, wd2_Y, 'L2, 0.001 ', 'L2, 0.005', 'Y', 'plots/reg_Y_wd1_wd2.jpg')


# plot for data augmentation
multiplot2(plain_CC, da_CC, 'Data augmentation', 'CC', 'plots/data_aug_CC.jpg')
multiplot2(plain_D, da_D, 'Data augmentation', 'D', 'plots/data_aug_D.jpg')
multiplot2(plain_Y,  da_Y, 'Data augmentation', 'Y', 'plots/data_aug_Y.jpg')
