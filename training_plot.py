import numpy as np
import matplotlib.pyplot as plt
early_CC = np.load('regularize_early_CC_training.npy', allow_pickle='True').item()
early_D = np.load('regularize_early_D_training.npy', allow_pickle='True').item()
early_Y = np.load('regularize_early_Y_training.npy', allow_pickle='True').item()

dropout_CC = np.load('regularize_dropout_CC_training.npy', allow_pickle='True').item()
dropout_D = np.load('regularize_dropout_D_training.npy', allow_pickle='True').item()
dropout_Y = np.load('regularize_dropout_Y_training.npy', allow_pickle='True').item()

wd1_CC = np.load('regularize_wd_CC_training.npy', allow_pickle='True').item()
wd1_D = np.load('regularize_wd_D_training.npy', allow_pickle='True').item()
wd1_Y = np.load('regularize_wd_Y_training.npy', allow_pickle='True').item()

wd2_CC = np.load('regularize_wd2_CC_training.npy', allow_pickle='True').item()
wd2_D = np.load('regularize_wd2_D_training.npy', allow_pickle='True').item()
wd2_Y = np.load('regularize_wd2_Y_training.npy', allow_pickle='True').item()

plain_CC = np.load('plain_CC_training.npy', allow_pickle='True').item()
plain_D = np.load('plain_D_training.npy', allow_pickle='True').item()
plain_Y = np.load('plain_D_training.npy', allow_pickle='True').item()
def plot_loss_acc(history, filename):
    history_dict = history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(epochs, loss_values, 'bo', label='Training loss')
    axs[0].plot(epochs, val_loss_values, 'b', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].legend()
    axs[1].plot(epochs, train_acc, 'ro', label='Training accuracy')
    axs[1].plot(epochs, val_acc, 'r', label='Validation accuracy')
    axs[1].set_title('Training and validation accuracy')
    axs[1].legend()
    fig.set_size_inches(8,5)
    plt.savefig('plots/' + filename)

plot_loss_acc(early_CC, 'CC_regularize_early.jpg')
plot_loss_acc(early_D, 'D_regularize_early.jpg')
plot_loss_acc(early_Y, 'Y_regularize_early.jpg')
plot_loss_acc(dropout_CC, 'CC_regularize_dropout.jpg')
plot_loss_acc(dropout_D, 'D_regularize_dropout.jpg')
plot_loss_acc(dropout_Y, 'Y_regularize_dropout.jpg')

def multiplot(plain, dropout, early, filename):
    plain_CC = plain
    dropout_CC = dropout
    early_CC = early

    p_loss_values = plain_CC['loss']
    p_val_loss_values = plain_CC['val_loss']
    p_train_acc = plain_CC['accuracy']
    p_val_acc = plain_CC['val_accuracy']

    do_loss_values = dropout_CC['loss']
    do_val_loss_values = dropout_CC['val_loss']
    do_train_acc = dropout_CC['accuracy']
    do_val_acc = dropout_CC['val_accuracy']
    epochs = range(1, len(do_loss_values) + 1)

    es_loss_values = early_CC['loss']
    es_val_loss_values = early_CC['val_loss']
    es_train_acc = early_CC['accuracy']
    es_val_acc = early_CC['val_accuracy']
    epochs_es = range(1, len(es_loss_values) + 1)



    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(epochs, p_loss_values, 'bo', label='No regularization - Training loss')
    axs[0].plot(epochs, p_val_loss_values, 'b', label='No regularization - Validation loss')
    axs[0].plot(epochs, do_loss_values, 'ro', label='Dropout - Training loss')
    axs[0].plot(epochs, do_val_loss_values, 'r', label='Dropout - Validation loss')
    axs[0].plot(epochs_es, es_loss_values, 'go', label='Early stopping - Training loss')
    axs[0].plot(epochs_es, es_val_loss_values, 'g', label='Early stopping  - Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].legend()

    axs[1].plot(epochs, p_train_acc, 'bo', label='No regularization - Training accuracy')
    axs[1].plot(epochs, p_val_acc, 'b', label='No regularization - Validation accuracy')
    axs[1].plot(epochs, do_train_acc, 'ro', label='Dropout - Training accuracy')
    axs[1].plot(epochs, do_val_acc, 'r', label='Dropout - Validation accuracy')
    axs[1].plot(epochs_es, es_train_acc, 'go', label='Early stopping - Training accuracy')
    axs[1].plot(epochs_es, es_val_acc, 'g', label='Early stopping  - Validation accuracy')
    axs[1].set_title('Training and validation accuracy')
    axs[1].legend()
    fig.set_size_inches(8, 5)
    plt.savefig(filename)

multiplot(plain_CC, dropout_CC, early_CC, 'plots/regularization_CC.jpg')
multiplot(plain_D, dropout_D, early_D, 'plots/regularization_D.jpg')
multiplot(plain_Y, dropout_Y, early_Y, 'plots/regularization_Y.jpg')

multiplot(plain_CC, wd1_CC, wd2_CC, 'plots/regularization_wd_CC.jpg')
multiplot(plain_D, wd1_D, wd2_D, 'plots/regularization_wd_D.jpg')
multiplot(plain_Y, wd1_Y, wd2_Y, 'plots/regularization_wd_Y.jpg')