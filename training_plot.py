import numpy as np
import matplotlib.pyplot as plt
history = np.load('my_history.npy', allow_pickle='True').item()

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
plot_loss_acc(history, 'CC_overfitted.jpg')