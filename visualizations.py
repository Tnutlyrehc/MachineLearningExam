from matplotlib import image
from matplotlib import pyplot as plt
import os



abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

data = image.imread('data/train/258.jpg')
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
plt.imshow(data)
plt.show()



plt.plot(CC_fit.history['loss'], label='Training loss')
plt.plot(CC_fit.history['val_loss'], label='Validation loss')
plt.plot(hist_ex1.history['loss'], label='Training loss (regularized)')
plt.plot(hist_ex1.history['val_loss'], label='Validation loss (regularized)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.plot(hist.history['accuracy'], label='Training accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation accuracy')
plt.plot(hist_ex1.history['accuracy'], label='Training accuracy (regularized)')
plt.plot(hist_ex1.history['val_accuracy'], label='Validation accuracy (regularized)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()