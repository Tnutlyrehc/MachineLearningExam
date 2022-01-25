from matplotlib import image
from matplotlib import pyplot
import os
from main import labels_df_CC

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

data = image.imread('data/train/258.jpg')
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()

