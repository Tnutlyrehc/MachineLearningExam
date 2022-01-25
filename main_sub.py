import pandas as pd
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray

path = 'data'
raw_imgs = []

#Function denfined for loading the data (no split)
def load_data(path):
    raw_imgs = []
    labels = []
    container = []

    for filename in os.listdir(path + '/original_data'):
        container.append(filename) #We want to add the filename to a container
        CCDY_img = load_img(path + f'/{filename}',
                            target_size = (150, 84),
                            color_mode="grayscale") #
        CCDY_img = img_to_array(CCDY_img)

        raw_imgs.img_to_array(CCDY_img)

        raw_imgs.append(CCDY_img)
        return asarray(raw_imgs), container;




def load_1d_grays():
    string_digits = pd.read_csv('DIDA_12000_String_Digit_Labels.csv',
                                header=None,
                                names=["index", "string"])
    # create empty class columns
    string_digits['CC'] = 0
    string_digits['D'] = 0
    string_digits['Y'] = 0
    string_digits = string_digits.astype(str)
    for i, row in string_digits.iterrows():
        if len(row['string']) != 4:
            row['CC'] = '1'
            row['D'] = '10'
            row['Y'] = '10'
        else:
            row['D'] = row['string'][2]
            row['Y'] = row['string'][3]
            if row['string'][0:2] == '18':
                row['CC'] = '0'
            else:
                row['CC'] = '1'

    # os.chdir(path_images)
    image_array, filename = load_data('DIDA_12000_String_Digit_Images/DIDA_1')

    img_df = pd.DataFrame({'filename': filename, 'gray_value': list(image_array)},
                          columns=['filename', 'gray_value'])

    #Merge img array with label df
    img_df['index'] = img_df['filename']
    for i, row in img_df.iterrows():
        row['index'] = str(img_df['index'][i]).split('.')[0]


    string_digits['index'] = string_digits['index'].astype(int)
    img_df['index'] = img_df['index'].astype(int)

    df_img_classes = string_digits.merge(img_df)

    df_img_classes = df_img_classes.reindex(columns=['index', 'string', 'CC', 'D', 'Y', 'gray_value', 'filename'])
    return df_img_classes

