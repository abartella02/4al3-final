'''

NOTES:
- label 1 = AI generated
- label 0 = not AI generated

- In final_train.csv, there are
    - 124823 instances with label = 1
    - 222154 instances with label = 0

- In final_test.csv, there are
    - 30742 instances with label = 1
    - 55845 instances with label = 0



'''



import pandas as pd
import numpy as np
import re



def data_cleanup(dataset, amount_per_class):

    # drop any rows from the train data that has emojis
    emoji_rows = dataset[dataset['text'].str.contains(r'[\u263a-\U0001f645]')]
    dataset = dataset.drop(emoji_rows.index, axis=0)

    # create dataframe for new dataset
    shortened_data = pd.DataFrame()

    # get an equal amount of random samples from each class and add it to the new dataset df
    label_1 = dataset[dataset['label']==1].sample(amount_per_class, random_state=42)
    shortened_data = shortened_data._append(label_1)
    label_0 = dataset[dataset['label']==0].sample(amount_per_class, random_state=42)
    shortened_data = shortened_data._append(label_0)

    # shuffle and reset the index for the new df
    shortened_data = shortened_data.sample(frac=1)
    shortened_data = shortened_data.reset_index(drop=True)

    return shortened_data


# load the datasets

train_dataset = pd.read_csv('final_train.csv')
test_dataset = pd.read_csv('final_test.csv')


# say 27000 samples per class, so 54000 samples altogether for training
train_shortened_data = data_cleanup(train_dataset, amount_per_class=27000)

# put this new train df into a csv
train_shortened_data.to_csv('edited_train.csv', index=False)
print("shortened train data now in new csv")

# say 5000 samples per class, so 10000 samples altogether for testing
test_shortened_data = data_cleanup(test_dataset, amount_per_class=5000)

# put this new test df into a csv
test_shortened_data.to_csv('edited_test.csv', index=False)
print("shortened test data now in new csv")

print("done cleanup!")


















    











