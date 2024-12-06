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

# Unchanged Jackie function
def data_cleanup(dataset, amount_per_class):
    # Drop rows with NaN in the 'text' column
    dataset = dataset.dropna(subset=['text'])

    # Drop any rows from the train data that has emojis
    emoji_rows = dataset[dataset['text'].str.contains(r'[\u263a-\U0001f645]', na=False)]
    dataset = dataset.drop(emoji_rows.index, axis=0)

    # Create dataframe for new dataset
    shortened_data = pd.DataFrame()

    # Get an equal amount of random samples from each class and add it to the new dataset df
    np.random.seed(123)
    label_1 = dataset[dataset['label'] == 1].sample(amount_per_class, random_state=42)
    shortened_data = shortened_data._append(label_1)
    label_0 = dataset[dataset['label'] == 0].sample(amount_per_class, random_state=42)
    shortened_data = shortened_data._append(label_0)

    # Shuffle and reset the index for the new df
    shortened_data = shortened_data.sample(frac=1)
    shortened_data = shortened_data.reset_index(drop=True)

    return shortened_data


# function to convert each text sample into single paragraphs with no spacing, new lines, etc
def condense_text(data, text_column_name):

    # Replace multiple line breaks with a space for each item in the column
    data[text_column_name] = data[text_column_name].str.replace(r'\s*\n\s*', ' ', regex=True)

    # Normalize spaces (remove multiple spaces)
    data[text_column_name] = data[text_column_name].str.replace(r'\s+', ' ', regex=True).str.strip()

    return data


# just combines the above functions and makes switching between train and test easier
def sample_and_clean(data: pd.DataFrame, samples_per_class: int, to_csv: bool = False, output_csv_path='data/preprocessed.csv') -> pd.DataFrame:

    # load the datasets

    # dataframe = pd.read_csv(input_csv)

    # shorten the size of the dataset
    train_shortened_data = data_cleanup(data, samples_per_class)

    # get rid of whitespace and other stuff from the texts
    train_no_space_data = condense_text(train_shortened_data, 'text')

    # rename columns
    train_no_space_data.columns = ["Text", "Label"]


    # put this new train df into a csv
    if to_csv:
        train_no_space_data.to_csv(output_csv_path, index=False)
    return train_no_space_data


# cleanup_time('final_train.csv', 1000, "edited_train.csv")
