'''Used to import, clean, visualize, describe, and model a predictive random forest classifier
which classifies text messages as either spam or not spam. The model depends on the use of
distilBERT encodings as explanatory variables, a smaller version of the BERTtransformer model.
'''
import re
import string
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import transformers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def import_clean(filename='encoded_spam_final.csv',
                 spam='spam',
                 mess='original_message',
                 encode=False,
                 index_col=0
                 ):
    '''
    Imports data from csv and adds features.

    Parameters:
    ==========
        filename : path including filename from which to load data
        columns: List of required column names. Index 0 is spam variable 1 or 0.
        Index 1 is the original_message.

        encode: boolean. If True, will create and add transformer encodings to dataset.
        index_col (optional): column to be taken as the index for input file data


    Returns:
    ========
    Pandas DataFrame containing the original message,
    cleaned message, distilBERT encodings, word count, character count,
    capital character count, capital word count.
    '''
    # Read in CSV
    in_df = pd.read_csv(filename, index_col=index_col)

    # Change column names
    in_df.rename({spam: 'spam', mess: 'original_message'})

    if encode:
        if in_df.shape[0] > 150:
            print("WARNING: Encoding a large dataset may use significant \
            memory")

        # Clean Message for BERT encoding
        in_df["cleaned_message"] = in_df["original_message"].apply(
            clean_sentence)

        # Tokenizer and Model object inputs
        tokenizer_in = transformers.DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
        model_in = transformers.DistilBertModel.from_pretrained(
            'distilbert-base-uncased')
        # Generate BERT encodings:
        df_encoded = encode_bert(in_df, tokenizer_in, model_in)
    else:
        df_encoded = in_df

    # Add features
    # Ensure message is a string for findall
    df_encoded['original_message'] = df_encoded['original_message'].astype(str)
    # Add Number of Words in text original_message
    df_encoded["num_words"] = df_encoded["original_message"].apply(
        lambda s: len(re.findall(r'\w+', s)))

    # Get the length of the text original_message
    df_encoded["num_chars"] = df_encoded["original_message"].apply(len)

    # Count the number of uppercased characters
    df_encoded["num_uppercase_chars"] = df_encoded["original_message"].apply(
        lambda s: sum(1 for c in s if c.isupper()))

    # Count the number of uppercased words
    df_encoded["num_uppercase_words"] = df_encoded["original_message"].apply(
        lambda s: len(re.findall(r"\b[A-Z][A-Z]+\b", s)))

    return df_encoded


def create_plot(in_df):
    '''
    Creates scatterplots of word count, character count, capital character count,
    capital word count vs spam.

    Parameters:
        dataframe : Pandas Dataframe structured as outputted from import_clean.

    Returns: matplotlib axes
    '''
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # Generate barplots of non-BERT variables
    sns.barplot(data=in_df, y='num_words', x='spam', ax=axs[0, 0])
    axs[0, 0].set_ylabel('Mean Number of Words')

    sns.barplot(data=in_df, y='num_chars', x='spam', ax=axs[0, 1])
    axs[0, 1].set_ylabel('Mean Number of Characters')

    sns.barplot(data=in_df, y='num_uppercase_chars', x='spam', ax=axs[1, 0])
    axs[1, 0].set_ylabel('Mean Number of Uppercase Characters')

    sns.barplot(data=in_df, y='num_uppercase_words', x='spam', ax=axs[1, 1])
    axs[1, 1].set_ylabel('Mean Number of Uppercase Words')

    # Set title
    fig.suptitle('Mean of exogenous variables grouped by endogenous Spam')

    return fig, axs


def describe_data(cleaned_df):
    '''
    Returns a dataframe with descriptive statistics for non-BERT encoding
    explanatory variables, and prints highlights alongside the proportion of
    dependent variable spam.

    Parameters:
        dataframe : Pandas Dataframe structured as outputted from import_clean.

    Returns: Dictionary
    '''
    descriptor = cleaned_df[['num_words', 'num_chars',
                             'num_uppercase_chars', 'num_uppercase_words']].describe()
    value_counts = cleaned_df['spam'].value_counts()
    proportion = value_counts.loc[1]/(value_counts.loc[0]+value_counts.loc[1])

    print(
        f"There are {cleaned_df.shape[0]} observations in your cleaned dataset. ")
    print(
        f"The percent of messages in your dataset that are spam is \
            {round(proportion, 3)*100}%")
    print(
        f"The mean number of words in each message in the dataset is \
            {round(descriptor['num_words']['mean'], 2)}")
    print(
        f"The mean number of characters in each message in the dataset is \
            {round(descriptor['num_chars']['mean'],2)}")
    print(
        f"The mean number of uppercase characters in each message in the dataset is \
            {round(descriptor['num_uppercase_chars']['mean'],2)}")
    print(
        f"The mean number of uppercase words in each message in the dataset is \
            {round(descriptor['num_uppercase_words']['mean'],2)}")

    return descriptor


def model(cleaned_df, estimators=200, test_size=0.3):
    '''
    Trains model using a RandomForestClassifier, with a test-train split of 0.3:0.7.

    Parameters:
        data : Pandas Dataframe structured as outputted from import_clean.

    Returns:
        trained Random Forest Classifier model object
    Prints:
        Model Accuracy score
    '''
    # Designate features, labels
    features = cleaned_df.drop(
        ['spam', "original_message", 'cleaned_message'], axis=1)
    labels = cleaned_df['spam']

    # Split data for cross validation
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, random_state=0, test_size=test_size)

    # Train Model
    rf_clf = RandomForestClassifier(n_estimators=estimators)
    rf_clf.fit(train_features, train_labels)
    print(f"Model Accuracy :{rf_clf.score(test_features, test_labels)}")

    return rf_clf


def clean_sentence(sentence):
    """
    Remove punctuation and stop words from a sentence and return first 30 characters.

    Parameters
    ==========
    sentence: sentence to be cleaned

    Returns
    =======
    cleaned sentence
    """
    stop_words = set(stopwords.words('english'))
    # remove punctuation
    no_punkt = sentence.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(no_punkt)
    # removing stop-words
    cleaned_s = [w for w in tokens if w not in stop_words]
    return " ".join(cleaned_s[:30])  # using the first 30 tokens only


def encode_bert(chunk_df, tokenizer, bert_model):
    '''applies BERT encoding to specific dataset

    Parameters:
    ===========
    Chunk_df: Pandas dataframe with column cleaned_message
    Output: df that has been encoded'''

    # Tokenize the sentences adding the special [cls] abnd [sep] tokens
    tokenized = chunk_df["cleaned_message"].apply(
        lambda x: tokenizer.encode(x, add_special_tokens=True))

    # Get the length of the longest tokenized sentence
    max_len = tokenized.apply(len).max()

    # Pad the rest of the sentence with zeros if the sentence is smaller than the longest sentence
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # Create the attention mask so BERT knows to ignore the zeros used for padding
    attention_mask = np.where(padded != 0, 1, 0)

    # Create the input tensors
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    # Pass the inputs through DistilBERT
    with torch.no_grad():
        encoder_hidden_state = bert_model(
            input_ids, attention_mask=attention_mask)

    # Create a new dataframe with the encoded features
    encoded_chunk = pd.DataFrame(encoder_hidden_state[0][:, 0, :].numpy())

    # Insert the original columns in the beginning of the encoded dataframe
    encoded_chunk = pd.concat(
        [chunk_df.reset_index(drop=True), encoded_chunk], axis=1)
    return encoded_chunk
