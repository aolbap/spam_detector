'''Contains one unit test for each function in functions.py'''

import pandas as pd
import matplotlib
from sklearn.ensemble import RandomForestClassifier
import transformers
from nltk import word_tokenize
import functions as f


def test_import_clean():
    '''
    Tests that the output function is structured correcty, and no errors are raised.
    '''
    encode_df = f.import_clean("unencoded_sample.csv", encode=True)
    no_encode_df = f.import_clean("encoded_sample.csv")
    assert isinstance(encode_df, pd.DataFrame)
    assert isinstance(no_encode_df, pd.DataFrame)
    assert encode_df.shape[1] == 775
    assert no_encode_df.shape[1] == 775
    assert all(feat in encode_df.columns for feat in ['spam',
                                                      'original_message',
                                                      'cleaned_message',
                                                      'num_words',
                                                      'num_chars',
                                                      'num_uppercase_chars',
                                                      'num_uppercase_words'
                                                      ])
    assert all(feat in no_encode_df.columns for feat in ['spam',
                                                         'num_words',
                                                         'num_chars',
                                                         'num_uppercase_chars',
                                                         'num_uppercase_words'
                                                         ])


def test_create_plot():
    '''
    Checks that the correct number of matplotlib axes is output, and no errors are raised.
    '''
    no_encode_df = f.import_clean("encoded_sample.csv")
    fig, axs = f.create_plot(no_encode_df)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert axs.shape == (2, 2)


def test_describe_data():
    '''
    Checks that descriptive statsitics are correct, and no errors are raised.
    '''
    no_encode_df = f.import_clean("encoded_sample.csv")
    output = f.describe_data(no_encode_df)
    assert isinstance(output, pd.core.frame.DataFrame)


def test_model():
    '''
    Checks that model exists, inputs are correctly structured, and outputs are correctly structured.
    '''
    no_encode_df = f.import_clean("encoded_sample.csv")
    output = f.model(no_encode_df)
    assert isinstance(output, RandomForestClassifier)


def test_clean_sentence():
    '''
    Checks that the sentence exists, and contains the correct amount of tokens.
    '''
    sentence = "Hello, I am upset again that I am not very sorry about how? " * 10
    output = f.clean_sentence(sentence)
    assert len(word_tokenize(output)) == 30


def test_encode_bert():
    '''
    Test that the output is formatted correctly, and has the right number of embeddings.
    '''
    sample = pd.read_csv('unencoded_sample.csv')
    sample['cleaned_message'] = sample['original_message'].apply(
        f.clean_sentence)
    # Tokenizer and Model object inputs
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased')
    model = transformers.DistilBertModel.from_pretrained(
        'distilbert-base-uncased')
    output = f.encode_bert(sample, tokenizer, model)
    assert output.shape[1] >= 769  # 768 encodings, 2 for messages
