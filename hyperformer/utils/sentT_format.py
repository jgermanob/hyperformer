import pandas as pd

def add_sentinel_tokens(tokenized_text):
        processed_text = ''
        for i,token in enumerate(tokenized_text):
            processed_text += '<extra_id_{}> {} '.format(i,token)
        processed_text = '{} <extra_id_{}>'.format(processed_text.strip(), i+1)
        return processed_text.strip()

def get_sentT_format(df):
    sentT_texts = []
    sentT_labels = []
    texts = df.text.values.tolist()
    labels = df.sbio_labels.values.tolist()
    for text, tag in zip(texts, labels):
        text = text.split()
        tag = tag.split()
        sentT_texts.append(add_sentinel_tokens(text))
        sentT_labels.append(add_sentinel_tokens(tag))
    return sentT_texts, sentT_labels


    