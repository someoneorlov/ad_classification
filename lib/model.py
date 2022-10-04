import os
import sys
import pickle

import re
import emoji
import pymorphy2

import numpy as np
import pandas as pd

from typing import Tuple, Union
from string import punctuation

from catboost import CatBoostClassifier
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler


# ========================================
#               Functions
# ========================================

def is_full_phone(text):
  phone = re.sub("[^0-9]", "", text)
  phone = re.findall(r'[7-8]?\s?[9][0-9][0-9]?\s?[0-9][0-9][0-9]?\s?[0-9][0-9]?\s?[0-9][0-9]$', phone)
  if phone:
    phone = phone[0]
  else:
    phone = ''
  return 1 if len(phone) >= 10 else 0


def full_phone(text):
  phone = re.sub("[^0-9]", "", text)
  phone = re.findall(r'[7-8]?\s?[9][0-9][0-9]?\s?[0-9][0-9][0-9]?\s?[0-9][0-9]?\s?[0-9][0-9]$', phone)
  return phone


def is_short_phone(text):
  phone = re.sub("[^0-9]", "", text)
  phone = re.findall(r'[0-9][0-9][0-9]?\s?[0-9][0-9]?\s?[0-9][0-9]$', phone)
  if phone:
    phone = phone[0]
  else:
    phone = ''
  return 1 if (len(phone) >= 5 and  len(phone) < 10) else 0


def short_phone(text):
  phone = re.sub("[^0-9]", "", text)
  phone = re.findall(r'[0-9][0-9][0-9]?\s?[0-9][0-9]?\s?[0-9][0-9]$', phone)
  return phone


def num_words_count(text):
  words = '|'.join([
                    'один', 'два', ' три', 'четыр',  'пят', 'шесть', 
                    'семь', 'восем', 'девят', 'десят', 
                    'двенадца', 'петнад', 'шеснад', 'семнад', 'сорок',
                    'шесдес', 'семдес',  'девян', 'сто ', 'двест', 
                    'шессот', 'семсот',])
  phone_words = re.findall(r'{0}'.format(words), text.lower())
  return len(phone_words)


def digits_count(text):
  digits = re.sub("[^0-9]", "", text)
  return len(digits)


def contact_words_count(text):
  words = '|'.join([
                    'почта', 'mail', '@', 'вконтакте', 'vk', 'скайп', 'skype', 
                    'inst', 'инст', 'тикток', 'tiktok', 'what', 'вотс', 'ватс', 
                    'teleg', 'телеф', 'тел\.', 't\.me', 'вайбер', 'вибер', 'viber', 
                    'дискорд', 'discort', 'ютуб', 'ютюб', 'ютью', 'youtub', 
                    'com', 'net', 'org', 'ru', '.pro', 'id', 'звон', 'youl', 'юла'])
  contact_words = re.findall(r'{0}'.format(words), text.lower())
  return len(contact_words)


def tel_count(text):
  contact_words = re.findall(r'тел', text.lower())
  return len(contact_words)


def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')


def count_emoji(text):
    return emoji.emoji_count(text)


def unique_words(df_col):
    text = df_col.agg(' '.join)[0]
    text = " ".join(set(text.split()))
    text = " ".join(word.lower() for word in text.split())
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)
    text = re.sub('[«».,:;%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) 
    text = remove_emoji(text)
    words = [word for word in text.split() 
            if word != " "
            and word.strip() not in punctuation
            and len(word) > 1]
    return list(set(words))


def preprocess_text(text, norm_dict, stop_words):
    text = " ".join(word.lower() for word in text.split())
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)
    text = re.sub('[«».,:;%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    text = remove_emoji(text)
    text = " ".join([
        norm_dict[word] for word in text.split() 
        if word != " "
        and word.strip() not in punctuation
        and len(word) > 1
        and word in norm_dict
        and word not in stop_words
        ])
    return text


def preprocess_text_light(text):
    text = " ".join(word.lower() for word in text.split())
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)
    text = re.sub('[]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    text = remove_emoji(text)
    text = " ".join([
        word for word in text.split() 
        if word != " "
        and word.strip() not in punctuation
        ])
    return text

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def task1(test: pd.DataFrame) -> list:

    MODEL_PATH = '/app/lib/pretrained_models/'

    # ========================================
    #               Common preprocess
    # ========================================

    with open('{}catboost_model/stop_words.pickle'.format(MODEL_PATH), 'rb') as f:
        stop_words = pickle.load(f)

    description_words = unique_words(test[['description']])
    title_words = unique_words(test[['title']])
    words = list(set(description_words + title_words))

    morph = pymorphy2.MorphAnalyzer()
    norm_dict = {}

    for word in words:
        if word not in norm_dict:
            norm_dict[word] = morph.parse(word)[0].normal_form
    params = {
        'norm_dict': norm_dict, 
        'stop_words': stop_words}

    test['description_prep'] = test['description'].parallel_apply(lambda x: preprocess_text(x, **params))
    test['title_prep'] = test['title'].parallel_apply(lambda x: preprocess_text(x, **params))

    test['title_prep_l'] = test['title'].parallel_apply(lambda x: preprocess_text_light(x))
    test['desc_prep_l'] = test['description'].parallel_apply(lambda x: preprocess_text_light(x))

    test['is_empty_price'] = test['price'].isna().astype('int64')
    test['month'] = pd.to_datetime(test['datetime_submitted']).dt.month
    test['hour'] = pd.to_datetime(test['datetime_submitted']).dt.hour
    test['is_full_phone'] = test['description'].parallel_apply(lambda x: is_full_phone(x))
    test['full_phone'] = test['description'].parallel_apply(lambda x: full_phone(x))
    test['is_short_phone'] = test['description'].parallel_apply(lambda x: is_short_phone(x))
    test['short_phone'] = test['description'].parallel_apply(lambda x: short_phone(x))
    test['num_words_count'] = test['description'].parallel_apply(lambda x: num_words_count(x))
    test['digits_count'] = test['description'].parallel_apply(lambda x: digits_count(x))
    test['contact_words_count'] = test['description'].parallel_apply(lambda x: contact_words_count(x))
    test['tel_count'] = test['description'].parallel_apply(lambda x: tel_count(x))
    test['count_emoji'] = test['description'].parallel_apply(lambda x: count_emoji(x))
    
    test.drop(columns=['description', 'title', 'datetime_submitted', 'city'], inplace=True)

    # ========================================
    #               CATBOOST
    # ========================================

    with open('{}catboost_model/median_price.pickle'.format(MODEL_PATH), 'rb') as f:
        median_price = pickle.load(f)
    with open('{}catboost_model/scaler.pickle'.format(MODEL_PATH), 'rb') as f:
        scaler = pickle.load(f)

    text_features = ['text']
    cat_features = ['category', 'subcategory', 'region', 'month', 'hour']
    num_features = ['price', 'num_words_count', 'digits_count', 
                    'contact_words_count', 'tel_count', 'count_emoji']
    bin_features = ['is_empty_price', 'is_full_phone', 'is_short_phone']

    test['title_prep'] = test['title_prep'].astype('str')
    test['description_prep'] = test['description_prep'].astype('str')
    test['category'] = test['category'].astype('str')
    test['subcategory'] = test['subcategory'].astype('str')
    test['region'] = test['region'].astype('str')
    test['month'] = test['month'].astype('str')
    test['hour'] = test['hour'].astype('str')

    test['text'] = test[['title_prep', 'description_prep']].agg(' '.join, axis=1)
    test['price'] = test['price'].fillna(test['subcategory'].map(median_price))
    test['price'] = scaler.transform(test['price'].values.reshape(-1, 1))

    catboost = CatBoostClassifier().load_model('{}catboost_model/catboost_model.cbm'.format(MODEL_PATH))
    catboost_predictions = catboost.predict_proba(
        test[text_features + cat_features + num_features + bin_features]
        )[:, 1]
    
    test.drop(columns=['title_prep', 'description_prep', 'text', 'price'], inplace=True)

    # ========================================
    #               BERT
    # ========================================
    
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('{}bert_model/'.format(MODEL_PATH))
    model = BertForSequenceClassification.from_pretrained('{}bert_model/'.format(MODEL_PATH))

    model.to(device)

    test['is_empty_price'] = test['is_empty_price'].parallel_apply(
        lambda x: 'цена пустая' if pd.isna(x) else 'цена {}'.format(x))
    test['month'] = test['month'].parallel_apply(
        lambda x: 'месяц {}'.format(str(x))).astype('str')
    test['hour'] = test['hour'].parallel_apply(
        lambda x: 'час {}'.format(str(x))).astype('str')
    test['is_full_phone'] = test['is_full_phone'].parallel_apply(
        lambda x: 'есть полный' if x == 1 else 'нет полного').astype('str')
    test['is_short_phone'] = test['is_short_phone'].parallel_apply(
        lambda x: 'есть короткий' if x == 1 else 'нет короткого').astype('str')
    test['num_words_count'] = test['num_words_count'].parallel_apply(
        lambda x: 'слов {}'.format(str(x))).astype('str')
    test['digits_count'] = test['digits_count'].parallel_apply(
        lambda x: 'цифр {}'.format(str(x))).astype('str')
    test['contact_words_count'] = test['contact_words_count'].parallel_apply(
        lambda x: 'контактных {}'.format(str(x))).astype('str')
    test['tel_count'] = test['tel_count'].parallel_apply(
        lambda x: 'телефонов {}'.format(str(x))).astype('str')
    test['count_emoji'] = test['count_emoji'].parallel_apply(lambda x: 'эмодзи {}'.format(str(x))).astype('str')

    join_cols = [
        'title_prep_l', 'desc_prep_l', 'subcategory', 'category', 
        'region', 'is_empty_price', 'month', 'hour', 
        'is_full_phone', 'is_short_phone', 'num_words_count', 
        'digits_count', 'contact_words_count', 'tel_count', 'count_emoji']

    test['text'] = test.fillna('')[join_cols].agg(' '.join, axis=1)
    test.drop(columns=join_cols, inplace=True)

    sentences = test['text'].values
    max_len = 270

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent, 
                            add_special_tokens = True,
                            max_length = max_len,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    batch_size = 64
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    model.eval()
    predictions = []

    for batch in prediction_dataloader:

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask  = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        
        predictions.append(logits)

    bert_predictions = np.concatenate(predictions, axis=0)
    bert_predictions = bert_predictions[:, 1].flatten()

    # # ========================================
    # #               STACKING
    # # ========================================

    with open('{}stack/log_reg.pickle'.format(MODEL_PATH), 'rb') as f:
        log_reg = pickle.load(f)

    X = np.hstack((
        catboost_predictions.reshape(-1, 1), 
        bert_predictions.reshape(-1, 1)))

    fin_predictions = log_reg.predict_proba(X)[:, 1]
    
    return fin_predictions


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)
    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
