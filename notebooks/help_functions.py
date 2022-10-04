import re
import emoji
from string import punctuation

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
    text = " ".join(word.lower() for word in text.split()) #lowercasing
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[«».,:;%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols 
    text = remove_emoji(text)
    words = [word for word in text.split() 
            if word != " "
            and word.strip() not in punctuation
            and len(word) > 1]
    return list(set(words))


def preprocess_text(text, norm_dict, stop_words):
    text = " ".join(word.lower() for word in text.split()) #lowercasing
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[«».,:;%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols 
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
    text = " ".join(word.lower() for word in text.split()) #lowercasing
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols 
    text = remove_emoji(text)
    text = " ".join([
        word for word in text.split() 
        if word != " "
        and word.strip() not in punctuation
        ])
    return text