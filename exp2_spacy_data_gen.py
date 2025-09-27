import json
import os
import re
import string

import unicodedata
from datasets import load_dataset
import spacy
import random

nlp = spacy.load("en_core_web_sm")


def normalize_sentence(s):
    # def remove_articles(text):
    #     return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    # def remove_punc(text):
    #     exclude = set(string.punctuation)
    #     return ''.join(ch for ch in text if ch not in exclude)

    # def lower(text):
    #     return text.lower()

    # def remove_special_characters(text):
    #     return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    #
    # def unicode_lower(text):
    #     text = re.sub(r'[\u2013-\u2014]', ' ', text)
    #     return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()
    #
    # if not s or not s.strip():
    #     return ''
    return white_space_fix(s)


def extract_verb_phrases(token):
    return ' '.join([sub_token.text for sub_token in token.subtree])


def process_sentences(sentences):
    input_data = []

    for idx, sentence_o in enumerate(sentences, start=1):
        sentence = normalize_sentence(sentence_o)
        doc = nlp(sentence)
        data = []

        for chunk in doc.noun_chunks:
            for token in chunk:
                if token.text == chunk.root.text:  # Ensure we are using the head of the noun chunk
                    word_location = (token.idx, token.idx + len(token.text))
                    word_group = chunk.text
                    first_token = chunk[0]
                    last_token = chunk[-1]
                    word_group_location = (chunk.start_char, chunk.end_char)
                    data.append({
                        'word': token.text,
                        'word_location': list(word_location),
                        'word_group': word_group,
                        'word_group_location': list(word_group_location)
                    })

        input_data.append({
            'id': str(idx),
            'sentence': sentence,
            'data': data
        })

    return input_data


def parsing(sentences):
    total_data = []
    total_token_case = 0
    for idx, sentence_o in enumerate(sentences):
        sentence = normalize_sentence(sentence_o)
        doc = nlp(sentence)
        data = []
        for token in doc:
            if token.dep_ == "det" or token.is_punct:
                continue
            # 获取 head 和 children
            head = token.head
            children = list(token.children)

            word_group_set = {head.text}  # 使用集合来避免重复, 加入head
            word_group_location_set = {(head.idx, head.idx + len(head.text))}

            word_group_set.add(token.text)  # 加入当前 token 自身
            word_group_location_set.add((token.idx, token.idx + len(token.text)))
            for child in children:
                word_group_set.add(child.text)  # 加入子节点文本
                word_group_location_set.add((child.idx, child.idx + len(child.text)))  # 加入子节点的位置

            # 构建词组
            # word_group = [head.text] + [child.text for child in children]
            # word_group_location = sorted([head.idx] + [token.idx] + [child.idx for child in children])
            # word_group_location = sorted(list(word_group_location_set))
            word_group_location = sorted(list(word_group_location_set), key=lambda x: x[0])
            if word_group_set:
                # 构建输出数据
                data.append({
                    'word': token.text,
                    'word_location': (token.idx, token.idx + len(token.text)),
                    'word_group': list(word_group_set),
                    'word_group_location': word_group_location
                })
                total_token_case += 1
        total_data.append({
            'id': str(idx),
            'sentence': sentence,
            'data': data
        })

    return total_data, total_token_case



# Save the result to a JSON file
def save_results(output_dir, dataset_name, results):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, dataset_name)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def clean_and_split_sentences_num(text_list, num_sentences):
    """
    Clean the text list by removing empty strings and splitting the text into sentences.
    Stops when the required number of sentences is reached.
    """
    cleaned_sentences = []

    # Iterate over the text list and process each item until we have enough sentences
    for text in text_list:
        # Split the text by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Clean and filter sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Add the cleaned sentences to the list
        cleaned_sentences.extend(sentences)

        # Check if we've reached the desired number of sentences
        if len(cleaned_sentences) >= num_sentences:
            return cleaned_sentences[:num_sentences]

    # In case there are fewer sentences than requested, return all available sentences
    return cleaned_sentences


def load_corpus(corpus_name, num_sentences=100):
    """
       Load a specified corpus, clean and split sentences, and return a list of cleaned sentences.
       Stops processing when the required number of sentences is reached.
    """
    corpus_map = {
        'gsm8k': ("gsm8k", "main", 'question'),
        'Yelp': ("yelp_review_full", None, 'text'),
        'glue': ("glue", "mnli", 'hypothesis'),
        'cnn_dailymail': ("cnn_dailymail", "3.0.0", 'article'),
        'openOrca': ("Open-Orca/OpenOrca", None, 'question'),
        'wikitext': ("wikitext", "wikitext-2-v1", 'text'),
    }

    if corpus_name not in corpus_map:
        raise ValueError(f"Corpus '{corpus_name}' not available. Choose from {list(corpus_map.keys())}")

    dataset_name, config, field = corpus_map[corpus_name]
    dataset = load_dataset(dataset_name, config, trust_remote_code=True)

    # cleaned_sentences = clean_and_split_sentences(raw_text_list)

    # Load and clean sentences with early stopping
    raw_text_list = dataset['train'][field]
    random.shuffle(raw_text_list)
    cleaned_sentences = clean_and_split_sentences_num(raw_text_list, num_sentences)
    random.shuffle(cleaned_sentences)
    max_words = 30
    truncated_sentences = []
    for sentence in cleaned_sentences:
        words = sentence.split()  # Split the sentence into words
        if len(words) > max_words:
            truncated_sentences.append(' '.join(words[:max_words]))  # Truncate and join back to a string
        else:
            truncated_sentences.append(sentence)
    return truncated_sentences[:num_sentences]


'''
Demo code of experiment to construct a specialized word dependency dataset using SpaCy.
'''

# Process the sentences

dataset_name = 'gsm8k'  # 'gsm8k', 'Yelp', 'glue', 'cnn_dailymail', 'openOrca', 'wikitext',
sentences_num = 1000
input_sentences = load_corpus(dataset_name, sentences_num)

result, total_token_case = parsing(input_sentences)
save_results(f'exp2/data', f'dependence_datasets_spacy_{sentences_num}_{total_token_case}.json', result)

