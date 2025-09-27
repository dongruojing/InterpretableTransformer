import json
import random
import re


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForMaskedLM, \
    AutoModelForQuestionAnswering
import numpy as np
import os
import torch
from datasets import load_dataset



def get_token_dependence_in_sequence(input_ids, token_positions, replacements):
    original_inputs = {'input_ids': torch.tensor([input_ids]).to(device)}
    # inputs = tokenizer.convert_ids_to_tokens(input_ids)
    # original_inputs = tokenizer(inputs, return_tensors='pt').to(device)
    original_hidden_states = model(**original_inputs).hidden_states[-1].squeeze(0)
    original_s_hidden_states = original_hidden_states[token_positions].to(device)

    original_token_list = tokenizer.convert_ids_to_tokens(input_ids)
    # print(original_token_list)
    var_list = []
    choose_token_list = []
    for token_index in token_positions:
        original_token = original_token_list[token_index]
        if original_token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>', '<pad>', '<unk>']:
            continue
        if original_token.startswith("##"):
            continue
        if original_token in ['.', ',', '!', '?', ';', ':', '"', '\'', '(', ')']:
            continue
        choose_token_list.append(original_token)
        delta_norms_list = []
        for replacement_token_id in replacements:
            replaced_token_ids = input_ids.copy()
            replaced_token_ids[token_index] = replacement_token_id

            # Convert token IDs to tensor and run the model
            replaced_inputs = {'input_ids': torch.tensor([replaced_token_ids]).to(device)}
            # rep_inputs = tokenizer.convert_ids_to_tokens(replaced_token_ids)
            # replaced_inputs = tokenizer(rep_inputs, return_tensors='pt').to(device)
            rep_hidden_states = model(**replaced_inputs).hidden_states[-1].squeeze(0)
            # Only take hidden states corresponding to s1 tokens
            rep_s_hidden_states = rep_hidden_states[token_positions].to(device)
            delta_norms = torch.norm(rep_s_hidden_states - original_s_hidden_states, dim=-1).detach().cpu().numpy()
            delta_norms_list.append(delta_norms)

        # Stack the hidden states and calculate variance
        delta_norms_tensor = torch.tensor(np.array(delta_norms_list)).to(device)
        variance_norms = torch.mean(delta_norms_tensor, dim=0).detach().cpu().numpy()
        var_list.append(variance_norms)

    return var_list, choose_token_list


def process_sentence_pair(s1, s2, random_num):
    original_s1_inputs = tokenizer(s1, return_tensors='pt').to(device)
    original_s2_inputs = tokenizer(s2, return_tensors='pt').to(device)

    original_s1_token_ids = original_s1_inputs['input_ids'][0].tolist()
    # original_s1_token_list = tokenizer.convert_ids_to_tokens(original_s1_token_ids)
    original_s2_token_ids = original_s2_inputs['input_ids'][0].tolist()
    # original_s2_token_list = tokenizer.convert_ids_to_tokens(original_s2_token_ids)

    s1_len = len(original_s1_token_ids)

    # Create the combined sequences
    input_s1 = original_s1_token_ids
    input_s1_s2 = original_s1_token_ids + original_s2_token_ids
    input_s2_s1 = original_s2_token_ids + original_s1_token_ids
    results = []
    # Prepare random replacements
    replacements = [random.randint(1, tokenizer.vocab_size - 1) for _ in range(random_num)]
    # Process s1 alone, get token dependence in s1
    s1_var_list, choose_token_list = get_token_dependence_in_sequence(input_s1, list(range(s1_len)), replacements)

    # Process <s1, s2>, get token dependence in s1
    s1_s2_var_list, _ = get_token_dependence_in_sequence(input_s1_s2, list(range(s1_len)), replacements)

    # Process <s2, s1>, get token dependence in s1
    s2_s1_var_list, _ = get_token_dependence_in_sequence(input_s2_s1, list(
        range(len(original_s2_token_ids), len(original_s2_token_ids) + s1_len)), replacements)

    results.append({
        'sentence1': s1,
        'sentence2': s2,
        'choose_token_list': choose_token_list,
        # 's1_variance_list': [var.tolist() for var in s1_var_list],
        # 's1_s2_variance_list': [var.tolist() for var in s1_s2_var_list],
        # 's2_s1_variance_list': [var.tolist() for var in s2_s1_var_list],

    })

    return s1_var_list, s1_s2_var_list, s2_s1_var_list, results, choose_token_list


def longest_common_subsequence(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def evaluate_dependence(corpus_name, num_sentences, chunk_size, random_num,top_num):
    sentences = load_corpus(corpus_name, num_sentences)
    final_results = []

    num_case = 0
    token_cases = 0
    context_dependent_left = 0
    context_dependent_right = 0
    order_dependent = 0
    total_context_case = 0
    total_order_case = 0

    order_influence_score = []
    context_influence_score_left = []
    context_influence_score_right = []

    # top
    context_dependent_left_top = 0
    context_dependent_right_top = 0
    order_dependent_top = 0
    order_influence_score_top = []
    context_influence_score_left_top = []
    context_influence_score_right_top = []
    for i in range(0, num_sentences, chunk_size):
        chunk = sentences[i:i + chunk_size]

        chunk_total_context_case = 0
        chunk_total_order_case = 0
        chunk_context_dependent_left = 0
        chunk_context_dependent_right = 0
        chunk_order_dependent = 0
        chunk_order_influence_score = 0
        chunk_context_influence_score_right = 0
        chunk_context_influence_score_left = 0

        for j in range(0, chunk_size, 2):
            s1, s2 = chunk[j], chunk[j + 1]
            # s1 token dependence when input s1, s1s2, s2s1
            s1_var_list, s1_s2_var_list, s2_s1_var_list, var_result, choose_token_list = process_sentence_pair(s1, s2,
                                                                                                               random_num)
            # var_results.append(var_result)
            # num_case += 1
            # print(f"case{num_case}")
            for token_idx in range(len(s1_var_list)):
                # for each token that is chosen to change in s1
                s1_var = s1_var_list[token_idx]
                s1_s2_var = s1_s2_var_list[token_idx]
                s2_s1_var = s2_s1_var_list[token_idx]

                # 检查排序顺序是否一致,返回的是从小到大的token的索引
                rank_s1 = np.argsort(s1_var)
                rank_s1_s2 = np.argsort(s1_s2_var)
                rank_s2_s1 = np.argsort(s2_s1_var)

                # 比较rank是否一致
                if not np.array_equal(rank_s1, rank_s1_s2):
                    context_dependent_right += 1
                    chunk_context_dependent_right += 1
                if not np.array_equal(rank_s1, rank_s2_s1):
                    context_dependent_left += 1
                    chunk_context_dependent_left += 1
                if not np.array_equal(rank_s1_s2, rank_s2_s1):
                    order_dependent += 1
                    chunk_order_dependent += 1

                # 计算最长公共子序列个数
                lcs_length_s1_s1_s2 = longest_common_subsequence(rank_s1, rank_s1_s2)
                lcs_length_s1_s2_s1 = longest_common_subsequence(rank_s1, rank_s2_s1)
                lcs_length_s1_s2_s2_s1 = longest_common_subsequence(rank_s1_s2, rank_s2_s1)

                s1_len = len(rank_s1)
                order_influence_score_s1_s2_s2_s1 = (s1_len - lcs_length_s1_s2_s2_s1) / s1_len if s1_len > 0 else 0
                context_influence_score_s1_s1_s2 = (s1_len - lcs_length_s1_s1_s2) / s1_len if s1_len > 0 else 0
                context_influence_score_s1_s2_s1 = (s1_len - lcs_length_s1_s2_s1) / s1_len if s1_len > 0 else 0

                order_influence_score.append(order_influence_score_s1_s2_s2_s1)
                context_influence_score_right.append(context_influence_score_s1_s1_s2)
                context_influence_score_left.append(context_influence_score_s1_s2_s1)
                total_context_case += 1
                total_order_case += 1

                # chunk
                chunk_total_context_case += 1
                chunk_total_order_case += 1
                chunk_order_influence_score += order_influence_score_s1_s2_s2_s1
                chunk_context_influence_score_right += context_influence_score_s1_s1_s2
                chunk_context_influence_score_left += context_influence_score_s1_s2_s1

                # top 5
                top_indices = np.argsort(-s1_var)[:top_num]
                s1_top_var = s1_var[top_indices]
                s1_s2_top_var = s1_s2_var[top_indices]
                s2_s1_top_var = s2_s1_var[top_indices]

                rank_top_s1 = np.argsort(s1_top_var)
                rank_top_s1_s2 = np.argsort(s1_s2_top_var)
                rank_top_s2_s1 = np.argsort(s2_s1_top_var)

                if not np.array_equal(rank_top_s1, rank_top_s1_s2):
                    context_dependent_right_top += 1
                if not np.array_equal(rank_top_s1, rank_top_s2_s1):
                    context_dependent_left_top += 1
                if not np.array_equal(rank_top_s1_s2, rank_top_s2_s1):
                    order_dependent_top += 1
                lcs_length_s1_s1_s2_top = longest_common_subsequence(rank_top_s1, rank_top_s1_s2)
                lcs_length_s1_s2_s1_top = longest_common_subsequence(rank_top_s1, rank_top_s2_s1)
                lcs_length_s1_s2_s2_s1_top = longest_common_subsequence(rank_top_s1_s2, rank_top_s2_s1)

                s1_len_top = len(rank_top_s1)
                order_influence_score_s1_s2_s2_s1_top = (s1_len_top - lcs_length_s1_s2_s2_s1_top) / s1_len_top if s1_len_top > 0 else 0
                context_influence_score_s1_s1_s2_top = (s1_len_top - lcs_length_s1_s1_s2_top) / s1_len_top if s1_len_top > 0 else 0
                context_influence_score_s1_s2_s1_top = (s1_len_top - lcs_length_s1_s2_s1_top) / s1_len_top if s1_len_top > 0 else 0
                order_influence_score_top.append(order_influence_score_s1_s2_s2_s1_top)
                context_influence_score_right_top.append(context_influence_score_s1_s1_s2_top)
                context_influence_score_left_top.append(context_influence_score_s1_s2_s1_top)

        chunk_result = {
            'chunk_number': i // chunk_size + 1,
            'total_context_case(left or right)': chunk_total_context_case,
            'context_dependent_left': chunk_context_dependent_left,
            'context_dependent_rate_left': chunk_context_dependent_left / chunk_total_context_case if chunk_total_context_case > 0 else 0,
            'context_influence_score_left': chunk_context_influence_score_left / chunk_total_context_case if chunk_total_context_case > 0 else 0,

            'context_dependent_right': chunk_context_dependent_right,
            'context_dependent_rate_right': chunk_context_dependent_right / chunk_total_context_case if chunk_total_context_case > 0 else 0,
            'context_influence_score_right': chunk_context_influence_score_right / chunk_total_context_case if chunk_total_context_case > 0 else 0,

            'total_order_case': chunk_total_order_case,
            'order_dependent_case': chunk_order_dependent,
            'order_dependent_rate': chunk_order_dependent / chunk_total_order_case if chunk_total_order_case > 0 else 0,
            'order_influence_score': chunk_order_influence_score / chunk_total_order_case if chunk_total_order_case > 0 else 0
        }
        print(chunk_result)
        if total_order_case > 10000:
            break
    final_results.append({
        'total_context_case(left or right)': total_context_case,
        'context_dependent_left': context_dependent_left,
        'context_dependent_left_top': context_dependent_left_top,
        'context_dependent_rate_left': context_dependent_left / total_context_case if total_context_case > 0 else 0,
        'context_dependent_rate_left_top': context_dependent_left_top / total_context_case if total_context_case > 0 else 0,
        'context_influence_score_left': np.mean(context_influence_score_left),
        'context_influence_score_left_top': np.mean(context_influence_score_left_top),

        'context_dependent_right': context_dependent_right,
        'context_dependent_right_top': context_dependent_right_top,
        'context_dependent_rate_right': context_dependent_right / total_context_case if total_context_case > 0 else 0,
        'context_dependent_rate_right_top': context_dependent_right_top / total_context_case if total_context_case > 0 else 0,
        'context_influence_score_right': np.mean(context_influence_score_right),
        'context_influence_score_right_top': np.mean(context_influence_score_right_top),

        'total_order_case': total_order_case,
        'order_dependent_case': order_dependent,
        'order_dependent_case_top': order_dependent_top,
        'order_dependent_rate': order_dependent / total_order_case if total_order_case > 0 else 0,
        'order_dependent_rate_top': order_dependent_top / total_order_case if total_order_case > 0 else 0,
        'order_influence_score': np.mean(order_influence_score),
        'order_influence_score_top': np.mean(order_influence_score_top)
    })
    print(final_results)
    save_results(f'exp3/{model_name}', f'{corpus_name}_final_results.json', final_results)


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
    max_words = 20
    truncated_sentences = []
    for sentence in cleaned_sentences:
        words = sentence.split()  # Split the sentence into words
        if len(words) > max_words:
            truncated_sentences.append(' '.join(words[:max_words]))  # Truncate and join back to a string
        else:
            truncated_sentences.append(sentence)
    return truncated_sentences[:num_sentences]


def save_results(output_dir, dataset_name, results):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, dataset_name)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Load model and tokenizer
hf_token = "hf_"
# model_name = "bert-base-uncased"
model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
model_name = "google-bert/bert-large-cased-whole-word-masking-finetuned-squad"
# model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
# model_name = "deepset/tinyroberta-squad2"
# model_name = "deepset/roberta-base-squad2"
# model_name = "distilbert/distilbert-base-uncased-distilled-squad"
# model_name = "twmkn9/albert-base-v2-squad2"

model_name = "deepset/minilm-uncased-squad2"
model_name = "deepset/deberta-v3-large-squad2"
# model_name = "csarron/mobilebert-uncased-squad-v2"
# model_name = "deepset/xlm-roberta-base-squad2"
# model_name = "deepset/bert-base-cased-squad2"
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Version: ", torch.version.cuda)
print("Current Device: ", torch.cuda.current_device())
print("Device Count: ", torch.cuda.device_count())
print("Device Name: ", torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "No CUDA Device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.model_max_length = 1000
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=hf_token, output_hidden_states=True)

    model.eval()
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
except OSError as e:
    print(f"Error loading the model: {e}")
    print("Please ensure the model name is correct and check your internet connection.")

config = model.config
hidden_size = config.hidden_size
num_attention_heads = config.num_attention_heads
d_head = hidden_size // num_attention_heads
num_layer = config.num_hidden_layers

'''
Demo code for calculating Semantic Dependency Alteration Score in A.2
'''

dataset = 'glue'  # 更换为 'gsm8k', 'Yelp', 'glue', 'cnn_dailymail', 'openOrca', 'wikitext',
# chunk size must be even
evaluate_dependence(dataset, 1100, 100, 5, 5)
