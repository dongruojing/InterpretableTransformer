import json
import random
import re

from transformers import AutoTokenizer, AutoModelForMaskedLM, \
    AutoModelForQuestionAnswering
import numpy as np
import os
import torch
from datasets import load_dataset


def generate_non_subword_tokens(random_num):
    replacements = []

    while len(replacements) < random_num:
        # 生成随机的 token ID
        random_token_id = random.randint(1, tokenizer.vocab_size - 1)

        # 将 token ID 转换为 token
        random_token = tokenizer.convert_ids_to_tokens([random_token_id])[0]

        # 对于 GPT-2 或 LLaMA，非子词以 "Ġ" 开头，确保不是子词
        if random_token.startswith("##"):
            replacements.append(random_token_id)

    return replacements


def process_sentence(sentence, random_num):
    original_inputs = tokenizer(sentence, return_tensors='pt').to(device)
    original_hidden_states = model(**original_inputs).hidden_states[-1].squeeze(0).to(
        device)  # Final layer hidden states
    original_token_ids = original_inputs['input_ids'][0].tolist()
    original_token_list = tokenizer.convert_ids_to_tokens(original_token_ids)

    case_count = 0
    successful_cases = 0
    subword_case_count = 0
    successful_subword_cases = 0
    punctuation_case_count = 0
    successful_punctuation_cases = 0
    # is_pun = False
    # is_subword = False
    exceptions = []
    f_case = 0  # influence right or left token
    var_zero_case = 0

    # Iterate over all tokens (excluding [CLS] and [SEP])
    for token_index in range(1, len(original_token_ids) - 1):

        original_token = original_token_list[token_index]
        if original_token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '<s>', '</s>', '<pad>', '<unk>']:
            continue

        if original_token.startswith("##"):
            is_subword = True
        elif token_index < len(original_token_list) - 1 and original_token_list[token_index + 1].startswith("##"):
            is_subword = True
        else:
            is_subword = False

        if original_token in ['.', ',', '!', '?', ';', ':', '"', '\'', '(', ')']:
            is_pun = True
        else:
            is_pun = False

        # Generate random replacements for each token
        replacements = [random.randint(1, tokenizer.vocab_size - 1) for _ in range(random_num)]
        # Initialize a list to store second norm deltas for each replacement
        delta_norms_list = []
        # replacements = generate_non_subword_tokens(random_num)
        for replacement_token_id in replacements:
            replaced_token_ids = original_token_ids[:]
            replaced_token_ids[token_index] = replacement_token_id

            # Convert token IDs to tensor
            replaced_inputs = {'input_ids': torch.tensor([replaced_token_ids]).to(device)}
            rep_hidden_states = model(**replaced_inputs).hidden_states[-1].squeeze(0).to(device)
            # rep_hidden_states_list.append(rep_hidden_states)

            # Calculate the second norm (Euclidean distance) for each token position
            delta_norms = torch.norm(rep_hidden_states - original_hidden_states, dim=-1).detach().cpu().numpy()
            delta_norms_list.append(delta_norms)

        delta_norms_tensor = torch.tensor(np.array(delta_norms_list)).to(device)
        variance_norms = torch.mean(delta_norms_tensor, dim=0).detach().cpu().numpy()

        # Find the index with the maximum variance norm
        max_var_idx = np.argmax(variance_norms)

        all_greater_than_zero = np.all(variance_norms > 0)

        if max_var_idx == token_index:  # If max variance is at the replaced token's position
            successful_cases += 1
            if is_subword:
                successful_subword_cases += 1
            if is_pun:
                successful_punctuation_cases += 1
        elif max_var_idx == token_index + 1 or max_var_idx == token_index - 1:
            f_case += 1
        else:
            exception_entry = {
                'sentence': sentence,
                'original_token': original_token,
                # 'replacement': tokenizer.convert_ids_to_tokens([replacements[0]])[0],
                # Using the first replacement as a representative
                'max_var_idx': int(max_var_idx),  # Index of max variance
                'variance': variance_norms.tolist(),  # Convert tensor to list
                'max_variance_token': original_token_list[max_var_idx],
                'is_pun': is_pun,
                'is_subword': is_subword
            }
            exceptions.append(exception_entry)

        case_count += 1  # Each replacement counts as a separate case

        if is_subword:
            subword_case_count += 1
        if is_pun:
            punctuation_case_count += 1
        if not all_greater_than_zero:
            var_zero_case += 1

    return exceptions, case_count, successful_cases, subword_case_count, successful_subword_cases, punctuation_case_count, successful_punctuation_cases, var_zero_case, f_case


def process_sentences_from_db(corpus_name, sen_num, random_num):
    sentences = load_corpus(corpus_name, sen_num)
    all_exceptions = []
    total_cases = 0
    successful_cases = 0
    total_subword_cases = 0
    successful_subword_cases = 0
    total_punctuation_cases = 0
    successful_punctuation_cases = 0

    for sentence in sentences:
        exceptions, case_count, successful_cases_for_sentence, subword_case_count, successful_subword_cases_for_sentence, punctuation_case_count, successful_punctuation_cases_for_sentence, var_zero, f_case = process_sentence(
            sentence, random_num)
        total_cases += case_count
        successful_cases += successful_cases_for_sentence
        total_subword_cases += subword_case_count
        successful_subword_cases += successful_subword_cases_for_sentence
        total_punctuation_cases += punctuation_case_count
        successful_punctuation_cases += successful_punctuation_cases_for_sentence
        if exceptions:
            all_exceptions.extend(exceptions)

    case_success_rate = successful_cases / total_cases if total_cases > 0 else 0
    subword_case_success_rate = successful_subword_cases / total_subword_cases if total_subword_cases > 0 else 0
    punctuation_success_rate = successful_punctuation_cases / total_punctuation_cases if total_punctuation_cases > 0 else 0
    non_subword_case_success_rate = (successful_cases - successful_subword_cases) / (
            total_cases - total_subword_cases) if (total_cases - total_subword_cases) > 0 else 0
    results = {
        'total_cases': total_cases,
        'successful_cases': successful_cases,
        'success_rate': case_success_rate,
        'total_subword_cases': total_subword_cases,
        'successful_subword_cases': successful_subword_cases,
        'subword_success_rate': subword_case_success_rate,
        'non_subword_success_rate': non_subword_case_success_rate,
        'total_punctuation_cases': total_punctuation_cases,
        'successful_punctuation_cases': successful_punctuation_cases,
        'punctuation_success_rate': punctuation_success_rate,
        'exceptions': all_exceptions
    }

    save_results(corpus_name, results)
    print(total_cases,
          successful_cases,
          case_success_rate)


def process_sentences_in_chunks(corpus_name, num_sentences, chunk_size, random_num):
    sentences = load_corpus(corpus_name, num_sentences)
    total_cases = 0
    successful_cases = 0
    total_subword_cases = 0
    successful_subword_cases = 0
    total_punctuation_cases = 0
    successful_punctuation_cases = 0
    results = []
    f_cases = 0
    var_zero_total = 0

    output_dir = f'exp1/{model_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{corpus_name}_results.json')

    for i in range(0, num_sentences, chunk_size):
        chunk = sentences[i:i + chunk_size]

        chunk_total_cases = 0
        chunk_successful_cases = 0
        chunk_subword_cases = 0
        chunk_successful_subword_cases = 0
        chunk_punctuation_cases = 0
        chunk_successful_punctuation_cases = 0
        chunk_exceptions = []

        for sentence in chunk:
            exceptions, case_count, successful_cases_for_sentence, subword_case_count, successful_subword_cases_for_sentence, punctuation_case_count, successful_punctuation_cases_for_sentence, f_case_sentence, var_zero = process_sentence(
                sentence, random_num)

            chunk_total_cases += case_count
            chunk_successful_cases += successful_cases_for_sentence
            chunk_subword_cases += subword_case_count
            chunk_successful_subword_cases += successful_subword_cases_for_sentence
            chunk_punctuation_cases += punctuation_case_count
            chunk_successful_punctuation_cases += successful_punctuation_cases_for_sentence
            var_zero_total += var_zero
            # if exceptions:
            #     chunk_exceptions.extend(exceptions)
            f_cases += f_case_sentence

        # Update totals
        total_cases += chunk_total_cases
        successful_cases += chunk_successful_cases
        total_subword_cases += chunk_subword_cases
        successful_subword_cases += chunk_successful_subword_cases
        total_punctuation_cases += chunk_punctuation_cases
        successful_punctuation_cases += chunk_successful_punctuation_cases

        # Calculate chunk success rates
        chunk_case_success_rate = chunk_successful_cases / chunk_total_cases if chunk_total_cases > 0 else 0
        chunk_subword_case_success_rate = chunk_successful_subword_cases / chunk_subword_cases if chunk_subword_cases > 0 else 0
        chunk_punctuation_success_rate = chunk_successful_punctuation_cases / chunk_punctuation_cases if chunk_punctuation_cases > 0 else 0
        chunk_non_subword_case_success_rate = (chunk_successful_cases - chunk_successful_subword_cases) / (
                chunk_total_cases - chunk_subword_cases) if (chunk_total_cases - chunk_subword_cases) > 0 else 0

        # Prepare chunk results
        chunk_results = {
            'chunk_number': i // chunk_size + 1,
            'total_cases': chunk_total_cases,
            'successful_cases': chunk_successful_cases,
            'success_rate': chunk_case_success_rate,
            'total_subword_cases': chunk_subword_cases,
            'successful_subword_cases': chunk_successful_subword_cases,
            'subword_success_rate': chunk_subword_case_success_rate,
            'non_subword_success_rate': chunk_non_subword_case_success_rate,
            'total_punctuation_cases': chunk_punctuation_cases,
            'successful_punctuation_cases': chunk_successful_punctuation_cases,
            'punctuation_success_rate': chunk_punctuation_success_rate,
            # 'exceptions': chunk_exceptions
        }

        # Append chunk results to the file
        # with open(output_file, 'a') as f:
        #     json.dump(chunk_results, f, indent=2)
        #     f.write('\n')
        results.append(chunk_results)
        print(chunk_results)
        if total_cases > 100000:
            break

    # Final summary after all chunks are processed
    final_case_success_rate = successful_cases / total_cases if total_cases > 0 else 0
    final_subword_case_success_rate = successful_subword_cases / total_subword_cases if total_subword_cases > 0 else 0
    final_punctuation_success_rate = successful_punctuation_cases / total_punctuation_cases if total_punctuation_cases > 0 else 0
    final_non_subword_case_success_rate = (successful_cases - successful_subword_cases) / (
            total_cases - total_subword_cases) if (total_cases - total_subword_cases) > 0 else 0
    final_nonsub_nonpun_success_rate = (successful_cases - successful_subword_cases - successful_punctuation_cases) / (
                total_cases - total_subword_cases - total_punctuation_cases) if (
                                                                                            total_cases - total_subword_cases - total_punctuation_cases) > 0 else 0
    f_rate = (f_cases + successful_cases) / total_cases if total_cases > 0 else 0
    var_greater_than_zero = 1 - (var_zero_total / total_cases) if total_cases > 0 else 0
    final_results = {
        'final_summary': True,
        'total_cases': total_cases,
        'successful_cases': successful_cases,
        'success_rate': final_case_success_rate,
        'total_subword_cases': total_subword_cases,
        'successful_subword_cases': successful_subword_cases,
        'subword_success_rate': final_subword_case_success_rate,
        'non_subword_success_rate': final_non_subword_case_success_rate,
        'total_punctuation_cases': total_punctuation_cases,
        'successful_punctuation_cases': successful_punctuation_cases,
        'punctuation_success_rate': final_punctuation_success_rate,
        'final_nonsub_nonpun_success_rate': final_nonsub_nonpun_success_rate,
        'f-successful-cases': f_cases,
        'f-rate': f_rate,
        'var_greater_than_zero': var_greater_than_zero
    }
    results.append(final_results)
    # Append final results to the file
    with open(output_file, 'a') as f:
        json.dump(results, f, indent=2)
        f.write('\n')
    print(final_results)


def save_results(dataset_name, results):
    output_dir = f'exp1/{model_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, dataset_name + '_test.json')
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
        # 'mmlu': ("cais/mmlu", "abstract_algebra", 'input'),
        # 'magicoder': ("magicoder", "magicoder-code-to-text", 'code'),
        'Yelp': ("yelp_review_full", None, 'text'),
        # 'glue': ("glue", "sst2", 'sentence'),
        'glue': ("glue", "mnli", 'hypothesis'),
        'cnn_dailymail': ("cnn_dailymail", "3.0.0", 'article'),
        'openOrca': ("Open-Orca/OpenOrca", None, 'question'),
        'wikitext': ("wikitext", "wikitext-2-v1", 'text'),
        # 'opensubtitles': ("opus_books", "en", 'translation'),
        # 'bookcorpus': ("bookcorpusopen", None, 'text'),
        # 'bnc': ("bnc_text", None, 'text'),
        # 'tatoeba': ("mteb/tatoeba-bitext-mining", "default", 'sentence1'),
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
    max_words = 300
    truncated_sentences = []
    for sentence in cleaned_sentences:
        words = sentence.split()  # Split the sentence into words
        if len(words) > max_words:
            truncated_sentences.append(' '.join(words[:max_words]))  # Truncate and join back to a string
        else:
            truncated_sentences.append(sentence)
            # Return the first num_sentences cleaned sentences
    return truncated_sentences[:num_sentences]


os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Load model and tokenizer
hf_token = "hf_"
# model_name = "bert-base-uncased"
# model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
model_name = "google-bert/bert-large-cased-whole-word-masking-finetuned-squad"
# model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
# model_name = "deepset/tinyroberta-squad2"
# model_name = "deepset/roberta-base-squad2"
# model_name = "distilbert/distilbert-base-uncased-distilled-squad"
# model_name = "twmkn9/albert-base-v2-squad2"

model_name = "deepset/minilm-uncased-squad2"
# model_name = "deepset/deberta-v3-large-squad2"
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
    # model = AutoModelForMaskedLM.from_pretrained(model_name, use_auth_token=hf_token, output_attentions=True, output_hidden_states=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name,
                                                          output_hidden_states=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True,
    #                                              output_hidden_states=True)
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

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

'''
Demo code for evaluating whether most tokens primarily retain their original semantic information across Transformer layers (Section 3)
'''

corpus_name = 'glue'  # 'gsm8k', 'Yelp', 'glue', 'cnn_dailymail', 'openOrca', 'wikitext',

for corpus_name in ['Yelp', 'glue', 'cnn_dailymail', 'openOrca', 'wikitext']:
    process_sentences_in_chunks(corpus_name, 10000, 500, 5)
# process_sentences_in_chunks(corpus_name, 10000, 500, 5)
