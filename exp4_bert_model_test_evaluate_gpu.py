import random
import unicodedata
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForMaskedLM
from datasets import load_dataset
import os
import numpy as np
from collections import OrderedDict
import argparse
import collections
import json
import re
import string
import sys

import torch
from transformers import BertModel, BertConfig


# SQUAD test from bert single test code

# SQuAD evaluation functions (imported from the official script)
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_special_characters(text):
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    def unicode_lower(text):
        text = re.sub(r'[\u2013-\u2014]', ' ', text)
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()

    if not s or not s.strip():
        return ''
    return white_space_fix(remove_articles(remove_special_characters((lower(unicode_lower(s))))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    if not a_gold and a_pred == "":
        return 1
    if isinstance(a_gold, list) and len(a_gold) == 0 and a_pred == "":
        return 1
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    if not a_gold and a_pred == "":
        return 1.0
    if isinstance(a_gold, list) and len(a_gold) == 0 and a_pred == "":
        return 1.0
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}

    for example in dataset:
        qid = example['id']
        context = example['context']
        question = example['question']
        gold_answers = example['answers']['text']  # 获取所有正确答案
        if not gold_answers:
            gold_answers = ['']
        if qid not in preds:
            print(f'Missing prediction for {qid}')
            continue

        a_pred = preds[qid]

        # 对所有正确答案计算 exact 和 f1 分数
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)

    return exact_scores, f1_scores


def make_eval_dict(exact_scores, f1_scores):
    total = len(exact_scores)
    return OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])


def generate_predictions(dataset, f1_threshold=0.6):
    predictions = {}
    mismatch_results = {"qas": []}
    mismatch_ids = []
    for idx, example in enumerate(dataset):
        data_id = example["id"]
        context = example["context"]
        question = example["question"]
        expected_answers = example["answers"]["text"]
        expected_answer_starts = example['answers']['answer_start']

        inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
        )
        predictions[example['id']] = answer

        # Check for mismatch and log results
        if expected_answers == [] and answer == "":
            is_correct = True
        else:
            is_correct = any(compute_exact(exp_answer, answer) for exp_answer in expected_answers)
        f1_score = max(compute_f1(exp_answer, answer) for exp_answer in expected_answers)

        if expected_answers == [] or answer == "" or answer in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            continue
        elif not is_correct and f1_score < f1_threshold:
            # if not is_correct:
            mismatch_results["qas"].append({
                "index": idx,
                "data_id": data_id,
                "question": question,
                "context": context,
                "expected_answers": expected_answers,
                "expected_answer_starts": expected_answer_starts,
                "model_answer": answer,
                "correct": is_correct,
                "f1": f1_score
            })
            mismatch_ids.append(data_id)

    return predictions, mismatch_results, mismatch_ids


def generate_predictions_var(dataset, f1_threshold=0.6, random_num=5):
    predictions = {}
    mismatch_ids = []
    mismatch_results = {"qas": []}
    case_count = 0
    total_cases = 0

    for idx, example in enumerate(dataset):
        data_id = example["id"]
        context = example["context"]
        question = example["question"]
        expected_answers = example["answers"]["text"]
        expected_answer_starts = example['answers']['answer_start']

        inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Get model's predicted answer
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
        )
        predictions[example['id']] = answer

        # Check for mismatch and log results
        if expected_answers == [] and answer == "":
            is_correct = True
        else:
            is_correct = any(compute_exact(exp_answer, answer) for exp_answer in expected_answers)
        f1_score = max(compute_f1(exp_answer, answer) for exp_answer in expected_answers)

        if expected_answers == [] or answer == "" or answer in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            # if not is_correct:
            #     mismatch_results["qas"].append({
            #         "index": idx,
            #         "question": question,
            #         "context": context,
            #         "expected_answers": expected_answers,
            #         "model_answer": answer,
            #         "correct": is_correct,
            #         "f1": f1_score,
            #         "dp_case": None
            #     })
            continue
        elif not is_correct and f1_score < f1_threshold:

            # Token positions for question tokens
            question_length = len(tokenizer.encode(question, add_special_tokens=False))
            question_token_positions = list(range(1, question_length + 1))  # no cls and sep

            # question_token_positions = [i for i, token_type_id in enumerate(inputs['token_type_ids'][0]) if
            #                             token_type_id == 0] # with cls and sep

            # correct_answer = min(expected_answers, key=len)
            # correct_answer_token_ids = tokenizer.encode(correct_answer, add_special_tokens=False)
            # input_ids = inputs['input_ids'][0].tolist()
            # correct_starts = []
            # correct_end = None
            # for i in range(question_length + 1, len(input_ids) - len(correct_answer_token_ids) + 1):
            #     if input_ids[i:i + len(correct_answer_token_ids)] == correct_answer_token_ids:
            #         correct_starts.append(i)
            #
            # if not correct_starts:
            #     continue
            #
            # correct_vars = []
            # for correct_start in correct_starts:
            #     correct_end = correct_start + len(correct_answer_token_ids)
            #     correct_token_positions = list(range(correct_start, correct_end))

            # Extract correct answer
            input_ids = inputs['input_ids'][0].tolist()
            replacement = [random.randint(1, tokenizer.vocab_size - 1) for _ in range(random_num)]
            correct_answer = min(expected_answers, key=len)
            correct_answer_start = expected_answer_starts[expected_answers.index(correct_answer)]
            answer_text = context[correct_answer_start:correct_answer_start + len(correct_answer)]
            correct_start = inputs.char_to_token(0, correct_answer_start, sequence_index=1)  # 1表示在context部分找token
            correct_end = inputs.char_to_token(0, correct_answer_start + len(correct_answer) - 1, sequence_index=1) + 1
            correct_token_positions = list(range(correct_start, correct_end))

            correct_answer_token_ids = [input_ids[pos] for pos in correct_token_positions]
            # Find and exclude the matching token IDs in the question
            correct_question_token_positions = [
                pos for pos in question_token_positions
                if input_ids[pos] not in correct_answer_token_ids
            ]
            if not len(correct_question_token_positions):
                continue
            # 调用 get_variance 计算 variance
            correct_var_list, correct_var, correct_flag, correct_replacement_flag, correct_t2t = get_variance(
                input_ids, correct_question_token_positions,
                correct_token_positions, replacement, random_num
            )

            # Token positions for wrong answer tokens
            wrong_token_positions = list(range(answer_start, answer_end))
            wrong_answer_token_ids = [input_ids[pos] for pos in wrong_token_positions]
            # Find and exclude the matching token IDs in the question
            wrong_question_token_positions = [
                pos for pos in question_token_positions
                if input_ids[pos] not in wrong_answer_token_ids
            ]
            if not len(wrong_question_token_positions):
                continue
            # Measure dependence for wrong answer tokens
            wrong_var_list, wrong_var, wrong_flag, wrong_replacement_flag, wrong_t2t = get_variance(input_ids,
                                                                                                    wrong_question_token_positions,
                                                                                                    wrong_token_positions,
                                                                                                    replacement,
                                                                                                    random_num)
            if correct_t2t["original token"] == wrong_t2t["original token"]:
                # print(data_id)
                continue
            if not wrong_var or not correct_var:
                continue
            # Compare and update case count
            if wrong_var > correct_var:
                case_count += 1
                dp_case = True
            else:
                dp_case = False
            total_cases += 1
            mismatch_ids.append(data_id)
            mismatch_results["qas"].append({
                "index": idx,
                "data_id": data_id,
                "question": question,
                "context": context,
                "expected_answers": expected_answers,
                "expected_answer_starts": expected_answer_starts,
                "model_answer": answer,
                "correct": is_correct,
                "f1": f1_score,
                "dp_case": {
                    "wrong_var": float(wrong_var),
                    "correct_var": float(correct_var),
                    "dp": dp_case,
                    "wrong_t2t_list": wrong_t2t,
                    "correct_t2t_list": correct_t2t,
                    # random的例子里有导致模型输出改变的个数
                    "wrong_ans_change": [wrong_flag, wrong_replacement_flag],
                    "correct_ans_change": [correct_flag, correct_replacement_flag]
                }
            })

    # Calculate case probability
    case_probability = case_count / total_cases if total_cases > 0 else 0

    return predictions, mismatch_results, case_probability, total_cases, case_count, mismatch_ids


def get_variance(input_ids, question_token_positions, answer_token_positions, replacement, random_num=5):
    original_inputs = {'input_ids': torch.tensor([input_ids]).to(device)}
    with torch.no_grad():
        original_outputs = model(**original_inputs)
    ori_answer_s = torch.argmax(original_outputs.start_logits)
    ori_answer_e = torch.argmax(original_outputs.end_logits) + 1
    original_answer_token_positions = list(range(ori_answer_s, ori_answer_e))
    original_hidden_states = original_outputs.hidden_states[-1].squeeze(0).to(device)

    # Gather hidden states for question tokens
    original_question_hidden_states = original_hidden_states[question_token_positions]

    var_list = []
    top_variances = []
    top_positions = []
    top_tokens = []
    flag = 0
    replacement_flag = 0
    t2t_list = []
    for token_index in answer_token_positions:
        # rep_hidden_states_list = []
        original_token = tokenizer.convert_ids_to_tokens(input_ids[token_index])
        # rep_hidden_states_list.append(original_question_hidden_states)
        if original_token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            continue
        if original_token.startswith("##"):
            continue
        if original_token in ['.', ',', '!', '?', ';', ':', '"', '\'', '(', ')']:
            continue
        # Generate replacements
        # replacements = [random.randint(1, tokenizer.vocab_size - 1) for _ in range(random_num)]
        replacements = replacement
        delta_norms_list = []
        for replacement_token_id in replacements:
            replaced_token_ids = input_ids.copy()
            replaced_token_ids[token_index] = replacement_token_id

            # Run the model with replaced token
            replaced_inputs = {'input_ids': torch.tensor([replaced_token_ids]).to(device)}
            with torch.no_grad():
                outputs = model(**replaced_inputs)
            rep_hidden_states = outputs.hidden_states[-1].squeeze(0).to(device)

            # Extract hidden states for question tokens
            # rep_hidden_states_list.append(rep_hidden_states[question_token_positions])
            rep_question_hidden_states = rep_hidden_states[question_token_positions]
            delta_norms = torch.norm(rep_question_hidden_states - original_question_hidden_states,
                                     dim=-1).detach().cpu().numpy()
            delta_norms_list.append(delta_norms)

            # model output
            answer_s = torch.argmax(outputs.start_logits)
            answer_e = torch.argmax(outputs.end_logits) + 1
            rep_ans_token_positions = list(range(answer_s, answer_e))
            replacement_flag += 1
            if not rep_ans_token_positions == original_answer_token_positions:
                flag += 1
            # ans = tokenizer.convert_tokens_to_string(
            #     tokenizer.convert_ids_to_tokens(replaced_inputs['input_ids'][0][answer_s:answer_e])
            # )
            # print(tokenizer.convert_tokens_to_string(
            #     tokenizer.convert_ids_to_tokens(replaced_inputs['input_ids'][0])))
            # print(ans)

        # Calculate variance
        delta_norms_tensor = torch.tensor(np.array(delta_norms_list)).to(device)
        variance_norms = torch.mean(delta_norms_tensor, dim=0).detach().cpu().numpy()
        # rep_hidden_states_tensor = torch.stack(rep_hidden_states_list, dim=0)
        # variance = torch.var(rep_hidden_states_tensor, dim=0)
        # variance_norms = torch.norm(variance, dim=-1).detach().numpy()
        var_list.append(variance_norms)
        # var_list.append(np.mean(variance_norms))

        top_3_indices = np.argsort(variance_norms)[-1:]
        top_3_variances = variance_norms[top_3_indices]
        top_mean_var = np.mean(top_3_variances)
        top_3_positions = [question_token_positions[idx] for idx in top_3_indices]
        top_3_tokens = [tokenizer.convert_ids_to_tokens([input_ids[pos]])[0] for pos in top_3_positions]
        top_variances.append(top_mean_var)
        top_positions.append(top_3_positions)
        top_tokens.append(top_3_tokens)
        t2t = {"original token": original_token, "question token": top_3_tokens, "var": float(top_mean_var)}
        t2t_list.append(t2t)

    var_list = np.array(var_list)  # Convert list to numpy array for easier manipulation
    # var_list_mean = np.mean(var_list, axis=1)  # Mean variance for each answer token
    # var = np.mean(var_list_mean)
    if top_variances:
        var = np.max(top_variances)
        max_t2t = max(t2t_list, key=lambda x: x["var"])
    else:
        var = 0
        max_t2t = {"original token": "None", "question token": "None", "var": 0}
    return var_list, var, flag, replacement_flag, max_t2t


def filter_contexts(data, max_context_length=200):
    filtered_data = [example for example in data if len(example["context"]) < max_context_length]
    return filtered_data


def filter_id(data, id_list):
    filtered_data = [example for example in data if example["id"] in id_list]
    return filtered_data


# only test models and get mismatch example list
def evaluate_dataset_in_chunks(dataset, chunk_size=100):
    total_examples = len(dataset)
    all_chunk_results = {'exact_scores': {}, 'f1_scores': {}}
    chunk_idx = 0
    mismatch_id_list = []
    for start_idx in range(0, total_examples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_examples)
        chunk = dataset[start_idx:end_idx]

        # Generate predictions for the chunk
        predictions, mismatch_results, chunk_mismatch_ids = generate_predictions(chunk, 0.6)

        # Evaluate the predictions for the chunk
        exact_scores, f1_scores = get_raw_scores(chunk, predictions)

        chunk_ = {'exact_scores': {}, 'f1_scores': {}, 'mismatch_results': {"qas": []}}
        chunk_['exact_scores'].update(exact_scores)
        chunk_['f1_scores'].update(f1_scores)
        chunk_['mismatch_results']["qas"].extend(mismatch_results["qas"])
        mismatch_id_list.extend(chunk_mismatch_ids)
        # Update chunk results
        all_chunk_results['exact_scores'].update(exact_scores)
        all_chunk_results['f1_scores'].update(f1_scores)
        # all_chunk_results['mismatch_results']["qas"].extend(mismatch_results["qas"])

        # Save results incrementally
        chunk_eval_results = make_eval_dict(chunk_['exact_scores'], chunk_['f1_scores'])
        chunk_results = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "eval_results": chunk_eval_results,
            "mismatch_results": chunk_['mismatch_results']
        }
        print(f'chunk:{chunk_idx}')
        print(chunk_eval_results)
        save_results(f'exp4/{model_name}/model_test_{dataset_select}', f"squad_eval_results_chunk_{chunk_idx}.json",
                     chunk_results)
        chunk_idx += 1

    # Final results
    final_eval_results = make_eval_dict(all_chunk_results['exact_scores'], all_chunk_results['f1_scores'])
    final_results = {
        "final_eval_results": final_eval_results,
    }
    save_results(f'exp4/{model_name}/model_test_{dataset_select}', f"mismatch_ids.json", mismatch_id_list)
    save_results(f'exp4/{model_name}/model_test_{dataset_select}', f"squad_eval_results_final.json", final_results)


def process_dataset_in_chunks(dataset, chunk_size=100):
    total_examples = len(dataset)
    all_chunk_results = {'exact_scores': {}, 'f1_scores': {}}
    chunk_idx = 0
    dp_case_count = 0
    dp_total = 0
    mismatch_id_list = []
    for start_idx in range(0, total_examples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_examples)
        chunk = dataset[start_idx:end_idx]

        # Generate predictions for the chunk
        # predictions, mismatch_results = generate_predictions(chunk)
        predictions, mismatch_results, chunk_dp_probability, chunk_dp_total_cases, chunk_dp_case_count, chunk_mismatch_ids = generate_predictions_var(
            chunk, 1.0, 5)
        dp_case_count += chunk_dp_case_count
        dp_total += chunk_dp_total_cases

        # Evaluate the predictions for the chunk
        exact_scores, f1_scores = get_raw_scores(chunk, predictions)

        chunk_ = {'exact_scores': {}, 'f1_scores': {}, 'mismatch_results': {"qas": []}}
        chunk_['exact_scores'].update(exact_scores)
        chunk_['f1_scores'].update(f1_scores)
        chunk_['mismatch_results']["qas"].extend(mismatch_results["qas"])
        mismatch_id_list.extend(chunk_mismatch_ids)
        # Update chunk results
        all_chunk_results['exact_scores'].update(exact_scores)
        all_chunk_results['f1_scores'].update(f1_scores)
        # all_chunk_results['mismatch_results']["qas"].extend(mismatch_results["qas"])

        # Save results incrementally
        chunk_eval_results = make_eval_dict(chunk_['exact_scores'], chunk_['f1_scores'])
        chunk_dp_result = {"probability": chunk_dp_probability, "dp_total_case": chunk_dp_total_cases,
                           "dp_case_count": chunk_dp_case_count}
        chunk_results = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "eval_results": chunk_eval_results,
            "dp_results": chunk_dp_result,
            "mismatch_results": chunk_['mismatch_results']
        }
        print(f'chunk:{chunk_idx}')
        print(chunk_eval_results)
        print(chunk_dp_result)
        save_results(f'exp4/{model_name}/mismatched_test_{dataset_select}',
                     f"squad_eval_results_chunk_{chunk_idx}.json", chunk_results)
        chunk_idx += 1

    dp_probability = dp_case_count / dp_total if dp_total > 0 else 0
    # Final results
    final_eval_results = make_eval_dict(all_chunk_results['exact_scores'], all_chunk_results['f1_scores'])
    final_results = {
        "final_eval_results": final_eval_results,
        "dp_results": {"probability": dp_probability, "dp_total_case": dp_total,
                       "dp_case_count": dp_case_count},
    }
    save_results(f'exp4/{model_name}/mismatched_test_{dataset_select}', f"mismatch_ids.json", mismatch_id_list)
    save_results(f'exp4/{model_name}/mismatched_test_{dataset_select}', f"squad_eval_results_final.json", final_results)


def save_results(output_dir, dataset_name, results):
    # output_dir = f'exp4/{model_name}/test3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, dataset_name)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    return filepath


# Load the model and tokenizer
hf_token = "hf_"
model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
model_name = "google-bert/bert-large-cased-whole-word-masking-finetuned-squad"
# model_name = "bert-base-uncased"
# model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
model_name = "deepset/tinyroberta-squad2"
model_name = "deepset/roberta-base-squad2"
# model_name = "distilbert/distilbert-base-uncased-distilled-squad"
model_name = "twmkn9/albert-base-v2-squad2"

# model_name = "deepset/minilm-uncased-squad2"
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.model_max_length = 1000
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, output_attentions=True,
                                                          output_hidden_states=True).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
except OSError as e:
    print(f"Error loading the model: {e}")
    print("Please ensure the model name is correct and check your internet connection.")

'''
Demo code for evaluating Semantic dependencies in QA Task in Section 5.1
'''

# process all examples
dataset_select = "validation"  # "train" "validation"
squad = load_dataset("squad")
filtered_squad = filter_contexts(squad[dataset_select], max_context_length=1500)  # 1500
# filtered_squad = [example for example in squad["validation"]]
evaluate_dataset_in_chunks(filtered_squad, chunk_size=300)

# process mismatched examples
file_path = f'exp4/{model_name}/model_test_{dataset_select}/mismatch_ids.json'
with open(file_path, 'r') as file:
    total_mismatch_id_list = json.load(file)

# squad = load_dataset("squad")
input_data = filter_id(squad[dataset_select], total_mismatch_id_list)
print(total_mismatch_id_list)

process_dataset_in_chunks(input_data, 300)
