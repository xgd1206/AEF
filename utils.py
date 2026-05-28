import os
import random
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import TrainerCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
import torch as th
from tqdm import tqdm
from evaluation import quadratic_weighted_kappa
import warnings
import re
from collections import OrderedDict

warnings.filterwarnings("ignore")

ASAP_TRAIT_NAMES = [
    "overall",
    "content",
    "organization",
    "word choice",
    "sentence fluency",
    "conventions",
    "prompt adherence",
    "language",
    "narrativity",
    "style",
    "voice",
]

FEEDBACK_TRAIT_NAMES = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]

FEEDBACK_SCORE_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=np.float32)


def extract_traits(text):
    matches = re.findall(r'(\w+)\s+([\d.]+)', text)
    traits_dict = {trait: float(value.rstrip('.')) for trait, value in matches[:7]} 
    return traits_dict


trait_map = {
    1: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
    2: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
    3: ["overall", "content", "prompt adherence", "language", "narrativity"],
    4: ["overall", "content", "prompt adherence", "language", "narrativity"],
    5: ["overall", "content", "prompt adherence", "language", "narrativity"],
    6: ["overall", "content", "prompt adherence", "language", "narrativity"],
    7: ["overall", "content", "organization", "style", "conventions"],
    8: ["overall", "content", "organization", "voice", "word choice", "sentence fluency", "conventions"]
}


def parse_traits(text, tokenizer, exclude_keys=[]):
    text = re.sub(r'(\[\w+(?:\s\w+)?\])', r'\n\1', text)
    parts = text.split('\n')
    
    traits_dict = OrderedDict()
    current_trait = None
    for part in parts:
        match = re.match(r'\[(\w+(?:\s\w+)?)\]', part)
        if match:
            current_trait = match.group(1).strip()
            traits_dict[current_trait] =  part.strip() + " "
        elif current_trait:
            traits_dict[current_trait] += part.strip() + " "
    
    for trait in traits_dict:
        traits_dict[trait] = traits_dict[trait].strip()

    filtered_dict = OrderedDict()
    for key, value in traits_dict.items():
        if key.lower() in exclude_keys:
            tokens = tokenizer.tokenize(value)
            filtered_dict['<pad>'] = ' '.join(['<pad>'] * len(tokens))
        else:
            filtered_dict[key] = value
    
    return " ".join(filtered_dict.values())


def transform_input(input_str):
    return ", ".join([f"[{match.group(1)}] {match.group(2)}" 
                      for match in re.finditer(r"(\w+ \w+|\w+) (\w+)", input_str)])


def parse_asap_traits_text(text):
    normalized_text = text.lower()
    pattern = re.compile(
        r"(overall|content|organization|word[-\s]choice|sentence[-\s]fluency|conventions|prompt[-\s]adherence|language|narrativity|style|voice)\s*[: ]\s*(nan|-?\d+(?:\.\d+)?)"
    )
    traits = {}
    for key, value in pattern.findall(normalized_text):
        key = key.replace("-", " ")
        if value == "nan":
            traits[key] = np.nan
        else:
            traits[key] = int(round(float(value)))
    return traits


def get_trait_names(args):
    return ASAP_TRAIT_NAMES if args.data == "asap" else FEEDBACK_TRAIT_NAMES


def get_sample_limit(args, split_name):
    return getattr(args, f"max_{split_name}_samples", None)


def maybe_limit_dataset(dataset, limit):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def prepare_score_metadata(train_data, dev_data, test_data, args):
    trait_names = get_trait_names(args)
    trait_min = []
    trait_max = []
    prompt_trait_stats = {}
    
    for trait in trait_names:
        values = []
        for dataset in [train_data, dev_data, test_data]:
            if trait in dataset.column_names:
                array = np.array(dataset[trait], dtype=np.float32)
                array = array[~np.isnan(array)]
                if len(array) > 0:
                    values.append(array)
        if values:
            merged = np.concatenate(values)
            trait_min.append(float(np.min(merged)))
            trait_max.append(float(np.max(merged)))
        else:
            trait_min.append(0.0)
            trait_max.append(1.0)
    
    args.trait_names = trait_names
    args.trait_min = trait_min
    args.trait_max = trait_max
    args.score_step = 1.0 if args.data == "asap" else 0.5
    
    if args.data == "asap":
        prompt_ids = sorted(set(int(x) for x in train_data["essay_set"]))
    else:
        prompt_ids = [0]
    
    for prompt_id in prompt_ids:
        prompt_trait_stats[prompt_id] = {}
        if args.data == "asap":
            prompt_mask = np.array(train_data["essay_set"]) == prompt_id
        for trait in trait_names:
            if args.data == "asap":
                raw_values = np.array(train_data[trait], dtype=np.float32)[prompt_mask]
            else:
                raw_values = np.array(train_data[trait], dtype=np.float32)
            raw_values = raw_values[~np.isnan(raw_values)]
            if len(raw_values) == 0:
                mean_value = 0.0
                std_value = 1.0
            else:
                mean_value = float(np.mean(raw_values))
                std_value = float(np.std(raw_values))
                if std_value < 1e-4:
                    std_value = 1.0
            prompt_trait_stats[prompt_id][trait] = {
                "mean": mean_value,
                "std": std_value,
            }
    args.prompt_trait_stats = prompt_trait_stats


def normalize_trait_value(prompt_id, trait, value, args):
    stats = args.prompt_trait_stats[int(prompt_id)][trait]
    return (float(value) - stats["mean"]) / stats["std"]


def denormalize_trait_predictions(predictions, prompt_ids, args):
    predictions = np.array(predictions, dtype=np.float32)
    denorm_predictions = np.zeros_like(predictions)
    for row_idx, prompt_id in enumerate(prompt_ids):
        stats = args.prompt_trait_stats[int(prompt_id)]
        for trait_idx, trait in enumerate(args.trait_names):
            denorm_predictions[row_idx, trait_idx] = (
                predictions[row_idx, trait_idx] * stats[trait]["std"] + stats[trait]["mean"]
            )
    return denorm_predictions


def quantize_trait_predictions(predictions, args):
    predictions = np.array(predictions, dtype=np.float32)
    for idx in range(predictions.shape[1]):
        predictions[:, idx] = np.clip(predictions[:, idx], args.trait_min[idx], args.trait_max[idx])
    if args.data == "feedback":
        score_grid = FEEDBACK_SCORE_VALUES.reshape(1, 1, -1)
        distances = np.abs(predictions[:, :, None] - score_grid)
        nearest_idx = distances.argmin(axis=-1)
        return FEEDBACK_SCORE_VALUES[nearest_idx]
    return np.rint(predictions).astype(np.float32)


def preprocess_data(examples, tokenizer, args):
    essay = tokenizer(
        ["<essay> " + example for example in examples["t5_input"]],
        max_length=args.max_essay_length,
        truncation=True,
        padding="max_length",
    )
    essay_input_ids = [token_ids[:] for token_ids in essay["input_ids"]]
    essay_attention_mask = [mask[:] for mask in essay["attention_mask"]]

    if getattr(args, "use_adaptive_rmts", False):
        gpt_criteria = tokenizer(
            [" <rationale> " + example for example in examples["gpt_criteria"]],
            max_length=args.max_rationale_length,
            truncation=True,
            padding="max_length",
        )
        llama_criteria = tokenizer(
            [" <rationale> " + example for example in examples["llama_criteria"]],
            max_length=args.max_rationale_length,
            truncation=True,
            padding="max_length",
        )
        primary_criteria = gpt_criteria
    else:
        if getattr(args, "concat_rationales", False):
            merged_rationales = [
                f"{gpt_text} {llama_text}"
                for gpt_text, llama_text in zip(examples["gpt_criteria"], examples["llama_criteria"])
            ]
            primary_criteria = tokenizer(
                [" <rationale> " + example for example in merged_rationales],
                max_length=args.max_rationale_length,
                truncation=True,
                padding="max_length",
            )
        elif args.llm == "gpt":
            primary_criteria = tokenizer(
                [" <rationale> " + example for example in examples["gpt_criteria"]],
                max_length=args.max_rationale_length,
                truncation=True,
                padding="max_length",
            )
        else:
            primary_criteria = tokenizer(
                [" <rationale> " + example for example in examples["llama_criteria"]],
                max_length=args.max_rationale_length,
                truncation=True,
                padding="max_length",
            )

    essay["input_ids"] = [sublist1 + sublist2 for sublist1, sublist2 in zip(essay["input_ids"], primary_criteria["input_ids"])]
    essay["attention_mask"] = [sublist1 + sublist2 for sublist1, sublist2 in zip(essay["attention_mask"], primary_criteria["attention_mask"])]

    with tokenizer.as_target_tokenizer():
        labels = examples["t5_output"]
        
        if args.data == "asap":
            if "t5" in args.model_name:
                labels = tokenizer(labels, max_length=64, truncation=True, padding="max_length")
            else:
                labels = tokenizer(labels, max_length=256, truncation=True, padding="max_length")
        else:
            if "flan-t5-base" in args.model_name:
                labels = tokenizer(labels, max_length=256, truncation=True, padding="max_length")
            else:
                labels = tokenizer(labels, max_length=64, truncation=True, padding="max_length")

    essay["labels"] = labels["input_ids"]

    if getattr(args, "use_adaptive_rmts", False):
        trait_labels = []
        trait_mask = []
        for row_idx in range(len(examples["t5_output"])):
            row_labels = []
            row_mask = []
            for trait in args.trait_names:
                value = examples[trait][row_idx] if trait in examples else np.nan
                if pd.isna(value):
                    row_labels.append(0.0)
                    row_mask.append(0.0)
                else:
                    if args.data == "asap":
                        prompt_id = int(examples["essay_set"][row_idx])
                    else:
                        prompt_id = 0
                    row_labels.append(normalize_trait_value(prompt_id, trait, value, args))
                    row_mask.append(1.0)
            trait_labels.append(row_labels)
            trait_mask.append(row_mask)

        if args.data == "asap":
            prompt_ids = [int(prompt) for prompt in examples["essay_set"]]
        else:
            prompt_ids = [0 for _ in range(len(examples["t5_output"]))]

        essay["essay_input_ids"] = essay_input_ids
        essay["essay_attention_mask"] = essay_attention_mask
        essay["gpt_input_ids"] = gpt_criteria["input_ids"]
        essay["gpt_attention_mask"] = gpt_criteria["attention_mask"]
        essay["llama_input_ids"] = llama_criteria["input_ids"]
        essay["llama_attention_mask"] = llama_criteria["attention_mask"]
        essay["prompt_ids"] = prompt_ids
        essay["trait_labels"] = trait_labels
        essay["trait_mask"] = trait_mask
    
    return essay


def read_data(data_path):
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df)
    return dataset


def set_seed(args):
    """
    Ensure reproducibility by setting the seed for random number generation.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    if th.cuda.is_available():
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def train(model, tokenizer, train_dataset, dev_dataset, args=None):
    if args.data == "asap":
        if args.train_epochs == 1:
            eval_steps = 500
        else:
            eval_steps = int(np.ceil(5000/(args.train_batch_size/4)))
    else:
        eval_steps = 1600
        
    print("Size of eval_steps: ", eval_steps)
    
    save_steps = eval_steps
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{args.result_path}",
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        num_train_epochs=args.train_epochs,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=15,
        save_safetensors=False,
        learning_rate=args.learning_rate,
        ddp_find_unused_parameters=None,
        dataloader_drop_last=False,
        group_by_length=False,
        length_column_name="length",
        report_to="none",
        logging_steps=100,
        remove_unused_columns=True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), SaveTopModelsCallback(args.save_model_fold_path)]
    )

    trainer.train()
    return model


def advanced_asap_test(model, test_data, args, trait_map):
    pred_dic = {prompt: {trait: [] for trait in traits} for prompt, traits in trait_map.items()}
    true_dic = {prompt: {trait: [] for trait in traits} for prompt, traits in trait_map.items()}
    qwk_result = {prompt: {trait: 0.0 for trait in traits} for prompt, traits in trait_map.items()}
    trait_to_idx = {trait: idx for idx, trait in enumerate(args.trait_names)}
    batch_size = args.test_batch_size

    model.eval()
    with th.no_grad():
        for i in range(0, len(test_data), batch_size):
            test = test_data[i:i + batch_size]
            predictions, _ = model.predict_trait_scores(
                essay_input_ids=th.tensor(test["essay_input_ids"]).to(args.device),
                essay_attention_mask=th.tensor(test["essay_attention_mask"]).to(args.device),
                gpt_input_ids=th.tensor(test["gpt_input_ids"]).to(args.device),
                gpt_attention_mask=th.tensor(test["gpt_attention_mask"]).to(args.device),
                llama_input_ids=th.tensor(test["llama_input_ids"]).to(args.device),
                llama_attention_mask=th.tensor(test["llama_attention_mask"]).to(args.device),
                prompt_ids=th.tensor(test["prompt_ids"]).to(args.device),
            )
            prompts = test["essay_set"]
            predictions = denormalize_trait_predictions(
                predictions.detach().cpu().numpy(),
                prompts,
                args,
            )
            predictions = quantize_trait_predictions(predictions, args)
            for row_idx, prompt in enumerate(prompts):
                for trait in trait_map[int(prompt)]:
                    trait_idx = trait_to_idx[trait]
                    true_value = test[trait][row_idx]
                    if pd.isna(true_value):
                        continue
                    pred_dic[int(prompt)][trait].append(int(predictions[row_idx, trait_idx]))
                    true_dic[int(prompt)][trait].append(int(true_value))

    for prompt in range(1, 9):
        for trait in trait_map[prompt]:
            if pred_dic[prompt][trait] and true_dic[prompt][trait]:
                qwk_result[prompt][trait] = quadratic_weighted_kappa(
                    np.array(pred_dic[prompt][trait]),
                    np.array(true_dic[prompt][trait]),
                )
    log = "Test Result"
    for prompt in range(1, 9):
        log += f"\n\n| Prompt: {prompt} |"
        log += f"\n| {qwk_result[prompt]} |"
    print(log)
    return qwk_result, pred_dic, true_dic


def advanced_feedback_test(model, test_data, args, trait_list):
    pred_dic = {trait: [] for trait in trait_list}
    true_dic = {trait: [] for trait in trait_list}
    qwk_result = {trait: 0.0 for trait in trait_list}
    trait_to_idx = {trait: idx for idx, trait in enumerate(args.trait_names)}
    batch_size = args.test_batch_size

    model.eval()
    with th.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            test = test_data[i:i + batch_size]
            predictions, _ = model.predict_trait_scores(
                essay_input_ids=th.tensor(test["essay_input_ids"]).to(args.device),
                essay_attention_mask=th.tensor(test["essay_attention_mask"]).to(args.device),
                gpt_input_ids=th.tensor(test["gpt_input_ids"]).to(args.device),
                gpt_attention_mask=th.tensor(test["gpt_attention_mask"]).to(args.device),
                llama_input_ids=th.tensor(test["llama_input_ids"]).to(args.device),
                llama_attention_mask=th.tensor(test["llama_attention_mask"]).to(args.device),
                prompt_ids=th.tensor(test["prompt_ids"]).to(args.device),
            )
            predictions = denormalize_trait_predictions(
                predictions.detach().cpu().numpy(),
                test["prompt_ids"],
                args,
            )
            predictions = quantize_trait_predictions(predictions, args)
            for row_idx in range(len(predictions)):
                for trait in trait_list:
                    trait_idx = trait_to_idx[trait]
                    pred_dic[trait].append(float(predictions[row_idx, trait_idx]))
                    true_dic[trait].append(float(test[trait][row_idx]))

    for trait in trait_list:
        if pred_dic[trait] and true_dic[trait]:
            qwk_result[trait] = quadratic_weighted_kappa(
                np.array(pred_dic[trait]),
                np.array(true_dic[trait]),
            )
    log = "Test Result"
    log += f"\n| {qwk_result} |"
    print(log)
    return qwk_result, pred_dic, true_dic


def asap_test(tokenizer, model, test_data, args):
    pred_dic = dict()
    true_dic = dict()
    qwk_result = dict()
    trait_map = {
        1: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
        2: ["overall", "content", "organization", "word choice", "sentence fluency", "conventions"],
        3: ["overall", "content", "prompt adherence", "language", "narrativity"],
        4: ["overall", "content", "prompt adherence", "language", "narrativity"],
        5: ["overall", "content", "prompt adherence", "language", "narrativity"],
        6: ["overall", "content", "prompt adherence", "language", "narrativity"],
        7: ["overall", "content", "organization", "style", "conventions"],
        8: ["overall", "content", "organization", "voice", "word choice", "sentence fluency", "conventions"]
    }
    
    for p in range(1, 9):
        pred_dic[p] = dict()
        true_dic[p] = dict()
        qwk_result[p] = dict()
        trait_list = trait_map[p]
        for trait in trait_list:
            pred_dic[p][trait] = list()
            true_dic[p][trait] = list()
            qwk_result[p][trait] = 0.0

    model.eval()
    batch_size = 128

    if getattr(args, "use_adaptive_rmts", False):
        try:
            return advanced_asap_test(model, test_data, args, trait_map)
        except Exception as e:
            print(f"Warning: advanced_asap_test failed: {e}, falling back to standard asap_test")
    
    with th.no_grad():
        for i in range(0, len(test_data), batch_size):
            test = test_data[i:i+batch_size]
            input_ids_all = th.tensor(test['input_ids']).to(args.device)
            attention_mask = th.tensor(test['attention_mask']).to(args.device)

            split_index = args.max_essay_length
            rationale_length = args.max_rationale_length
            essay_input_ids = input_ids_all[:, :split_index]
            essay_attention_mask = attention_mask[:, :split_index]

            if 'bart' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
            elif 't5' in args.model_name:
                encoder_outputs = model.encoder(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
            elif 'pegasus' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
            elif 'led' in args.model_name:
                encoder_outputs = model.led.encoder(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
            
            criteria_ids = input_ids_all[:, split_index:split_index + rationale_length]
            criteria_attention_mask = attention_mask[:, split_index:split_index + rationale_length]
            
            if 'bart' in args.model_name:
                criteria_encoder_outputs = model.model.encoder(input_ids=criteria_ids, attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.model.proj(th.concat([encoder_outputs[0], criteria_encoder_outputs[0]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            elif 't5' in args.model_name:
                criteria_encoder_outputs = model.encoder(input_ids=criteria_ids, attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.proj(th.concat([encoder_outputs[0], criteria_encoder_outputs[0]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            elif 'pegasus' in args.model_name:
                criteria_encoder_outputs = model.model.encoder(input_ids=criteria_ids, attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.model.proj(th.concat([encoder_outputs[0], criteria_encoder_outputs[0]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            elif 'led' in args.model_name:
                criteria_encoder_outputs = model.led.encoder(input_ids=criteria_ids, attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.led.proj(th.concat([encoder_outputs[0], criteria_encoder_outputs[0]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)

            labels = test['t5_output']
            prompts = test["essay_set"]

            decoder_start_token_id = model.config.decoder_start_token_id
            input_ids = th.tensor([[decoder_start_token_id] for _ in range(encoder_outputs[0].size(0))]).to(args.device)

            if "t5" in args.model_name:
                outputs = model.generate(input_ids=input_ids, encoder_outputs=encoder_outputs, max_new_tokens=64, num_beams=1)
            else:
                outputs = model.generate(input_ids=input_ids, encoder_outputs=encoder_outputs, max_new_tokens=256, num_beams=1)

            for j, (output, true) in enumerate(zip(outputs, labels)):
                pred = tokenizer.decode(output, skip_special_tokens=True)
                
                try:
                    pred_result = parse_asap_traits_text(pred)
                    true_result = parse_asap_traits_text(true)

                    prompt = prompts[j]
                    trait_list = trait_map[prompt]

                    for trait in trait_list:
                        if trait not in pred_result:
                            continue
                        
                        if np.isnan(pred_result[trait]):
                            pred_dic[prompt][trait].append(0)
                            true_dic[prompt][trait].append(true_result.get(trait, 0))
                        else:
                            pred_dic[prompt][trait].append(pred_result[trait])
                            true_dic[prompt][trait].append(true_result.get(trait, 0))
                    
                except Exception as e:
                    print(f"Error processing prediction: {e}")
                    continue
        
        for prompt in range(1, 9):
            trait_list = trait_map[prompt]
            for trait in trait_list:
                if len(pred_dic[prompt][trait]) > 0 and len(true_dic[prompt][trait]) > 0:
                    try:
                        qwk_result[prompt][trait] = quadratic_weighted_kappa(
                            np.array(pred_dic[prompt][trait]), 
                            np.array(true_dic[prompt][trait])
                        )
                    except Exception as e:
                        qwk_result[prompt][trait] = 0.0
                        
        log = "Test Result"
        for prompt in range(1, 9):
            log += f"\n\n| Prompt: {prompt} |"
            log += f"\n| {qwk_result[prompt]} |"
        print(log)

    return qwk_result, pred_dic, true_dic


def feedback_test(tokenizer, model, test_data, args):
    pred_dic = dict()
    true_dic = dict()
    qwk_result = dict()
    trait_list = ["conventions", "grammar", "phraseology", "vocabulary", "syntax", "cohesion"]

    for trait in trait_list:
        pred_dic[trait] = list()
        true_dic[trait] = list()
        qwk_result[trait] = 0.0

    model.eval()
    batch_size = 128

    if getattr(args, "use_adaptive_rmts", False):
        try:
            return advanced_feedback_test(model, test_data, args, trait_list)
        except Exception as e:
            print(f"Warning: advanced_feedback_test failed: {e}, falling back to standard feedback_test")

    with th.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            test = test_data[i:i+batch_size]
            input_ids_all  = th.tensor(test['input_ids']).to(args.device)
            attention_mask =  th.tensor(test['attention_mask']).to(args.device)

            split_index = args.max_essay_length
            rationale_length = args.max_rationale_length
            essay_input_ids = input_ids_all[:, :split_index]
            essay_attention_mask = attention_mask[:, :split_index]

            if 'bart' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
            elif 't5' in args.model_name:
                encoder_outputs = model.encoder(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
            elif 'pegasus' in args.model_name:
                encoder_outputs = model.model.encoder(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
            elif 'led' in args.model_name:
                encoder_outputs = model.led.encoder(input_ids=essay_input_ids, attention_mask=essay_attention_mask)
            
            criteria_ids = input_ids_all[:, split_index:split_index + rationale_length]
            criteria_attention_mask = attention_mask[:, split_index:split_index + rationale_length]
            
            if 'bart' in args.model_name:
                criteria_encoder_outputs = model.model.encoder(input_ids=criteria_ids, attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.model.proj(th.concat([encoder_outputs[0], criteria_encoder_outputs[0]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            elif 't5' in args.model_name:
                criteria_encoder_outputs = model.encoder(input_ids=criteria_ids, attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.proj(th.concat([encoder_outputs[0], criteria_encoder_outputs[0]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            elif 'pegasus' in args.model_name:
                criteria_encoder_outputs = model.model.encoder(input_ids=criteria_ids, attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.model.proj(th.concat([encoder_outputs[0], criteria_encoder_outputs[0]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
            elif 'led' in args.model_name:
                criteria_encoder_outputs = model.led.encoder(input_ids=criteria_ids, attention_mask=criteria_attention_mask)
                encoder_outputs.last_hidden_state = model.led.proj(th.concat([encoder_outputs[0], criteria_encoder_outputs[0]], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
                    
            labels = test['t5_output']
            decoder_start_token_id = model.config.decoder_start_token_id
            input_ids = th.tensor([[decoder_start_token_id] for _ in range(encoder_outputs[0].size(0))]).to(args.device)

            if "flan-t5-base" in args.model_name:
                outputs = model.generate(input_ids=input_ids, encoder_outputs=encoder_outputs, max_new_tokens=256, num_beams=1)
            else:
                outputs = model.generate(input_ids=input_ids, encoder_outputs=encoder_outputs, max_new_tokens=64, num_beams=1)

            for i, (output, true) in enumerate(zip(outputs, labels)):
                pred = tokenizer.decode(output, skip_special_tokens=True)
                
                try:
                    pred = pred.replace(" ,", ",").replace(". ", ", ").replace(".,", ",").replace("  "," ").replace(" ;",",").replace(" :", ",").replace("and", ",").strip()
                    pred = pred.replace("1.0", " 1.0").replace("1.5", " 1.5").replace("2.0", " 2.0").replace("2.5", " 2.5").replace("3.0", " 3.0").replace(
                        "3.5", " 3.5").replace("4.0", " 4.0").replace("4.5", " 4.5").replace("5.0", " 5.0")

                    if args.model_name == "bart":
                        pred_result = extract_traits(pred)
                    else:
                        preds = pred.split(",")
                        pred_result = dict()
                        for p in preds:
                            p = p.strip()
                            key, value = p.split(' ', 1)
                            pred_result[key] = float(value)
                    
                    true_result = "{" + re.sub(r'(\w+)\s([\d\.]+)', r'"\1": \2', true) + "}"
                    true_result = eval(true_result)

                    for trait in trait_list:
                        pred_dic[trait].append(pred_result[trait])
                        true_dic[trait].append(true_result[trait])
                    
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
                    
        for trait in trait_list:
            try:
                qwk_result[trait] = quadratic_weighted_kappa(np.array(pred_dic[trait]), np.array(true_dic[trait]))
            except Exception as e:
                print(f"An error occurred: {e} for BART")
                qwk_result[trait] = 0.0
                                           
        log = "Test Result"
        log += f"\n| {qwk_result} |"
        print(log)

    return qwk_result, pred_dic, true_dic


def deep_copy_state_dict(state_dict):
    copy_dict = {}
    for key, value in state_dict.items():
        copy_dict[key] = value.clone()
    return copy_dict


class SaveTopModelsCallback(TrainerCallback):
    
    def __init__(self, save_path, top_k=2):
        self.save_path = save_path
        self.top_k = top_k
        self.top_models = []  

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_loss = metrics['eval_loss']
        current_step = state.global_step
        kwargs["model"] = kwargs["model"].cpu()
        model_state_dict = deep_copy_state_dict(kwargs['model'].state_dict())  
        kwargs["model"] = kwargs["model"].to(args.device)

        self.top_models.append((current_loss, current_step, model_state_dict))
        self.top_models.sort(key=lambda x: x[0])  
        self.top_models = self.top_models[:self.top_k]  

        self.cleanup_and_save_top_models()

    def cleanup_and_save_top_models(self):
        for filename in os.listdir(self.save_path):
            if filename.startswith("checkpoint"):
                os.remove(os.path.join(self.save_path, filename))
        
        for rank, (loss, step, state_dict) in enumerate(self.top_models):
            model_path = os.path.join(self.save_path, f"checkpoint-{rank+1}-loss-{loss:.4f}")
            th.save(state_dict, model_path)
            print(f"Saved top {rank+1} model to {model_path} with loss {loss:.4f}")

