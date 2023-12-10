#### data

from datasets import load_dataset
import os

class args:
   def __init__(self):
        return
data_args = args()
data_args.task_name = "rte"
data_args.dataset_cache_dir = '/storage/ice1/6/5/afischer39/llmft/cache'
print(data_args.dataset_cache_dir)

from eval_utils import create_few_shot_context
in_context_args = args()
in_context_args.num_shots = 10
in_context_args.pattern = "{text1} {text2} ?"
in_context_args.task_description = "The answer is yes if sentence2 follows from sentence1. "
in_context_args.balanced = True
in_context_args.separate_shots_by = '\n\n'

training_args = args()
training_args.data_seed = 123

from task_utils import task_to_keys
limit = 50


raw_datasets = load_dataset(
        "glue",
        data_args.task_name,
        cache_dir=data_args.dataset_cache_dir,
        token=None,
    )


target_tokens = "ĠYes,ĠNo"
target_tokens = [t.strip() for t in target_tokens.split(",")]
id_to_target_token = {idx: t for idx, t in enumerate(target_tokens)}

context, contex_indices = create_few_shot_context(
    dataset_name = data_args.task_name, 
    dataset = raw_datasets["train"], 
    num_shots = in_context_args.num_shots, 
    pattern = in_context_args.pattern,
    label_to_tokens=id_to_target_token,
    separate_shots_by=in_context_args.separate_shots_by, 
    description=in_context_args.task_description,
    # target_prefix=in_context_args.target_prefix,
    # from_indices=in_context_args.sample_indices_file, 
    balanced=in_context_args.balanced, 
    # shuffle=in_context_args.shuffle,
    seed=training_args.data_seed
)
pattern = f"{context}{in_context_args.pattern}"


examples = raw_datasets['train']

sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
print(sentence1_key)

pattern_examples = [
    pattern.format(
        text1=examples[sentence1_key][idx],
        text2=examples[sentence2_key][idx] if sentence2_key is not None else None)
    for idx in range(len(examples[sentence1_key][:limit]))
]

pattern_examples

#### modeling

from models.opt_wrapper import OPTWithClassifier, OPTWithLMClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoConfig
# print("Libraries imported")

num_labels=2

##Load Model
##tokenizer: needed for max length
def load_model(model_name,dataset_name):
    config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            finetuning_task=dataset_name,
            cache_dir=data_args.dataset_cache_dir,
            revision="main",
            use_auth_token=None
            )
    if "facebook/opt" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config,
                cache_dir=data_args.dataset_cache_dir,
                revision="main",
                use_auth_token= None,
                ignore_mismatched_sizes=False
            )
        tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=data_args.dataset_cache_dir,
                revision="main",
                use_auth_token=None,
            )
    else:
        raise NotImplementedError(f"Unsupported model_name: {model_name}")
    return model, tokenizer
model, tokenizer = load_model("facebook/opt-125m", "rte")
print(model)
print(tokenizer)

# result = tokenizer(context, padding=padding, max_length=max_seq_length, truncation=False)

outputs = {}
for i, prompt in enumerate(pattern_examples[:limit]):
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=30)
    data = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  
    print(data)
    outputs[i] = data



outputs = {o:outputs[o].split('?')[-1] for o in outputs}

labels = [id_to_target_token[l][1:] for l in examples['label'][:limit]]
correct = [labels[i] == outputs[i] for i in range(len(labels))]

print('Accuracy:', sum(correct) / len(correct))
print('Minority class:', min(list(outputs.values()).count('Yes'), list(outputs.values()).count('No')) / len(correct))