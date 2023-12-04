def preprocess_raw_datasets(data_args, model):
    ## maybe this should be in a totally separate file?
    # --------------- Preprocessing the raw_datasets ---------------

    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(
            num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}

    if ft_args.target_tokens is not None and not ft_args.target_tokens_logits_only:
        # we need to convert the label ids to target ids
        target_tokens = [t.strip() for t in ft_args.target_tokens.split(",")]
        target_tokens_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        model.config.label2id = {
            l: target_tokens_ids[i] for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}

    # Compute max_seq_length
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts

        # Apply a pattern to the inputs
        pattern_examples = [
            ft_args.pattern.format(
                text1=examples[sentence1_key][idx],
                text2=examples[sentence2_key][idx] if sentence2_key is not None else None)
            for idx in range(len(examples[sentence1_key]))
        ]
        args = (pattern_examples,)
        result = tokenizer(*args, padding=padding,
                           max_length=max_seq_length, truncation=True)

        # Get mask for soft prompt tokens
        # TODO(mm): For GPT-J and GPT-NeoX we have a different tokenizer. Adjust accordingly
        if "opt" in model_args.model_name_or_path:
            # For OPT models, the first token is always the bos token </s>
            # Which happens to be also the unk token we use to mark soft prompt tokens
            # Hence, we have to be careful about which tokens to mask as part of the soft prompt
            result["soft_prompt_mask"] = [[0 if (idx != tokenizer.unk_token_id or pos == 0) else 1 for pos, idx in enumerate(indices)]
                                          for indices in result["input_ids"]]  # <unk> is the placeholder for prompt embeddings

        # Get tokens
        result["input_tokens"] = [tokenizer.convert_ids_to_tokens(
            ids) for ids in result["input_ids"]]

        # Decode input
        result["input_text"] = [tokenizer.decode(
            ids) for ids in result["input_ids"]]

        # Replace labels by target tokens indices when using lm_head
        # - special case: when using target logits only, we keep class indices instead of token indices
        if ft_args.target_tokens is not None and not ft_args.target_tokens_logits_only:
            result["label"] = [target_tokens_ids[l] for l in examples["label"]]
        else:
            result["label"] = examples["label"]

        result["label_text"] = [model.config.id2label[l] if l != -1 else "unlabeled"
                                for l in result["label"]]

        return result

    # We need to update the number of classes of the dataset when using the lm_head
    if ft_args.target_tokens is not None and not ft_args.target_tokens_logits_only:
        for split in raw_datasets:
            # raw_datasets[split].features["label"].num_classes = len(tokenizer)
            # raw_datasets[split].features["label"].names = [
            #     f"{idx}" for idx in np.arange(len(tokenizer))]

            new_features = raw_datasets[split].features.copy()
            names = [f"{idx}" for idx in np.arange(len(tokenizer))]
            new_features["label"] = ClassLabel(
                names=names, num_classes=len(tokenizer))
            raw_datasets[split] = raw_datasets[split].cast(new_features)

        for name, dataset in additional_evaluation_datasets.items():
            # dataset.features["label"].num_classes = len(tokenizer)
            # dataset.features["label"].names = [
            #     f"{idx}" for idx in np.arange(len(tokenizer))]

            new_features = dataset.features.copy()
            names = [f"{idx}" for idx in np.arange(len(tokenizer))]
            new_features["label"] = ClassLabel(
                names=names, num_classes=len(tokenizer))
            additional_evaluation_datasets[name] = dataset.cast(new_features)

    # before running the pre-processing, subsample datsets if specified

    # subsample datasets (if specified)

    # we fix the random seed that controls the sampling of the training data
    np.random.seed(training_args.data_seed)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # randomly select a subset of the training data
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            indices = np.random.choice(
                range(len(train_dataset)), size=max_train_samples, replace=False)
            train_dataset = train_dataset.select(indices)

    if training_args.do_eval:
        # we fix the random seed that controls the sampling of the validation data
        np.random.seed(123)  # we only use this for debugging

        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name in
                                    ["mnli", "mnli-original"] else "validation"]

        # (optional) subsample eval datasets
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            # randomly select a subset of the eval data
            indices = np.random.choice(
                range(len(eval_dataset)), size=max_eval_samples, replace=False)
            eval_dataset = eval_dataset.select(indices)

        for name, dataset in additional_evaluation_datasets.items():
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(dataset), data_args.max_eval_samples)
                # randomly select a subset of the eval data
                indices = np.random.choice(
                    range(len(dataset)), size=max_eval_samples, replace=False)
                dataset = dataset.select(indices)
                additional_evaluation_datasets[name] = dataset

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        # we fix the random seed that controls the sampling of the validation data
        np.random.seed(123)  # we only use this for debugging

        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name in
                                       ["mnli", "mnli-original"] else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))

    # set all random seeds again (not sure if this is really needed)
    set_seed(training_args.seed)

    # tokenize and encode datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        if training_args.do_train:
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on training dataset",
            )

        if training_args.do_eval:
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

        if training_args.do_predict:
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )

        for name, dataset in additional_evaluation_datasets.items():
            if "hans" in name:
                sentence1_key, sentence2_key = task_to_keys["hans"]
            elif "mnli" in name:
                sentence1_key, sentence2_key = task_to_keys["mnli"]
            elif "paws-qqp" in name:
                sentence1_key, sentence2_key = task_to_keys["paws-qqp"]
            elif "cola-ood" in name:
                sentence1_key, sentence2_key = task_to_keys["cola-ood"]

            dataset = dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on {name} validation dataset",
            )
            additional_evaluation_datasets[name] = dataset

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 1):
            print(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    # Log training and evaluation examples to training_args.output_dir for reproducibility
    if training_args.do_train:
        save_dataset(train_dataset, path=os.path.join(
            training_args.output_dir, f"{data_args.task_name}-train.csv"))
    if training_args.do_eval:
        save_dataset(eval_dataset, path=os.path.join(
            training_args.output_dir, f"{data_args.task_name}-eval.csv"))
        for name, dataset in additional_evaluation_datasets.items():
            save_dataset(dataset, path=os.path.join(
                training_args.output_dir, f"{name}-eval.csv"))

    # --------------- End preprocessing of the raw_datasets ---------------


def main():
    # Load training and validation datasets
    raw_datasets, label_list, num_labels, is_regression = load_glue_datasets(
        data_args, model_args)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load Classifier
    if "facebook/opt" in model_args.model_name_or_path:
        if ft_args.target_tokens is not None:
            model = OPTWithLMClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        else:
            model = OPTWithClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
    elif "llama" in model_args.model_name_or_path:
        if ft_args.target_tokens is not None:
            model = LlamaWithLMClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                torch_dtype=torch.float16,
            )

            # We need to add a padding token for llama
            tokenizer.pad_token = tokenizer._convert_id_to_token(
                config.pad_token_id)  # let's use the <unk> token
            tokenizer.padding_side = "right"

        else:
            raise NotImplementedError(
                f"Unsupported model_name_or_path: {model_args.model_name_or_path}")

    # preprocess datasets
    preprocess_raw_datasets()

    # Initialize Trainer
    trainer = #figure out what we need here

    # Train
    train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=["past_key_values"])

    # save results
    
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)
