from typing import Union, Optional, Callable, Dict, List, Tuple
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from torch import nn
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F



class ContextDistillationTrainer(Trainer):
    def __init__(
            self,
            teacher: Union[PreTrainedModel, nn.Module],
            distillation_weight: float,
            student_loss_fn = nn.CrossEntropyLoss(),
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics)
        self.teacher = teacher
        self.distillation_weight = distillation_weight
        self.student_loss_fn = student_loss_fn
    

    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        student_logits = model(inputs)

        teacher_out = nn.functional.log_softmax(teacher_logits, dim=-1)
        student_out = nn.functional.log_softmax(student_logits, dim=-1)

        teacher_loss = F.kl_div(student_out, teacher_out)

        labels = inputs.pop("labels")
        student_loss = self.student_loss_fn(student_logits, labels)

        loss = self.distillation_weight * teacher_loss + (1. - self.distillation_weight) * student_loss

        return (loss, student_out) if return_outputs else loss



