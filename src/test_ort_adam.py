import gc
import tempfile
import unittest
from typing import List, Optional
from unittest.mock import Mock, patch

import nltk
import numpy as np
from datasets import load_dataset, load_metric

from onnxruntime.training.optim import FusedAdam
from parameterized import parameterized
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_available,
)
from transformers.training_args import OptimizerNames


default_adam_kwargs = {
    "betas": (TrainingArguments.adam_beta1, TrainingArguments.adam_beta2),
    "eps": TrainingArguments.adam_epsilon,
    "lr": TrainingArguments.learning_rate,
}

optim_test_params = [
    (
        OptimizerNames.ADAMW_ORT_FUSED,
        FusedAdam,
        default_adam_kwargs,
    )
]


class ORTTrainerOptimizerChoiceTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = min(args.num_train_epochs, 1)
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size

        self.max_seq_length = 128
        self.max_train_samples = 200
        self.max_valid_samples = 50
        self.max_test_samples = 20

        self.warmup_steps = 500
        self.weight_decay = 0.01

        self.model_name = "bert-base-cased"
        self.feature = "sequence-classification"

    def check_optim_and_kwargs(self, optim: OptimizerNames, mandatory_kwargs, expected_cls):
        args = TrainingArguments(optim=optim, output_dir="None")
        actual_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        self.assertEqual(expected_cls, actual_cls)
        self.assertIsNotNone(optim_kwargs)

        for p, v in mandatory_kwargs.items():
            self.assertTrue(p in optim_kwargs)
            actual_v = optim_kwargs[p]
            self.assertTrue(actual_v == v, f"Failed check for {p}. Expected {v}, but got {actual_v}.")

    @parameterized.expand(optim_test_params, skip_on_empty=True)
    def test_optim_supported(self, name: str, expected_cls, mandatory_kwargs):
        # exercises all the valid --optim options
        self.check_optim_and_kwargs(name, mandatory_kwargs, expected_cls)

        with tempfile.TemporaryDirectory() as tmp_dir:

            # Prepare model
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Prepare dataset
            dataset_name = "sst2"
            dataset = load_dataset("glue", dataset_name)
            metric = load_metric("glue", dataset_name)

            max_seq_length = min(128, tokenizer.model_max_length)
            padding = "max_length"

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

            def preprocess_function(examples):
                args = (examples["sentence"],)
                return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            encoded_dataset = dataset.map(preprocess_function, batched=True)
            max_train_samples = 200
            max_valid_samples = 50
            max_test_samples = 20
            train_dataset = encoded_dataset["train"].select(range(max_train_samples))
            valid_dataset = encoded_dataset["validation"].select(range(max_valid_samples))
            test_dataset = encoded_dataset["test"].select(range(max_test_samples)).remove_columns(["label"])

            def compute_metrics(eval_pred):
                predictions = (
                    eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
                )
                if dataset_name != "stsb":
                    predictions = np.argmax(predictions, axis=1)
                else:
                    predictions = predictions[:, 0]
                return metric.compute(predictions=predictions, references=eval_pred.label_ids)

            print(f"optimizer is {name}")
            training_args = TrainingArguments(
                optim=name,
                output_dir=tmp_dir,
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=self.weight_decay,
                logging_dir=tmp_dir,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
            )

            train_result = trainer.train()
            trainer.save_model()
            train_metrics = train_result.metrics
            eval_metrics = trainer.evaluate(inference_with_ort=True)
            prediction = trainer.predict(test_dataset, inference_with_ort=True)
            print("Training metrics:\n", train_metrics)
            print("Evaluation metrics:\n", eval_metrics)
            print("Prediction results:\n", prediction)
            gc.collect()

    # @unittest.skip("Skip ORT Fused Adam optimizer test.")  # Not merged yet.
    def test_ort_fused_adam(self):
        # Pretend that onnxruntime-training is installed and mock onnxruntime.training.optim.FusedAdam exists.
        # Trainer.get_optimizer_cls_and_kwargs does not use FusedAdam. It only has to return the
        # class given, so mocking onnxruntime.training.optim.FusedAdam should be fine for testing and allow
        # the test to run without requiring an onnxruntime-training installation.
        mock = Mock()
        modules = {
            "onnxruntime.training": mock,
            "onnxruntime.training.optim": mock.optimizers,
            "onnxruntime.training.optim.FusedAdam": mock.optimizers.FusedAdam,
        }
        with patch.dict("sys.modules", modules):
            self.check_optim_and_kwargs(
                OptimizerNames.ADAMW_ORT_FUSED,
                default_adam_kwargs,
                mock.optimizers.FusedAdam,
            )
        print("ort tested")


if __name__ == "__main__":
    unittest.main()
