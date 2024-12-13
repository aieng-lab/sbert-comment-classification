"""Train SetFit.

This module provides the required logic to fine-tune a Sentence-BERT model combined with a classification head. The process is carried out independently by language.
"""

from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    EarlyStoppingCallback,
    IntervalStrategy,
)
from sentence_transformers import losses
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
from setfit import (
    SetFitModel,
    TrainingArguments,
    Trainer,
)


MODEL_ALIASES = {
    "sentence-transformers/paraphrase-MiniLM-L3-v2": "pml3",
    "sentence-transformers/all-mpnet-base-v2": "amb",
    "sentence-transformers/all-MiniLM-L6-v2": "aml6",
    "sentence-transformers/paraphrase-albert-small-v2": "pas",
    "sentence-transformers/all-distilroberta-v1": "adr"
}

NEW_MODEL_NAME = "{}-i{}-hLR"


def main():
    
    model_name = "sentence-transformers/all-distilroberta-v1"
    multi_target_strategy = "multi-output"
    batch_size = 32
    num_epochs = {"java": 5, "python": 10, "pharo": 10}
    num_iterations = 60
    
    langs = ["java", "python", "pharo"]
    labels = {
        "java": ["summary", "Ownership", "Expand", "usage", "Pointer", "deprecation", "rational"],
        "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
        "pharo": ["Keyimplementationpoints", "Example", "Responsibilities", "Classreferences", "Intent", "Keymessages", "Collaborators"]
    }  

    new_model_name = NEW_MODEL_NAME.format(MODEL_ALIASES[model_name], num_iterations)
    print("=====")
    print("New model:", new_model_name)
    print("=====")

    dataset = load_dataset("NLBSE/nlbse25-code-comment-classification")

    for lan in langs:

        model = SetFitModel.from_pretrained(model_name, multi_target_strategy=multi_target_strategy, device="cuda")

        args = TrainingArguments(
            output_dir=f"../results/{new_model_name}/{lan}",
            num_iterations=num_iterations,
            num_epochs=num_epochs[lan],
            body_learning_rate=2e-5,
            head_learning_rate=2e-5,
            batch_size=batch_size,
            seed=42,
            use_amp=False,
            warmup_proportion=0.1,
            distance_metric=BatchHardTripletLossDistanceFunction.cosine_distance,
            margin=0.25,
            samples_per_label=2,
            loss=losses.CosineSimilarityLoss,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset[f"{lan}_train"],
            metric="accuracy",
            column_mapping={"combo": "text", "labels": "label"}
        )

        trainer.train()

        trainer.model.save_pretrained(f"../models/{new_model_name}/{lan}")

    print("=====")
    print("New model:", new_model_name)
    print("=====")


if __name__ == "__main__":
    main()
