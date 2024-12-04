# Sentence-BERT for Code Comment Classification

This repository contains all the source code required to replicate the results reported in *F. Peña, S. Herbold, "Evaluating the Performance and Efficiency of Sentence-BERT for Code Comment Classification", 2024*.

## General description

The scope of the paper includes to train a set of sentence embedding ([Sentence-BERT](https://sbert.net/index.html)) models and classification heads for solving the task of code comment classification in languages such as Java, Python and Pharo. The development framework used is [SetFit](https://huggingface.co/docs/setfit/index).

The experimentation process is done in two stages:
1. Fine-tuning of embedding models, including the training of a default logistic regression classification head.
2. Optimization of classification heads using different classification algorithms such as support vector machines, random forest, and xgboost.

Models and results are managed using a local registry following this sintax: `"{alias}-i{num_iterations}-h{head}"`, where `alias` corresponds to a short version of the pre-trained embedding model, `num_iterations` to the reference value required to calculate the number of contrastive examples used to fine-tune the embedding model, and `head` to the head classifier type (logistic regression = LR, support vector machines = SVM, random forest = RF, xgboost = XG).

Pre-trained embedding models and aliases are referenced below:

```
sentence-transformers/paraphrase-MiniLM-L3-v2 = pml3
sentence-transformers/all-mpnet-base-v2 = amb
sentence-transformers/all-MiniLM-L6-v2 = aml6
sentence-transformers/paraphrase-albert-small-v2 = pas
sentence-transformers/all-distilroberta-v1 = adr
```

## Repository organization

1. `competition_s1`: Detailed, raw results reported in the first experimentation stage (fine-tuning of embedding models) in terms of classification performance, runtime and GFLOPS across languages.
2. `competition_s2`: Detailed, raw results reported in the second experimentation stage (optimization of classification heads) in terms of classification performance, runtime and GFLOPS across languages.
3. `optimized_models`: Part of the model registry which includes the classification performance for the best head found for each fine-tuned embedding model by language.
4. `src`: Source code for replication. The order of execution of relevant scripts is as follows:  
    4.1. `evaluate_baseline.ipynb`: Replicate the results from *G. Colavito, A. Al-Kaswan, N. Stulova, and P. Rani, "The NLBSE'25
tool competition," in Proceedings of The 4th International Workshop on
Natural Language-based Software Engineering (NLBSE'25), 2025*.  
    4.2. `train_setfit.py`: Fine-tune a Sentence-BERT model with a classification head on top by language using the SetFit framework.  
    4.3. `evaluate_submission_s1.ipynb`: Evaluate candidates in terms of classification performance, runtime and GFLOPS across languages. Candidates must use the same pre-trained embedding model.  
    4.4. `analyze_results_s1.ipynb`: Analyze the results of the first experimentation stage.  
    4.5. `optimize_head.py`: Run experiments by language to find a better classification head for a Sentence-BERT embedding model.  
    4.6. `evaluate_submission_s2.ipynb`: Evaluate candidates in terms of classification performance, runtime and GFLOPS across languages. Candidates migth use different pre-trained embedding models and head types.  
    4.7. `analyze_results_s2.ipynb`: Analyze the results of the second experimentation stage.  

The hardware used for training and inference includes 8 GPUs NVDIA A100-SXM4-80GB. 
