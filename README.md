## Evaluating the Performance and Efficiency of Sentence-BERT for Code Comment Classification

Source code required to replicate the results reported in *F. Peña, S. Herbold, "Evaluating the Performance and Efficiency of Sentence-BERT for Code Comment Classification", 2024*.

### Abstract

This work evaluates ([Sentence-BERT](https://sbert.net/index.html)) for a multi-label code comment classification task seeking to maximize the classification performance while controlling efficiency constrains during inference. Using a dataset of 13,216 labeled comment sentences, embedding models are fine-tuned and combined with different classification heads to recognize the type of comment. As expected, larger embedding models outperform smaller ones in terms of F1 while the latter require shorter runtime and less floating-point operations per second (GFLOPS). Nevertheless, small embedding models may still be competitive in both aspects when combining with more complex classification heads.

### Experimentation process

The experimentation process is done in two stages:
1. Fine-tuning of embedding models combined with a logistic regression classification head by default.
2. Optimization of classification heads using different algorithms such as support vector machines, random forest, and xgboost.

Models and results are managed using a local registry following this sintax: `"{alias}-i{num_iterations}-h{head}"`, where `alias` corresponds to a short version of the pre-trained embedding model, `num_iterations` to the reference value required to calculate the number of contrastive examples used to fine-tune the embedding model, and `head` to the head type (logistic regression = LR, support vector machines = SVM, random forest = RF, xgboost = XG).

Pre-trained embedding models and aliases are referenced below:

```
sentence-transformers/paraphrase-MiniLM-L3-v2 = pml3
sentence-transformers/all-mpnet-base-v2 = amb
sentence-transformers/all-MiniLM-L6-v2 = aml6
sentence-transformers/paraphrase-albert-small-v2 = pas
sentence-transformers/all-distilroberta-v1 = adr
```

### Repository organization

1. `competition_s1`: Detailed results reported in the first experimentation stage (fine-tuning of embedding models). Each JSON file contains the F1 for each label, average for each language, and overall average, runtime, GFLOPS, and submission score.
2. `competition_s2`: Detailed results reported in the second experimentation stage (optimization of classification heads). Results are organized by language. Each JSON file contains the F1 for each label and average, runtime, GFLOPS, and submission score. The head corresponding to the best candidate it is also included. `baseline.json` corresponds to results for the baseline embedding model with optimized head and `nlbse25.json` without optimized head (default LR). `final.json` corresponds to results for the joint evaluation of the best combinations of embedding models and classification heads for each language.
3. `src`: Source code for replication. The order of execution of relevant scripts is as follows:  
    3.1. `evaluate_baseline.ipynb`: Replicate the results from *G. Colavito, A. Al-Kaswan, N. Stulova, and P. Rani, "The NLBSE'25
tool competition," in Proceedings of The 4th International Workshop on Natural Language-based Software Engineering (NLBSE'25), 2025*.  
    3.2. `train_setfit.py`: Fine-tune a Sentence-BERT model combined with a classification head. The process is carried out independently by language.  
    3.3. `evaluate_submission_s1.ipynb`: Evaluate candidates in terms of classification performance, runtime, and GFLOPS across languages. Candidates must use the same pre-trained embedding model.  
    3.4. `analyze_results_s1.ipynb`: Analyze the results of the first experimentation stage. Highlight the contributions of the classification performance, runtime, and GFLOPS to the submission score.   
    3.5. `optimize_head.py`: Run experiments by language to find a better classification head for an embedding model. The process is carried out independently by language.  
    3.6. `analyze_results_s2-1.ipynb`: Analyze the first part of the results of the second experimentation stage. Choose the best candidate by embedding model and language across number of iterations.  
    3.7. `evaluate_submission_s2-1.ipynb`: Evaluate candidates in terms of classification performance, runtime, and GFLOPS for a given language.  
    3.8. `analyze_results_s2-2.ipynb`: Analyze the second part of the results of the second experimentation stage. Aggregate classification performance, runtime, GFLOPS, and submission scores by language.  
    3.9. `evaluate_submission_s2-2.ipynb`: Evaluate candidates in terms of classification performance, runtime, and GFLOPS across languages. Candidates migth use different pre-trained embedding models and head types.  

### Environment

The hardware used for training and evaluation includes 8 GPUs NVDIA A100-SXM4-80GB. 
