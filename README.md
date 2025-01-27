## Evaluating the Performance and Efficiency of Sentence-BERT for Code Comment Classification

Source code required to replicate the results reported in *F. Peña, S. Herbold, "Evaluating the Performance and Efficiency of Sentence-BERT for Code Comment Classification," in Proceedings of The 4th International Workshop on Natural Language-based Software Engineering (NLBSE’25), 2025*.

### Abstract

This work evaluates [Sentence-BERT](https://sbert.net/index.html) for a multi-label code comment classification task seeking to maximize the classification performance while controlling efficiency constraints during inference. Using a dataset of 13,216 labeled comment sentences, Sentence-BERT models are fine-tuned and combined with different classification heads to recognize comment types. While larger models outperform smaller ones in terms of F1, the latter offer outstanding efficiency, both in runtime and GFLOPS. As result, a balance between a reasonable F1 improvement (+0.0346) and a minimal efficiency degradation (+1.4x in runtime and +2.1x in GFLOPS) is reached.

### Experimentation process

The experimentation process is done in two stages:
1. Fine-tuning of embedding models combined with a multi-output logistic regression classification head by default.
2. Selection of the best classification heads using different algorithms such as Random Forest, XGBoost, and Support Vector Machines.

Models and evaluation results are managed in a local registry following this naming sintax: `"{alias}-i{num_iterations}-h{head}"`, where `alias` corresponds to a short version of the pre-trained embedding model, `num_iterations` to the reference value required to calculate the number of contrastive examples used to fine-tune the embedding model (see paper, Section III), and `head` to the head type (logistic regression = LR, random forest = RF, xgboost = XG, support vector machines = SVM).

Pre-trained embedding models and aliases are referenced below:

```
paraphrase-MiniLM-L3-v2 = pml3
all-mpnet-base-v2 = amb
all-MiniLM-L6-v2 = aml6
paraphrase-albert-small-v2 = pas
all-distilroberta-v1 = adr
```

### Setup

The training and evaluation process leverages an 8-GPU NVIDIA A100 setup. However, final evaluation of the submitted models is also carried out on Google Colab T4, as specified in the competition rules (see paper, Section II and [NLBSE'25](https://nlbse2025.github.io/tools/)).

### Submission

The best combinations of embedding model and classification head (one for each language) are published on Hugging Face under the IDs `fabiancpl/nlbse25_java`, `fabiancpl/nlbse25_python`, and `fabiancpl/nlbse25_pharo`. When evaluating these models in Google Colab T4, the submission score obtained is 0.6536 (see [notebook](https://colab.research.google.com/drive/17Bep6v_1Ia_dVKNnVtg_myr7GMRhPfn1?usp=sharing)).

### Repository organization

1. `competition_s1`: Detailed results reported in the first experimentation stage (fine-tuning of embedding models). Each JSON file contains the F1 for each label, average for each language, and overall average, runtime, GFLOPS, and submission score.
2. `competition_s2`: Detailed results reported in the second experimentation stage (optimization of classification heads). Results are organized by language. Each JSON file contains the F1 for each label and average, runtime, GFLOPS, and submission score. The head corresponding to the best candidate it is also included. `baseline.json` corresponds to results for the baseline embedding model with optimized head and `nlbse25.json` without optimized head (default LR). `submission.json` corresponds to results for the joint evaluation of the best combinations of embedding models and classification heads for each language.
3. `src`: Source code for replication. The order of execution of relevant scripts is as follows:  
    3.1. `evaluate_baseline.ipynb`: Replicate the results from *G. Colavito, A. Al-Kaswan, N. Stulova, and P. Rani, "The NLBSE'25
tool competition," in Proceedings of The 4th International Workshop on Natural Language-based Software Engineering (NLBSE'25), 2025*.  
    3.2. `train_setfit.py`: Fine-tunes a Sentence-BERT model combined with a classification head. The process is carried out independently by language.  
    3.3. `evaluate_submission_s1.ipynb`: Evaluates candidates in terms of classification performance, runtime, and GFLOPS across languages. Candidates must use the same pre-trained embedding model.  
    3.4. `analyze_results_s1.ipynb`: Analyzes the results of the first experimentation stage. Highlights the contributions of the classification performance, runtime, and GFLOPS to the submission score.   
    3.5. `optimize_head.py`: Runs experiments by language to find a better classification head for an embedding model. The process is carried out independently by language.  
    3.6. `analyze_results_s2-1.ipynb`: Analyzes the first part of the results of the second experimentation stage. Chooses the best candidate by embedding model and language across number of iterations.  
    3.7. `evaluate_submission_s2-1.ipynb`: Evaluates candidates in terms of classification performance, runtime, and GFLOPS for a given language.  
    3.8. `analyze_results_s2-2.ipynb`: Analyzes the second part of the results of the second experimentation stage. Aggregates classification performance, runtime, GFLOPS, and submission scores by language.  
    3.9. `evaluate_submission_s2-2.ipynb`: Evaluates candidates in terms of classification performance, runtime, and GFLOPS across languages. Candidates migth use different pre-trained embedding models and head types.  
    3.10. `evaluate_submission_s2-2_alt.ipynb`: Same as the previous one, however, this notebook considers the special case when the original baseline (embedding model + classification head) is considered as the best for a given language (e.g., Java).  
    3.11. `evaluate_nb.ipynb`: Trains and evaluates a very simple Naive Bayes model that uses a bag-of-words representation of the comments. This candidate achieves a very high submission score because (0.682) but is not competitive in terms of classification performance (0.47).

### Cite as

```
@inproceedings{pena2025sentencebert,
  author    = {Fabian Peña and Steffen Herbold},
  title     = {Evaluating the Performance and Efficiency of Sentence-BERT for Code Comment Classification},
  booktitle = {Proceedings of the 4th International Workshop on Natural Language-based Software Engineering (NLBSE'25)},
  year      = {2025}
}
```
