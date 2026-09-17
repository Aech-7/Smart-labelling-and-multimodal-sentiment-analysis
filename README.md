# Smart Labeling & Multimodal Sentiment Analysis

An end-to-end pipeline for building, expanding, and validating sentiment datasets using **human annotation, weak supervision, active learning, LLM-assisted labeling, synthetic data generation, multilingual augmentation, and speech processing**.

The project explores how a small amount of reliable human-labeled data can be used to efficiently construct a larger training dataset while continuously checking the quality of automatically generated labels and data.

---

## Overview

Creating a high-quality labeled dataset is often one of the most expensive parts of developing a machine learning system. Manually labeling every sample is costly, while automatically generated labels can introduce noise.

This project builds a multi-stage pipeline that combines **human expertise with programmatic and AI-based labeling methods**.

The pipeline starts with a small human-annotated gold-standard dataset and progressively expands it using weak supervision, active learning, and LLM-assisted labeling. The resulting data is then augmented across text, multilingual, and audio modalities and evaluated using automated quality-control metrics.

```text
                         Raw Reviews
                              │
                              ▼
                    ┌───────────────────┐
                    │  Human Annotation │
                    │  + Agreement      │
                    │    Analysis       │
                    └─────────┬─────────┘
                              │
                              ▼
                       Gold Standard
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
       Weak Supervision   Active Learning   LLM Labeling
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                       Label Curation
                              │
                              ▼
                    Consolidated Dataset
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      Text Augmentation  Synthetic Data  Multilingual
             │                │          Augmentation
             └────────────────┼────────────────┘
                              ▼
                     Audio Generation
                              │
                              ▼
                       Quality Control
                              │
                              ▼
                       Final Dataset
                              │
                              ▼
                        ML Evaluation
```

---

## Key Components

### 1. Human Annotation & Label Agreement

The first stage establishes a trusted set of labels through independent annotation.

* Three annotators label the same set of reviews.
* Inter-annotator agreement is measured using **Fleiss' Kappa implemented from scratch**.
* Results are compared against the `statsmodels` implementation.
* Conflicting annotations are automatically identified.
* A majority-vote adjudication strategy is used to create the final gold-standard labels.
* Three-way disagreements are resolved using a predefined neutral-label rule.

**Output:**

```text
gold_standard_100.csv
```

---

### 2. Weak Supervision

Instead of manually labeling the remaining reviews, programmatic labeling functions are used to generate weak labels.

Example labeling strategies include:

* Sentiment keyword rules
* Review-length heuristics
* Pattern-based sentiment detection
* Other rule-based labeling functions

The rules are implemented using **Snorkel**.

The pipeline measures:

* Labeling coverage
* Labeling conflicts
* Agreement between labeling functions
* Majority-vote weak labels

**Output:**

```text
weak_labels_200.csv
```

---

### 3. Active Learning

Active learning is used to simulate a limited annotation budget.

A Logistic Regression classifier is trained on the gold-standard dataset and used to identify the most informative samples from the unlabeled pool.

Implemented query strategies:

* **Least Confidence**
* **Entropy Sampling**
* **Random Sampling baseline**

The model is iteratively retrained as additional samples are labeled.

The resulting learning curves compare how quickly each strategy improves model performance as the annotation budget increases.

---

### 4. LLM-Assisted Labeling & Noise Detection

LLMs are used as another source of automatically generated sentiment labels.

A few-shot prompting strategy provides examples from the gold-standard dataset before labeling previously unseen reviews.

Because LLM-generated labels may contain errors, the pipeline performs a second-stage quality check.

A review is flagged when:

```text
Model confidence > 0.90
        AND
Model prediction != LLM label
```

This identifies **high-confidence disagreements** that may represent potentially noisy LLM annotations.

**Output:**

```text
llm_labels_150.csv
```

---

## Multimodal Data Expansion

Once the dataset has been curated, the project explores multiple ways of expanding the training data.

### 5. Classical Text Augmentation

Minority-class samples are augmented using:

* WordNet synonym replacement
* English → Hindi → English back translation

Generated samples are filtered using **Jaccard similarity** to reduce excessive duplication or overly similar examples.

---

### 6. LLM-Based Synthetic Data

LLMs are used to generate additional sentiment-labeled reviews.

The generated data is evaluated using:

* **Self-BLEU** for measuring textual diversity
* Sentiment consistency between the generated label and an independent sentiment classifier
* Similarity-based filtering for potentially problematic generations

This separates **data generation** from **data validation**, rather than assuming all generated samples are automatically usable.

---

### 7. Multilingual Augmentation

A subset of the dataset is translated through a Hindi round-trip:

```text
English
   ↓
Hindi
   ↓
English
```

The translated samples are evaluated using:

* **BLEU** for translation similarity
* Sentiment consistency between the original and translated reviews

This provides a simple way to test whether augmentation preserves the original semantic and sentiment information.

---

### 8. Speech Generation & Validation

The project extends the text dataset into the audio modality.

```text
Sentiment Review
      │
      ▼
Text-to-Speech
      │
      ▼
WAV Audio
      │
      ├── Audio Features
      │
      └── Whisper Transcription
                │
                ▼
         Round-trip Validation
```

Extracted audio information includes features such as:

* Duration
* Spectral centroid
* Zero-crossing rate
* MFCCs

Generated speech is transcribed using **Whisper**, and the transcription is compared against the original text using **Word Error Rate (WER)**.

This provides an automated check that the generated audio remains faithful to its source text.

---

## Quality Control

A major focus of the project is detecting unreliable data before it is added to the final training set.

| Data Source            | Quality Check                     |
| ---------------------- | --------------------------------- |
| Human annotations      | Fleiss' Kappa                     |
| Heuristic labels       | Coverage & conflict rate          |
| LLM labels             | Model confidence disagreement     |
| Classical augmentation | Jaccard similarity                |
| Synthetic reviews      | Self-BLEU & sentiment consistency |
| Back translation       | BLEU & sentiment consistency      |
| Generated audio        | Audio features & WER              |

The goal is not simply to generate more data, but to **generate additional data while measuring whether it is trustworthy enough to use**.

---

## Dataset Construction

The datasets are progressively consolidated through the pipeline:

```text
Human Labels
     +
Weak Labels
     +
Filtered LLM Labels
     │
     ▼
Consolidated Base Dataset
     │
     ├── Classical Augmentation
     ├── LLM Synthetic Data
     ├── Multilingual Data
     └── Audio Data
     │
     ▼
Quality Filtering
     │
     ▼
Final Augmented Dataset
```

---

## Evaluation

The final stage compares the predictive performance of datasets produced at different stages of the pipeline.

The evaluation includes:

* Baseline model performance
* Consolidated dataset performance
* Augmented dataset performance
* Active-learning learning curves
* Augmentation ablations

The main question is:

> **Does automated labeling and data augmentation improve the quality and usefulness of the training dataset?**

Results and figures are stored under:

```text
results/
├── figures/
└── metrics/
```

---

## Technologies

**Programming**

* Python
* Pandas
* NumPy

**Machine Learning**

* Scikit-learn
* Logistic Regression
* TF-IDF

**Weak Supervision & Labeling**

* Snorkel
* Label Studio
* Active Learning
* LLM-assisted labeling

**NLP**

* NLTK
* WordNet
* BLEU
* Jaccard Similarity
* Self-BLEU

**Multilingual & Speech**

* Translation APIs
* gTTS
* Librosa
* Whisper

**Evaluation**

* Statsmodels
* Custom evaluation metrics
* BlackBoxEvaluator

---

## Key Learning Outcomes

This project demonstrates experience with:

* Designing an end-to-end ML data pipeline
* Human-in-the-loop annotation
* Inter-annotator agreement
* Weak supervision and programmatic labeling
* Active learning and uncertainty sampling
* LLM-assisted data labeling
* Detection of potentially noisy labels
* Synthetic data generation
* Text and multilingual augmentation
* Speech generation and validation
* Automated dataset quality control
* Experimental evaluation and ablation studies

---

## Project Motivation

The central idea behind the project is simple:

> **High-quality machine learning models require high-quality training data, but obtaining that data does not have to rely entirely on manual annotation.**

By combining human labels with programmatic supervision, active learning, LLMs, synthetic data, and automated quality checks, the project explores a practical approach to building larger and more diverse datasets while keeping **label quality and data reliability** at the center of the pipeline.
