# Building a Fast and Accurate Dependency Parser

This project implements a neural **transition-based dependency parser**. The design is heavily inspired by the highly efficient approach detailed in the paper, *A Fast and Accurate Dependency Parser using Neural Networks*.

Dependency parsing is a core challenge in Natural Language Processing (NLP) that aims to map out the grammatical relationships (who modifies whom) within a sentence. Because building a system like this is complex, the project is structured into modular steps—from data preparation and feature engineering to model training and evaluation—allowing for manageable, incremental development.

## Project Structure and Implementation Steps

The implementation is broken down into four main phases:

### 1. Data Processing and Setup

The first phase prepares the raw corpus for the neural network. This involves:

* **Data Loading:** Using the **UD\_English** corpus (version 1.4) from the Universal Dependencies project.
* **Vocabulary Initialization:** Creating the **`Vocabulary` class** to map every unique word, POS tag, and dependency label to a unique integer ID. This also defines the complete set of $2n+1$ possible transition actions (Shift, Left-Arc, Right-Arc, each with its labels).
* **Core Data Structures:** Implementing the essential components of the transition system: the **`Stack`** (for words being processed) and the **`Buffer`** (for remaining words).

### 2. Training Data Generation (The Oracle)

Since the parser needs supervision at every decision point, this phase transforms the static 'gold' dependency trees into sequences of training examples:

* **Implementing the Oracle:** The **`get_gold_action`** function acts as the "teacher," determining the single correct next transition (Shift, Left-Arc, or Right-Arc) for any given parser state, based on the final correct dependency structure.
* **Feature Engineering:** Implementing the **`extract_features`** functions. The network requires a fixed-size vector of **48 features** (Word IDs, POS IDs, and Label IDs) pulled from the top words on the stack and buffer, as well as their children and grandchildren, to make its prediction.

### 3. Model Definition and Training

This phase defines and trains the core predictive component:

* **Defining the Model:** The parser uses a simple, fast **feed-forward neural network** architecture consisting of an embedding layer, a single hidden layer, and an output layer. The hidden layer uses a cube activation function ($g(x) = x^3$).
* **Dataset Integration:** Using a PyTorch **`Dataset`** to efficiently handle the large collection of feature vectors and gold actions generated from the training data.
* **Training:** Executing the training loop, optimizing the model's weights to predict the correct transition action at every step.

### 4. Evaluation and Analysis

The final phase tests the trained model's performance on unseen sentences:

* **Inference Logic:** Implementing the **`select_best_legal_action`** function. At test time, the model must only choose structurally **legal** transitions (e.g., preventing a Shift when the buffer is empty), even if an illegal action has the highest score.
* **Attachment Score:** Measuring performance using standard metrics:
    * **Unlabeled Attachment Score (UAS):** Percentage of words with the correct head.
    * **Labeled Attachment Score (LAS):** Percentage of words with both the correct head and the correct label.
* **Qualitative Analysis:** Visualizing the predicted dependency trees against the gold trees to diagnose specific error patterns and understand the model's strengths and weaknesses.

***

## Running the Project

**Note on Runtime:** For efficient development and debugging, it's recommended to leverage a **CPU** runtime until the model is fully prepared. The transition to a **GPU** runtime should be reserved for the final training phase to maximize performance. Debugging on CPU often provides clearer error messages and a faster development cycle.

### Data and Dependencies

The project relies on external dependency treebank files loaded dynamically:

* Training Data: `https://drive.google.com/uc?export=download&id=1N4B4bC4ua0bFMIYeNdtNQQ72LxxKLPas`
* Test Data: `https://drive.google.com/uc?export=download&id=1TE2AflhABbz41dLMGmD1kTm7WevYpU31`
* Testing Bundle: `https://drive.google.com/uc?export=download&id=1-9lWbXuXZYGjJWQRKD7um_83O-I3ER63`

The necessary Python classes and data structures are defined within the notebook itself, followed by the specific functions for each step.
