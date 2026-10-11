# SentimentScope: Transformers for Movie Review Sentiment Analysis

A deep learning project that implements a custom Transformer-based neural network to classify IMDB movie reviews as positive or negative. The project explores self-attention, token embeddings, positional embeddings, and multi-head attention for natural language processing and sentiment classification.

## Project Overview

SentimentScope demonstrates how Transformer architecture can be implemented and trained for a real-world text classification task. Using the IMDB movie review dataset, the model learns to identify sentiment patterns in written reviews and predicts whether a review expresses a positive or negative opinion.

The model is implemented using PyTorch, while the `bert-base-uncased` tokenizer is used to tokenize and encode text. The Transformer model itself is trained from scratch rather than using pretrained BERT model weights.

## Project Objectives

* Implement a custom Transformer architecture for sentiment classification.
* Understand the roles of token embeddings and positional embeddings.
* Explore causal self-attention and multi-head attention.
* Train and evaluate a neural network on movie review data.
* Measure model performance using validation and test accuracy.
* Apply Transformer concepts to a practical natural language processing task.

## Dataset

The project uses the **IMDB Large Movie Review Dataset**, containing 50,000 movie reviews divided into positive and negative sentiment classes.

* **Training dataset:** 25,000 reviews
* **Test dataset:** 25,000 reviews
* **Training split:** 22,500 reviews
* **Validation split:** 2,500 reviews
* **Classification task:** Binary sentiment classification
* **Positive label:** 1
* **Negative label:** 0

The training dataset was split into training and validation subsets to monitor model performance during training.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers tokenizer
* NumPy
* Jupyter Notebook / Google Colab
* IMDB Movie Review Dataset

## Model Architecture

The project implements a custom Transformer-based classifier with the following components:

1. **Token Embeddings:** Converts token IDs into dense vector representations.
2. **Positional Embeddings:** Adds information about token positions.
3. **Self-Attention:** Enables the model to learn relationships between tokens.
4. **Multi-Head Attention:** Allows the model to learn different representation patterns simultaneously.
5. **Feed-Forward Networks:** Transform token representations within each Transformer block.
6. **Layer Normalization:** Supports stable model training.
7. **Mean Pooling:** Aggregates token representations into a review-level representation.
8. **Classification Head:** Produces logits for the positive and negative sentiment classes.

### Model Configuration

| Parameter                    |              Value |
| ---------------------------- | -----------------: |
| Embedding dimension          |                128 |
| Number of Transformer blocks |                  4 |
| Attention heads              |                  4 |
| Head size                    |                 32 |
| Maximum sequence length      |                128 |
| Dropout rate                 |                0.1 |
| Batch size                   |                 32 |
| Optimizer                    |              AdamW |
| Learning rate                |             0.0003 |
| Training epochs              |                  3 |
| Loss function                | Cross-Entropy Loss |

## Results

The model was evaluated on the validation and test datasets after training for three epochs.

| Evaluation Metric           |   Accuracy |
| --------------------------- | ---------: |
| Initial validation accuracy |     51.04% |
| Epoch 1 validation accuracy |     69.88% |
| Epoch 2 validation accuracy |     75.72% |
| Final validation accuracy   | **77.92%** |
| Final test accuracy         | **76.54%** |

The final test accuracy exceeded the project target of 75%, demonstrating that the custom Transformer learned useful sentiment representations from the IMDB movie reviews.

## Key Takeaways

* **Transformer implementation:** Building the architecture from scratch provided practical experience with embeddings, attention mechanisms, feed-forward networks, and classification layers.
* **Learning progress:** Validation accuracy increased from 51.04% before training to 77.92% after three epochs.
* **Generalization:** The model achieved 76.54% test accuracy on unseen reviews, exceeding the target of 75%.
* **Evaluation matters:** Monitoring validation performance and evaluating on a separate test set helped assess how well the model generalized beyond its training data.

## How to Run the Project

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   ```

2. Navigate to the project directory:

   ```bash
   cd YOUR_REPOSITORY
   ```

3. Install the required packages:

   ```bash
   pip install torch transformers numpy jupyter
   ```

4. Download and extract the IMDB dataset. The notebook expects the extracted dataset to be available in the `aclImdb` directory, with `train` and `test` folders containing `pos` and `neg` subfolders.

5. Open the notebook in Jupyter or upload it to Google Colab.

6. Run the notebook cells in order to preprocess the reviews, build the model, train it, and evaluate the results.

**Note:** Adjust the dataset path and installation commands if your notebook uses a different environment or setup.

## Future Improvements

Potential improvements include:

* Experimenting with additional epochs and hyperparameter tuning.
* Comparing performance with a pretrained Transformer model.
* Improving padding and attention-mask handling during pooling.
* Tracking precision, recall, F1-score, and a confusion matrix.
* Testing alternative sequence lengths and embedding dimensions.
* Evaluating model performance across different types of text.

## Project Report

The notebook includes a project report summarizing the dataset, model performance, evaluation results, key takeaways, and possible future improvements.

## Acknowledgements

* [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
* [PyTorch](https://pytorch.org/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## License

This project is available for educational and portfolio purposes. Add a license file to the repository if you intend to specify formal reuse and distribution permissions.

---

**Project:** SentimentScope — Transformers for Movie Review Sentiment Analysis
**Task:** Binary sentiment classification
**Final validation accuracy:** 77.92%
**Final test accuracy:** 76.54%
