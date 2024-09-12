# Repository for ss24.2.5/team204

**Topic:** SS24 Assignment 2.5: Classify Math Publications

## Requirements

### Programming Language

The assignment was implemented using Python. Python is widely used in machine learning and natural
language processing tasks. I used Python version 3.11, but it should run with any version >3.9.0

### NVIDIA CUDA Toolkit

Even though it is totally optional, I recommend using CUDA to reduce the training time significantly.
With CUDA, we can train our model with GPU instead of CPU and this accelerates the training. For that,
we need an NVIDIA GPU. We can run ``nvidia-smi`` in command prompt to see our CUDA version.
Then we should install PyTorch with CUDA, but we should pay attention to our CUDA version. For example
my CUDA version is 12.6, so I can install PyTorch with CUDA 12.4. Then we can run the following on
command prompt to test if we can use CUDA:

- ``import torch``
- ``torch.cuda.is_available()``

If this returns ``True`` then we can use CUDA.

### Libraries Used

- ``PyTorch``: Used for building and training the neural network.
- ``Transformers``: Used for BERT (Bidirectional Encoder Representations from Transformers), 
a language model that is used in understanding the context of words in text. I used BERT for 
its strong performance in text classification tasks.
- ``scikit-learn (MultiLabelBinarizer)``: Used for handling multi-label classification by 
converting lists of labels into a binary format that can be used during training and prediction.
- ``JSON``: Used for reading and writing data in a structured format.
- ``Torch CUDA (optional)``: For utilizing GPU acceleration to significantly reduce training time.
If a compatible NVIDIA GPU is available, PyTorch with CUDA support is used to perform computations 
on the GPU rather than the CPU.
- ``tqdm``: Used only to make training look better. It provides a progress bar during training, makes
it easier to monitor and estimate the remaining time.
- ``os``: Used for file and path operations such as checking if a file exists or not.

Since I worked with ``.txt`` files, I didn't use `gzip` library.

## Repository Structure

- ``train.txt.gz``: Train dataset that is used to train our model. Even though I used the .txt version,
I uploaded the .gz file because of the .txt file's size.
- ``validation-without-classifications.txt``: File that contains only the titles. Used it to make 
predictions for each title.
- ``validation-solutions.txt``: File that contains the predictions of my model.
- ``validation.txt``: File that contains the actual correct labels.
- ``evaluate.py``: Script that is used to evaluate my model. I compared the `validation-solutions.txt`
with ``validation.txt`` to see how well my model performs using this script.
- ``test-without-classifications.txt``: File that contains the titles needed to be labeled. The assignmet
goal.
- ``solutions.txt``: Solution file for test file. My answers for this assignment.
- ``config.json``: Config file that is needed to communicate with the server.
- ``server_interaction.py``: Script that communicates with the server and receives a list of titles,
then returns a list of labels for each title.
- ``model.py``: Script that creates and trains our model. At the end of each epoch, it saves the progress.
The model gets created as ``bert_model.pth``.
- ``predict.py``: Script that uses the model that has been created by `model.py` to make predictions.
It takes an input file and an output file, then it makes predictions for each title in the input file and 
writes them into the output file.

I didn't include my trained model ``bert_model.pth`` into this repository since it is around 1.2 GB.

## How To Run

First we should run the ``model.py`` script to create and train our model. For that we need
a training dataset, and we should provide the path of this dataset in the script. It might take
long hours to train our model but after that we can start making predictions.

We can use ``predict.py`` script to make predictions. We need an input file that contains 
titles that need classifications. We should provide the path of this file along with an output
file to write our predictions.

If we have a file with correct solutions, we can run ``evaluate.py`` script to evaluate our model
and see how many correct predictions it makes.

``server_interaction.py`` can be run to test our agent performing on the server.

PS: I changed the file encodings with ``utf-8`` in the `evaluation.py` while opening files to
read. I will explain the reason in ``Challenges`` section under `Handling Special Characters`.

## The Problem

The assignment asks us to classify mathematical publication titles into multiple categories.
Each title could belong to multiple categories, thus making it a ``multi-label classification``
problem.

## My Approach

### Model Used

I chose ``BERT (Bidirectional Encoder Representations from Transformers)`` because of its effectiveness 
in understanding the context and semantics of text. BERT’s pre-trained model provides a 
solid foundation that can be fine-tuned on specific tasks, such as text classification, 
with relatively smaller datasets compared to training from scratch.

### Implementation

- **Data Preparation**: I started by reading and preprocessing the data from the provided 
datasets. Titles were tokenized using BERT’s tokenizer, and labels were processed using 
``MultiLabelBinarizer`` to handle the multi-label classification.
- **Model Definition and Training**: I defined a custom classification model based on BERT,
adding a dropout layer for regularization and a fully connected layer to output 
classification logits. The model was trained using ``PyTorch``, with 
``Binary Cross-Entropy Loss`` suitable for multi-label classification tasks.
- **Prediction Script**: A prediction script was developed to take a list of titles from a text file, 
run them through the trained model, and output the top 5 classifications for each title.
- **Evaluation**: An evaluation script was used to compare the model’s predictions 
against the actual validation dataset, providing a measure of the model’s performance. The
server was also used to measure model's accuracy.

### Why This Approach

- **BERT’s Contextual Understanding**: BERT’s ability to capture the context and meanings 
of words made it an ideal choice for this task, where the correct classification often 
depends on understanding complex and specialized language.
- **Multi-Label Handling**: Using ``MultiLabelBinarizer`` allowed for efficient handling 
of multiple labels per title, ensuring that the model’s output aligned with the 
requirements of the assignment.
- **Scalability and Performance**: Leveraging PyTorch with GPU acceleration significantly
reduced training time, making it feasible to work with large datasets and complex 
models efficiently.

## Challenges

- **Handling Special Characters**: I faced some challenges during reading and writing data because
of special characters. So, I used ``utf-8`` encoding while opening files to avoid encoding errors.
- **Long Training Times**: The first time I ran the ``model.py`` script, the estimated training
time was around 90 hours. It was because I was using my CPU instead of GPU. After implementing
CUDA and using my GPU, the estimated time was reduced to 6 hours instead. I also used ``torch.save``
to save my progress between epochs.
- **Cooling Problems**: Since I started using my GPU, my laptop started to see 94-95°C, and I was worried.
For that reason make sure to use either a very good cooling system or a desktop PC instead.



