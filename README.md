# Question-Answering-System

Final project of the 5th semester Application of AI course.

### Content

#### Directories

- `Data`: files with the test questions and a file with the training data for the question classification

#### Files

##### Question Answering

- `question_answering.py`: Python script with the whole Question Answering System which is used to create the submission file for scoring `shallow_learning.csv`

##### Misc

- `elastic_search_k.ipynb`: Python notebook with the code and results for determining "precision @ k" to to validate the vaarious Elasticsearch queries
- `question_classification.ipynb`: Python notebook with the code and results for question calssification

### Requirements

External python libraries used for QA-System `question_answering.py`:

- pandas (https://pandas.pydata.org)
- elasticsearch (https://elasticsearch-py.readthedocs.io/en/v8.6.1/)
- transformers (https://huggingface.co/docs/transformers/index)

External python libraries used for Elasticsearch `elastic_serach.ipynb`:

- pandas (https://pandas.pydata.org)
- elasticsearch (https://elasticsearch-py.readthedocs.io/en/v8.6.1/)
- nltk (https://www.nltk.org)

External python libraries used for question classification `question_classification.ipynb`:

- numpy (https://numpy.org)
- pandas (https://pandas.pydata.org)
- sklearn (https://scikit-learn.org/stable/)
