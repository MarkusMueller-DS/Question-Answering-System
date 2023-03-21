import pandas as pd
from transformers import pipeline
from elasticsearch import Elasticsearch


# Connection to Elasticsearch
es = Elasticsearch("http://localhost:9200")
if es.ping():
    print("successfully connected to Elasticsearch")
else:
    print("Error: can't connect to Elasticsearch")

# Path to Question-File
TEST_DATA_PATH = "Data/test_sel_blacked_WS2223.txt"
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    test_question_list = [line.strip() for line in f]
print("Test-questions successfully loaded")
    
# Hugging Face question-answering-pipeline with distilbert-base-cased-distilled-squad model
question_answerer = pipeline("question-answering", 
                             model="distilbert-base-cased-distilled-squad")
print("Transformer loaded")

def build_query(query, k):
    """
    Build boosted Elasticsearch-query.

    :param str query: Question
    :param int k: Number of Documents to be returned by Elasticsearch
    :return: Result of Elasticsearch-Query
    """
    my_query = {
        "multi_match": {
            "query": query,
            "fields": ["title^1.1", "text"]
        }
    }
    res = es.search(index="wikibase", query=my_query, size=k)
    return res

def save_scores(res) -> list:
    """
    Save scores from Elasticsearch for each document.

    :param Elasticsearch.elastic_transport.ObjectApiResponse res: Returned Result of ElasticSearch-Query
    :return: List of scores of all returned documents
    """
    score_list = []
    for hit in res["hits"]["hits"]:
        score = hit["_score"]
        score_list.append(score)
    return score_list

def format_answer(ans) -> str:
    """
    Format answer correctly.
    
    :param str ans: Unformatted Answer that still contains fragments and incorrect characters
    :return: Processed answer
    """
    # Remove fragments of wikipedia
    ans = ans.replace("\n", "")
    ans = ans.replace("===", "")
    # Remove leftover puctuation
    if ans[-1] == '.':
        ans = ans.replace(".", "")
    if ans[-1] == ';':
        ans = ans.replace(";", "")
    if ans[-1] == ',':
        ans = ans.replace(",", "")
    # Remove wrong bracketing
    if ans.count("(") != ans.count(")"):
        ans = ans.replace("(", "")
        ans = ans.replace(")", "")
    return ans

def extract_answers(res_es, question) -> dict:
    """
    Generate two answers per document.

    :param Elasticsearch.elastic_transport.ObjectApiResponse res: Returned response of Elasticsearch that contains the documents
    :return: Dictionary of the two answers per question
    """
    answer_dict = {}
    for idx, hit in enumerate(res_es["hits"]["hits"]):
            doc = hit["_source"]["text"]
            # Ignore documents with empty text-field
            if doc == "":
                continue
            # Generate two answers from one document
            res_transformer = question_answerer(question=question, context=doc, top_k=2)
            # Add result of transformer to answer-dictionary
            answer_dict[idx] = res_transformer
    return answer_dict

def prepare_answers_for_ranking(answer_dict, score_es) -> pd.DataFrame:
    """
    Collect and sort answers according to their scores.

    :param dict answer_dict: Dictionary that contains two answers per question
    :param list score_es: list of scores returned by Elasticsearch
    :return: Dataframe containing all answers with their corresponding score
    """
    # Create lists to store answers and scores
    answer_id_list = []
    combined_score_list = []
    answer_list = []
    for k, v in answer_dict.items():
        for i in v:
            answer_id_list.append(k)
            # Combined score is the product of the Elasticsearch-Score and the BERT-Score
            combined_score_list.append(score_es[k]*i["score"])
            answer_list.append(i["answer"])
    # Create DataFrame to store answers with score
    df_answer = pd.DataFrame({"answer_id":answer_id_list, "combined_score":combined_score_list, "answer": answer_list})

    # Sort DataFrame in descending order of the combined score
    df_answer.sort_values("combined_score", ascending=False, inplace=True)
    return df_answer

def question_answering_system(question_list):
    """
    Loops over every question, retrieves five of the most relevant documents for each question from Elasticsearch.
    Generate two answers per document, rank those answers and return it in the right format for the submission.

    :param list question_list: list of questions
    """
    # Counter-variable shows how many questions have been processed
    counter = 0
    # List to store the ranked answers
    question_answer_list = []
    for question in question_list:
        counter += 1
        # Get 5 most relevant documents from Elasticsearch
        res_es = build_query(question, 5)
        # Get the score from the Elasticsearch-documents
        score_es = save_scores(res_es)
        # Extract answers
        answer_dict = extract_answers(res_es, question)
        # Get DataFrame with all scores in descending order       
        df_answer = prepare_answers_for_ranking(answer_dict, score_es)
        # Prepare answers to be processed
        answer_list = [list(df_answer["answer"].values)]
        processed_ans_list = []
        for ans in answer_list[0]:
            ans = format_answer(ans)
            processed_ans_list.append(ans)

        question_answer_list.append(processed_ans_list)
        print(f"{counter} question processed")

    # Create DataFrame with the submission data
    df_submission = pd.DataFrame(question_answer_list)
    # Save DataFrame as csv-file
    df_submission.to_csv("shallow_learning.csv", index=False, sep=";", header=None)
    print("saved submission file")

question_answering_system(test_question_list)