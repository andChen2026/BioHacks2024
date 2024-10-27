import os
import json
import logging
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import pipeline
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_pinecone import PineconeEmbeddings
from langchain_aws import ChatBedrock
from sentence_transformers import SentenceTransformer
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import time
from datetime import datetime, timedelta
import csv
from collections import Counter
import string
import re
import pandas as pd

# Configuration
CONFIG = {
    "vector_store_k": 20,
    "reranker_top_n": 3,
    "answer_model_temperature": 0.1,
    "pinecone_api_key": "",
    "pinecone_retriver_api_key": "",
    "answer_model": "google/flan-t5-base", # "meta.llama3-1-70b-instruct-v1:0"
    "max_call_per_min": "4",
    "embedding_model": "multilingual-e5-large",# "abhinand/MedEmbed-large-v0.1",  "multilingual-e5-large", "sentence-transformers/all-mpnet-base-v2", #
    "index_name": "hackthon", #  "pitts-sent1", "hackthon"
    "namespace": "clinical_trial", # "pittsburgh",
    "questions_file": "clinical_trial_test.csv",
    "log_dir": "logs"
}
answer_model = CONFIG["answer_model"].replace("/", "-")
embedding_model = CONFIG["embedding_model"].split('.csv')[0]
question_file = CONFIG["questions_file"].split("/")[-1]
CONFIG["eval_output"] = f"eval_output_{answer_model}_{embedding_model}_{question_file}.csv"
CONFIG["output_file"] = f"answers_{answer_model}_{embedding_model}_{question_file}.txt"

# Utility functions
def setup_logging(log_dir):
    """Configure logging with both file and console output."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"qa_session_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def detect_device():
    """Detect the optimal device for computation."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def initialize_pinecone(config):
    """Initialize Pinecone client and embeddings."""
    pc = Pinecone(api_key=config["pinecone_api_key"])
    if config["embedding_model"] == 'sentence-transformers/all-mpnet-base-v2':
      embeddings = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    else:
      embeddings = PineconeEmbeddings(
          model=config["embedding_model"],
          pinecone_api_key=config["pinecone_retriver_api_key"]
      )
    return pc, embeddings


def setup_vector_store(pc, embeddings, config):
    """Set up the vector store and retriever with Cross Encoder Reranker."""

    # Initialize the vector store (Pinecone in this case)
    index = pc.Index(config["index_name"])
    vector_store = LangchainPinecone(
        index,
        embeddings.embed_query,
        "text"
    )

    # Create the base retriever
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": config.get("vector_store_k", 20),
            "namespace": config.get("namespace", "")
        }
    )

    # Initialize the Cross Encoder Reranker model
    reranker_model_name = config.get("reranker_model_name", "BAAI/bge-reranker-base")
    cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model_name)

    # Set up the Reranker with desired parameters
    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=config.get("reranker_top_n", 3)
    )

    # Wrap the base retriever with the ContextualCompressionRetriever to include reranking
    retriever_with_reranker = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    return retriever_with_reranker

# QA System configuration
SYSTEM_PROMPT = """
You are a clinical professional assistant, tell me why the patients list below match the clinical trials.
"""

SYSTEM_PROMPT = """
You are a clinical professional assistant, tell me why the clinical trails list below match the patients.
"""

def generate_prompt(question, context):
    """Generate the full prompt for the model."""
    return f"{SYSTEM_PROMPT}\n\nQ: {question}\nA: "

# Define constants for rate limiting
MAX_CALLS_PER_MINUTE = int(CONFIG["max_call_per_min"])
SECONDS_PER_MINUTE = 60
call_times = []

def rate_limit():
    """Rate limit function to ensure we don't exceed MAX_CALLS_PER_MINUTE"""
    global call_times

    # Get the current time
    now = datetime.now()

    # Remove timestamps older than 1 minute
    call_times = [t for t in call_times if now - t < timedelta(seconds=SECONDS_PER_MINUTE)]

    # If we have already hit the limit of requests, wait
    if len(call_times) >= MAX_CALLS_PER_MINUTE:
        sleep_time = SECONDS_PER_MINUTE - (now - call_times[0]).seconds
        logging.info(f"Rate limit reached. Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

    # Add the current time to the call times
    call_times.append(now)

JSON_LOG_FILE = ''
def predict():
    """Main execution function."""
    try:
        global JSON_LOG_FILE

        # Setup
        log_file = setup_logging(CONFIG["log_dir"])
        device = detect_device()
        logging.info(f"Starting QA session using device: {device}")

        # Initialize components
        pc, embeddings = initialize_pinecone(CONFIG)

        if CONFIG['embedding_model'] == 'sentence-transformers/all-mpnet-base-v2':
            index = pc.Index(CONFIG["index_name"])
        else:
            retriever = setup_vector_store(pc, embeddings, CONFIG)

        if "llama" in CONFIG["answer_model"]:
            qa_model = ChatBedrock(
                model_id=CONFIG["answer_model"],
                model_kwargs=dict(temperature=0),
                aws_access_key_id=CONFIG["aws_access_key_id"],
                aws_secret_access_key=CONFIG["aws_secret_access_key"],
                region_name="us-west-2"
            )
        else:
            qa_model = pipeline(
                "text2text-generation",
                model=CONFIG["answer_model"],
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

        # Load questions
        questions_df = pd.read_csv(CONFIG["questions_file"])

        # Process questions and write answers
        with open(CONFIG["output_file"], "w") as f_out:
            qa_records = []

            for idx, row in tqdm(questions_df.iterrows(), desc="Processing questions", total=len(questions_df)):
                if idx == 5:
                  break
                try:
                    question = row['Question']
                    logging.info(f"Processing question {idx + 1}")

                    # Rate limit the requests
                    if 'llama' in CONFIG['answer_model']:
                        rate_limit()

                    # Get context and generate answer
                    if CONFIG['embedding_model'] == 'sentence-transformers/all-mpnet-base-v2':
                        query_embedding = embeddings.encode(question).tolist()
                        results = index.query(
                            vector=query_embedding,
                            top_k=5,  # Retrieve top 5 most similar results
                            namespace=CONFIG.get("namespace", ""),
                            include_metadata=True  # Include metadata in results
                        )
                        context = " ".join([match['metadata']['text'] for match in results['matches']])
                    else:
                        docs = retriever.get_relevant_documents(question)
                        print(docs)
                        context = " ".join([doc.page_content for doc in docs])
                        context_source =  " ".join([doc.metadata['source'] for doc in docs])


                    prompt = generate_prompt(question, context)

                    if "llama" in CONFIG["answer_model"]:
                        messages = [
                            ("system", prompt)
                        ]
                        answer = qa_model.invoke(messages)
                        generated_text = answer.content
                    else:
                        answer = qa_model(
                            prompt,
                            max_length=150,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            num_return_sequences=1
                        )
                        generated_text = answer[0]['generated_text'].strip()

                    # Write just the answer
                    f_out.write(f"{generated_text}\n")

                    # Store record for JSON log
                    qa_records.append({
                        "id": idx,
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "answer": context_source,
                        "context": context,
                    })

                except Exception as e:
                    error_msg = f"Error on question {idx + 1}: {str(e)}"
                    logging.error(error_msg)
                    f_out.write(f"Error generating answer\n")
                    qa_records.append({
                        "id": idx,
                        "timestamp": datetime.now().isoformat(),
                        "question": question if 'question' in locals() else None,
                        "error": str(e)
                    })

            # Save detailed logs as JSON
            json_log_file = log_file.replace('.log', '.json')
            JSON_LOG_FILE = json_log_file
            with open(json_log_file, 'w', encoding='utf-8') as f_json:
                json.dump({
                    "qa_session": {
                        "timestamp": datetime.now().isoformat(),
                        "total_questions": len(questions_df),
                        "successful_responses": len([r for r in qa_records if "error" not in r]),
                        "records": qa_records
                    }
                }, f_json, ensure_ascii=False, indent=2)

        logging.info(f"QA session completed. Answers written to {CONFIG['output_file']}")
        logging.info(f"Detailed logs saved to {json_log_file}")

    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise

predict()

"""## Evaluation"""



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(ground_truth) in normalize_answer(prediction))

def process_files(questions_answers_csv, json_log_file, output_csv):
    # Read the questions and answers (ground truth) from CSV
    # with open(questions_answers_csv, mode='r', newline='', encoding='utf-8') as qa_file:
    #     reader = csv.DictReader(qa_file)
    #     questions_answers = [(row['Question'], row['Answer']) for row in reader]

    questions_answers = []
    with open(questions_answers_csv, mode='r', newline='', encoding='utf-8') as qa_file:
        reader = csv.DictReader(qa_file)
        for i, row in enumerate(reader):
            if i >= 5:  # Stop after 100 lines
                break
            questions_answers.append((row['Question'], row['Answer']))

    # Read predictions from the JSON log file
    with open(json_log_file, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        qa_records = data.get("qa_session", {}).get("records", [])
        predictions = [record.get('answer', 'Error generating answer') for record in qa_records]
        contexts = [record.get('context', '') for record in qa_records]

    # Ensure the number of predictions matches the number of questions
    if len(predictions) != len(questions_answers):
        raise ValueError("The number of predictions does not match the number of questions.")

    # Create the output CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as output_file:
        fieldnames = ['question', 'answer', 'prediction', 'f1_score', 'exact_match','context']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        # Calculate F1 and Exact Match for each question-answer pair
        for (question, answer), prediction, context in zip(questions_answers, predictions, contexts):

            f1 = f1_score(prediction, answer)
            exact_match = exact_match_score(prediction, answer)
            writer.writerow({
                'question': question,
                'answer': answer,
                'prediction': prediction,
                'f1_score': f1,
                'exact_match': exact_match,
                'context': context
            })

questions_answers_csv = CONFIG['questions_file']  # Input CSV file with questions and ground truth answers
json_log_file = f'./{JSON_LOG_FILE}'  # Predictions stored in the JSON log file
output_csv = CONFIG['eval_output']  # Output CSV file

process_files(questions_answers_csv, json_log_file, output_csv)

# Output score

def calculate_f1_mean(output_csv):
    df = pd.read_csv(output_csv)
    mean_f1 = df['f1_score'].mean()
    return mean_f1

def calculate_exact_match_mean(output_csv):
    df = pd.read_csv(output_csv)
    mean_exact_match = df['exact_match'].mean()  # True will be 1, False will be 0
    return mean_exact_match

output_csv = CONFIG['eval_output']  # Output CSV file from previous step

mean_f1_score = calculate_f1_mean(output_csv)
mean_exact_match = calculate_exact_match_mean(output_csv)
print(f"Mean F1 Score: {mean_f1_score}")
print(f"Mean Exact Match: {mean_exact_match}")