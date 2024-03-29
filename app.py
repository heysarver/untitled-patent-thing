import argparse
import logging
import torch
from transformers import pipeline
from elasticsearch import Elasticsearch
from pprint import pprint

logging.basicConfig(level=logging.DEBUG)

def parse_arguments():
    logging.debug('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='anferico/bert-for-patents', help="Huggingface model to use")
    parser.add_argument('--es-host', default='localhost', help="Elasticsearch host")
    parser.add_argument('--es-port', default=9200, type=int, help="Elasticsearch port")
    parser.add_argument('--es-user', default='elastic', help="Elasticsearch user")
    parser.add_argument('--es-password', default='password', help="Elasticsearch password")
    parser.add_argument('--es-scheme', default='https', help="Elasticsearch scheme")
    parser.add_argument('--es-index-name', default='data', help="Elasticsearch index name")
    return parser.parse_args()

def sanitize_inputs(args):
    logging.debug('Sanitizing inputs')
    model = args.model.replace('"', '').replace("'", "")
    es_host = args.es_host.replace('"', '').replace("'", "")
    index_name = args.es_index_name.replace('"', '').replace("'", "")
    return model, es_host, index_name

def connect_elasticsearch(es_host, args):
    logging.debug('Connecting to Elasticsearch')
    return Elasticsearch([{'host': es_host, 'port': args.es_port, 'http_auth': (args.es_user, args.es_password), 'scheme': args.es_scheme}])

def load_model(model):
    logging.debug('Loading Huggingface model')
    return pipeline('text-generation', model=model)

def generate_results(model_pipeline, prompt, max_new_tokens=1024):
    logging.debug('Generating results with model')
    return model_pipeline(prompt, max_new_tokens=max_new_tokens)

def store_in_elasticsearch(es, index_name, data):
    logging.debug('Storing data in Elasticsearch')
    for item in data:
        pprint(item)
        #es.index(index=index_name, document=item)

def main():
    logging.debug('Starting main function')
    args = parse_arguments()
    model, es_host, index_name = sanitize_inputs(args)
    model_pipeline = load_model(model)
    data = generate_results(model_pipeline, "Patent", 1024)
    pprint(data)
    # es = connect_elasticsearch(es_host, args)
    # store_in_elasticsearch(es, index_name, data)

if __name__ == "__main__":
    main()
