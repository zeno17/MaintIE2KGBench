import argparse
import os
import json

from time import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from langchain_chroma import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.llms import VLLMOpenAI

class TLPExtraction:

    def __init__(self, args):
        #https://api.python.langchain.com/en/latest/llms/langchain_community.llms.vllm.VLLMOpenAI.html
        self.client = VLLMOpenAI(api_key="local",
                                 base_url="http://0.0.0.0:{}/v1".format(args.port),
                                 model_name=args.model,
                                 temperature=args.temperature,
                                 frequency_penalty=args.frequency_penalty,
                                 presence_penalty=args.presence_penalty)
                                 #model_kwargs={"stop": [128001, 128009]})

        print("Running the following model with following client settings")
        print(self.client)

        self.data_file = args.data_file
        self.output_folder = args.output_folder
        self.output_file = args.output_file
        self.num_fewshot = args.num_fewshot

        self.data = None
        self.prompt = None

        self.train_data = None
        self.test_data = None
        self.load_data()

    def run_extraction(self):
        with open(os.path.join(self.output_folder, self.output_file), 'w') as file:
            example_selector = SemanticSimilarityExampleSelector.from_examples(self.train_data,
                                                                               HuggingFaceEmbeddings(model_name="all-mpnet-base-v2"),
                                                                               Chroma,
                                                                               k=self.num_fewshot)
            for i, item in enumerate(tqdm(self.test_data)):
                selected_fewshot_examples = example_selector.select_examples({"text": item['text']})
                result = self.single_extract(target=item, 
                                             examples=selected_fewshot_examples)
                # print(i, result['target_answer'])
                # for example in selected_fewshot_examples:
                #     print('example: ', example)
                # print('generated text: ', repr(result['generated_text']))
                json_data = json.dumps(result)
                file.write(json_data + '\n')
                #examples_idx = random.sample([j for j in range(len(self.data)) if j != i], self.num_fewshot)
                #selected_fewshot_examples = [self.data[i] for i in examples_idx]

    def single_extract(self, target, examples):
        #target: is the sentence we want to extract information from
        #examples: are the examples we want to feed to the model (few shot)
        base_prompt = """Given the following ontology and sentence, please extract the triples from the sentence according to the relations in the ontology. 
        In the output, only include the triples in the given output format.

        Context:  
        Ontology Concepts: Activity, PhysicalObject, Process, Property, State
        Ontology Relations: contains, hasPart, hasAgent, hasPatient, hasProperty, isA"""

        example_template = """Example sentence: {text}
        Example answer: {answer}"""

        example_prompt = PromptTemplate(
            input_variables=["text", "answer"], 
            template=example_template
        )

        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=base_prompt,
            suffix="Given these examples, give the correct answer for the following sentence:\nTest sentence: {input}\nTest answer: ",
            input_variables=["input"],
        )

        model_input = prompt.format(input=target['text'])

        start_time = time()
        output = self.client.invoke(model_input)
        end_time = time()
        duration = end_time - start_time
        
        return {'target': target['text'] ,
                'target_answer': target['answer'],
                'generated_text': output,
                'prompt': model_input,
                'time_taken (seconds)': duration}

    def load_data(self):
        print("Loading data")
        if self.data_file.endswith(".json"):
            with open(self.data_file, "r") as json_file:
                lines = json_file.read()
                self.data = json.loads(lines)

            self.data = self.restructure_examples(self.data)

        #Random state set for the telephone area code of Eindhoven, The Netherlands :)
        self.train_data, self.test_data = train_test_split(self.data, random_state=40)

        print("Loaded data, example looks as follows:")
        print("From train data: ", self.train_data[0])
        print("From test data: ", self.test_data[0])

    def restructure_examples(self, source_data):
        """
        Option 1: Paper of Text2KGBench, relation(object1, object2)
        Option 2: Paper of MaintIE converting original string into simplified
        - Convert CamelCase format strings like "PhysicalObject/EmittingObject/ElectricCoolingObject" into a format like "<electric cooling object>".

        """

        data = []
        for entry in tqdm(source_data):
            if len(entry['relations']) == 0:
                continue
            entities = []
            tokens = entry['tokens']
            for entity in entry['entities']:
                start_id = entity['start']
                end_id = entity['end']
                entities.append({'entity_string': ' '.join(tokens[start_id:end_id]), 
                                 'entity_type': entity['type']})
            data.append({"text": entry['text'],
                        "answer": '\n'.join(["{}({},{})".format(relation['type'],
                                                                 entities[relation['head']]['entity_string'],
                                                                 entities[relation['tail']]['entity_string']) for relation in entry['relations']])})  
        return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--data_file', type=str, default="./data/silver_release.json")
    parser.add_argument('--output_folder', type=str, default='./predictions/')
    parser.add_argument('--output_file', type=str, default='data.ndjson')
    parser.add_argument('--num_fewshot', type=int, default=3)
    parser.add_argument('--port', type=int, default=8003)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--frequency_penalty', type=float, default=0)
    parser.add_argument('--presence_penalty', type=float, default=-1)
    
    args = parser.parse_args()

    process = TLPExtraction(args)
    
    process.run_extraction()