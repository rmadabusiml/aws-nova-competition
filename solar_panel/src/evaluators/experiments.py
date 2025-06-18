from datetime import datetime
import time
import re
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from llm_eval import nova_micro_llm, nova_lite_llm, nova_pro_llm
from score_eval import score_bleu, score_rouge
import nltk

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
langfuse = Langfuse()
# nltk.download('punkt_tab') # this needs to be run first time to download tokenizers for scoring

@observe()
def extract_energy_company_name(prompt_input, model):
  prompt_template = """{prompt_input}"""

  # print(prompt_input)

  PROMPT = PromptTemplate(template=prompt_template, input_variables=["prompt_input"])
  chain = PROMPT | model
  result = chain.invoke(input=prompt_input)
  # print(result)
  # print(result.content)
  original_question = result.content
  return original_question

@observe()
def run_experiment(experiment_name=None, model_id=None, model=None, dataset=None):
  # print(dataset)
  dataset = langfuse.get_dataset(dataset)

  for item in dataset.items:
    print(item)
    generationStartTime = datetime.now()
    print(item.input)
    llm_input = item.input

    expected_output = item.expected_output
    print(expected_output)
    start_time = time.time()
    generationStartTime = datetime.fromtimestamp(start_time)
    llm_output = extract_energy_company_name(llm_input, model)
    end_time = time.time()
    generationEndTime = datetime.fromtimestamp(end_time)
    print(llm_output)

    langfuse_generation = langfuse.generation(
      name=item.id,
      input=item.input,
      output=llm_output,
      model=model_id,
      start_time=generationStartTime,
      end_time=generationEndTime
    )
    
    langfuse_context.flush()
    time.sleep(3)

    rouge_score = score_rouge("rougeL", llm_output, expected_output)
    time.sleep(2)
    item.link(langfuse_generation, experiment_name)
    langfuse_context.flush()
    time.sleep(2)

    langfuse_generation.score(
      name="rouge-L",
      value=rouge_score
    )
    time.sleep(2)
    

eval_llms_dict = {
    "amazon.nova-micro-v1:0": nova_micro_llm,
    "amazon.nova-lite-v1:0": nova_lite_llm,
    "amazon.nova-pro-v1:0": nova_pro_llm,
}

dataset_name = "energy-company-name-dataset"

# Loop through the dictionary and use model_id and model_name
for model_id, model in eval_llms_dict.items():
    experiment_name = f"{model_id}-{datetime.now()}"
    run_experiment(experiment_name=experiment_name, model_id=model_id, model=model, dataset=dataset_name)