from flashrag.generator import BaseGenerator
from flashrag.pipeline import SequentialPipeline, BasicPipeline
from typing import List
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
from flashrag.evaluator import Evaluator

from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
import json

import flashrag

import vllm
from vllm import LLM, SamplingParams

import json
import os

def write_jsonl(list_of_dicts, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for d in list_of_dicts:
            json_line = json.dumps(d, ensure_ascii=False)
            file.write(json_line + '\n')
            
def read_jsonl(file_path):
    list_of_dicts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line.strip())
            list_of_dicts.append(d)
    return list_of_dicts


class DummyGenerator(BaseGenerator):
    """Class for a dummy generator that directly outputs the input texts."""

    def __init__(self, config):
        
        super().__init__(config)
        
        # No additional initialization needed for the dummy generator.
        return 

    def generate(self, input_list: List[str], **params) -> List[str]:
        """Directly return the input list as the output.

        Args:
            input_list: A list of input texts.

        Returns:
            List[str]: The same list of input texts.
        """
        if isinstance(input_list, str):
            input_list = [input_list]
        return input_list
    

class DummyPipeline(SequentialPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """
        BasicPipeline.__init__(self, config, prompt_template)
        
        self.generator = DummyGenerator(config)
        
        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever
    
        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

# r = {
#     'succeed': bool,
#     'response': str,
#     'reflection': str,
#     'confidence': int,
# }
def process_reflection_response(idx:int, response:str):
    SPLIT = "<<<SPLIT114514>>>"
    
    r = {
            'idx': idx,
            'succeed': False,
            'response': response,
            'extra_query': '',
            'confidence': -1,
        }
    if not (("Question: " in response) and ("Confidence: " in response)):
        print(f"{idx}: Error happened in making reflection for idx - Skipped")
        return r
        
    _response = response.replace("Question: ", SPLIT).replace("Confidence: ", SPLIT)
    parts = _response.split(SPLIT)
    if len(parts) != 3:  
        print(f"{idx}: Error happened in spliting reflection for idx - Skipped")
        return r  
    
    _, confidence_str, question = parts
    question = question.strip()
    try:
        confidence = int(confidence_str.strip())  
    except ValueError:  
        print(f"{idx}: Error happened in converting confidence to int for idx - Skipped")
        return r
    
    # 如果响应格式正确且信心评分有效，将其添加到结果列表中
    r = {
        'idx': idx,
        'succeed': True,
        'response': response,
        'extra_query': question,
        'confidence': confidence,
    }
    return r


def generate_reflections(reflection_model: LLM, sampling_params,
                     dataset:flashrag.dataset.dataset.Dataset, split = 'dev', key='question', 
                     output = './reflection.jsonl',
                     max_reroll=3):
    
    
    queries = dataset[split].__getattribute__(key)
    outputs = reflection_model.generate(queries, sampling_params = sampling_params)

    responses = [output.outputs[0].text for output in outputs]
    results = [process_reflection_response(idx, response) for idx, response in enumerate(responses)]
    
    for reroll in range(max_reroll):
        
        failed_idx = [r['idx'] for r in results if not r['succeed']]
        if not failed_idx: break

        # Regenerate outputs for failed queries
        failed_queries = [queries[idx] for idx in failed_idx]
        new_outputs = reflection_model.generate(failed_queries,sampling_params=sampling_params, )

        # Update responses and results for failed indices
        new_responses = [output.outputs[0].text for output in new_outputs]
        for i, idx in enumerate(failed_idx):
            results[idx] = process_reflection_response(idx, new_responses[i])
    
    for q, r in zip(queries, results):
        r['query'] = q
    
    write_jsonl(results, output)
    return results

class SayRAGPipeline(SequentialPipeline):
    
    
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template, retriever, generator)
        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )
    
    def get_prompt(self, question, retrieval_result, extra_retrieval_result,
                       reflection_result, min_confidence_run_naive, min_confidence_drop_reflection,
                       topk:int, augument_num:int):
            
            if (not reflection_result['succeed']) or (reflection_result['confidence'] >= min_confidence_drop_reflection):
                prompt = self.prompt_template.get_string(question=question, retrieval_result=retrieval_result)
                return prompt

            if reflection_result['confidence'] >= min_confidence_run_naive:
                prompt = self.zero_shot_templete.get_string(question=question)
                return prompt
            
            query_retrieval_num = topk - augument_num
            query_retrieval = retrieval_result[:query_retrieval_num]
            extra_retrieval_result = [r for r in extra_retrieval_result if r not in query_retrieval]
            augument_num = min(len(extra_retrieval_result), augument_num)
            
            query_retrieval_num = topk - augument_num
            query_retrieval = retrieval_result[:query_retrieval_num]
            reflection_retrieval = extra_retrieval_result[:augument_num]
            augumented_result = query_retrieval + reflection_retrieval
            prompt = self.prompt_template.get_string(question=question, retrieval_result=augumented_result)
            
            return prompt
    
    def run(self, dataset, do_eval=True, pred_process_fun=None,
            reflection_path:str='./reflection.jsonl', 
            augument_ratio:float=1.0,
            min_confidence_run_naive:int=114514, 
            min_confidence_drop_reflection:int=10,
            ):
        
        input_query = dataset.question
        
        assert os.path.isfile(reflection_path)
        all_reflections = read_jsonl(reflection_path)
        all_reflections = {ref['idx']:ref for ref in all_reflections}

        idxs = [int(idx.split('_')[1]) for idx in dataset.id]
        reflections = [all_reflections[idx] for idx in idxs]
        
        extra_query = [r['extra_query'] for r in reflections]
        dataset.update_output("extra_query", extra_query)
        
        extra_retrieval_result = [ref['retrieval_result'] for ref in reflections]
        dataset.update_output("extra_retrieval_result", extra_retrieval_result)
            
        retrieval_result = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_result)
        
        augument_num = int(self.retriever.topk * augument_ratio)
            
        input_prompts = [
            self.get_prompt(question, retrieval_result, extra_retrieval_result, reflection_result, 
                            min_confidence_run_naive, min_confidence_drop_reflection, 
                            self.retriever.topk, augument_num)
            for question, retrieval_result, extra_retrieval_result, reflection_result
            in zip(dataset.question, 
                   dataset.retrieval_result,
                   dataset.extra_retrieval_result,
                   reflections)
        ]
        dataset.update_output("prompt", input_prompts)

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc for doc in docs])
        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset