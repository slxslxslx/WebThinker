import re
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import string
import os, time
from collections import defaultdict
# from lcb_runner.evaluation import codegen_metrics
import sys
sys.path.append('./scripts/utils')
from math_equivalence import is_equiv
from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List


# 从原始输出中"抽提"答案
def extract_answer_fn(output, mode='qa', extract_answer=False):
    print(f"开始执行 extract_answer_fn，参数：output={output}")
    print(f"开始执行 extract_answer_fn，参数：mode={mode}")
    print(f"开始执行 extract_answer_fn，参数：extract_answer={extract_answer}")
    if extract_answer == False and mode not in ['infogen', 'summary', 'research']:  # qa 模式且 extract_answer=False 时的特殊处理
        if mode == 'qa':  # # 直接返回完整输出
            return output.strip()
        pred_answer_lines = output.replace("\n\n", "\n").strip().split('\n')
        pred_answer = '\n'.join(pred_answer_lines[-3:])
        return pred_answer
    extracted_text = ''
    if mode == 'codegen':
        pattern = r'```python\s*(.*?)\s*```'  # Extract the code between ```python and ```
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode in ['infogen', 'summary', 'research']:  # （信息生成/摘要/研究）
        pattern_info = "**Final Information"
        if "</think>\n" in output:
            extracted_text = output.split("</think>\n")[-1].split("<|begin_click_link|>")[0].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()  # 提取</think>后面的内容
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
        elif pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].split("<|begin_click_link|>")[0].strip('\n').strip(':**').strip("```").strip()  # 提取**Final Information**后面的内容
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
        else:
            # extracted_text = "No helpful information found."
            extracted_text = '\n'.join(output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])  # 若没提取到，只保留最后5行
        if mode == 'research':
            extracted_text = extracted_text[:6000]
        else:
            extracted_text = extracted_text[:2500]
    elif mode in ['math', 'choose', 'qa']:  # （数学/选择/问答）
        extracted_text = ''
        pattern = r'\\boxed\{(.*)\}'  # # 优先匹配 \boxed{...}
        matches = re.findall(pattern, output)
        print(f"extract_answer_fn 里面的 matches={matches}")
        if matches:
            print(f"extract_answer_fn 里面的 matches={matches} 进入 if matches")
            extracted_text = matches[-1]  # Take the last match
        # else:  # # 其次匹配 ANSWER: 后面的内容
        #     pattern = 'ANSWER:'
        #     if pattern in output:
        #         print(f"extract_answer_fn 里面的 matches={matches} 进入 if pattern = ANSWER:  if pattern in output")
        #         extracted_text = output.split(pattern)[-1].strip('**').strip()
        else:
            # 增加 "Final Answer:" 匹配
            for ans_pattern in ['Final Answer:', 'final answer:', 'FINAL ANSWER:', 'ANSWER:', 'Answer:']:
                if ans_pattern in output:
                    extracted_text = output.split(ans_pattern)[-1].strip('**').strip().strip('.')
                    break
        if mode in ['choose']:  # # choose 模式额外处理 \text{} 和括号
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
    print(f"extract_answer_fn 处理后结果，extracted_text={extracted_text}")
    return extracted_text


# 用 LLM 判断答案是否等价
# 当 use_llm=True 时，系统会调用另一个 LLM（如 Qwen2.5-72B）来判断预测答案与标准答案是否语义等价。
async def llm_evaluate_equivalence_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
    extract_answer: bool = False,
) -> bool:
    """Evaluate a single pair of answers using LLM"""
    print(f"开始执行llm_evaluate_equivalence_single ")

    if extract_answer:  # 评估 Prompt
        prompt = f"""You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Predicted Answer: {pred_answer}

Are these answers equivalent? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""
    else:
        prompt = f"""You are an evaluation assistant. Please determine if the model output is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Model Output (Last few lines): {pred_answer}

Did the model give an answer equivalent to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                chat_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = chat_response.choices[0].message.content.strip()
                print(f"llm_evaluate_equivalence_single里面的 response_text={response_text}")
                print(f"llm_evaluate_equivalence_single里面的 pred_answer={pred_answer}")
                print(f"llm_evaluate_equivalence_single里面的 labeled_answer={labeled_answer}")
                print(f"llm_evaluate_equivalence_single里面的 is_equiv(pred_answer, labeled_answer)={is_equiv(pred_answer, labeled_answer)}")
                # 数学等价 或 LLM 明确说 Correct 且没有否定词。
                # llm_judge = is_equiv(pred_answer, labeled_answer) or \  # 太严格了
                #     response_text.lower() == "correct" and \
                #     not ("incorrect" in response_text.lower() or \
                #          "wrong" in response_text.lower() or \
                #          "not correct" in response_text.lower())
                resp_low = response_text.lower()
                print(f"llm_evaluate_equivalence_single里面的 resp_low={resp_low}")
                llm_judge = is_equiv(pred_answer, labeled_answer) or (
                    "correct" in resp_low and
                    not ("incorrect" in resp_low or "wrong" in resp_low or "not correct" in resp_low)
                )
                print(f" llm_evaluate_equivalence_single 的  response_text={response_text}")
                print(f" llm_evaluate_equivalence_single 的  llm_judge={llm_judge}")
                return llm_judge, response_text
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Error in LLM evaluation: {e}")
                return is_equiv(pred_answer, labeled_answer), "Error"
            await asyncio.sleep(1 * (attempt + 1))
    
    return is_equiv(pred_answer, labeled_answer), "Error"


async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str], 
    pred_answers: List[str],
    api_base_url: str = None,
    model_name: str = None,
    api_key: str = "empty",
    concurrent_limit: int = 50,
    extract_answer: bool = False
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """
    if api_base_url is None:
        api_base_url = None
    if model_name is None:
        # model_name = "Qwen2.5-72B-Instruct"
        model_name = "qwen3.5-9b"

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )

    semaphore = asyncio.Semaphore(concurrent_limit)
    
    tasks = [
        llm_evaluate_equivalence_single(
            client=client,
            question=q,
            labeled_answer=l,
            pred_answer=p,
            model_name=model_name,
            semaphore=semaphore,
            extract_answer=extract_answer
        )
        for q, l, p in zip(questions, labeled_answers, pred_answers)
    ]

    with tqdm(total=len(tasks), desc="LLM Evaluation") as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)
    
    return results


# 计算 EM/Acc/F1/Math 等指标
def evaluate_predictions(output, labeled_answer, mode='math', use_llm=False, question=None, extract_answer=False):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0, 'llm_equal': 0} # 2. 初始化指标字典
    pred_answer = extract_answer_fn(output, mode=mode, extract_answer=extract_answer)  # 调用 extract_answer_fn() 提取 Pred_Answer
    pred_answer_new = pred_answer
    if pred_answer != '':  # 判断答案是否有效
        print(f"evaluate.py里面 extract_answer_fn 得到的 pred_answer有效={pred_answer} ")
        final_metric["is_valid_answer"] = True
    else: # 输出最后 5 行
        print(f"❌❌❌evaluate.py里面 extract_answer_fn 得到的 pred_answer 无效={pred_answer} ")
        # If no answer was extracted, keep only the last 3 lines
        pred_answer_new = '\n'.join(output.replace("\n\n", "\n").strip().split('\n')[-5:])

    if mode in ['qa']:
        def normalize_answer_qa(s):  #  # 去冠词(a/an/the) → 去标点 → 转小写 → 合并空格
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)
            def white_space_fix(text):
                return " ".join(text.strip().split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        normalized_pred_answer = normalize_answer_qa(pred_answer_new)

        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    elif mode in ['math', 'choose']:
        def normalize_answer(text):  # # 仅小写+合并空格
            text = text.lower()
            text = " ".join(text.strip().split())
            return text
        normalized_pred_answer = normalize_answer(pred_answer_new)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)
        
        # Add LLM-based evaluation if requested
        if use_llm and question is not None:
            final_metric["llm_equal"] = 0  # Will be updated in batch later

    return final_metric, pred_answer


# 主流程：遍历数据、调用上述函数
# def run_evaluation(filtered_data, input_list, output_list, task_type, output_dir, output_metrics_path, output_metrics_overall_path, use_llm=False, extract_answer=False, domain_fields=None, api_base_url=None, model_name=None):
async def run_evaluation(filtered_data, input_list, output_list, task_type, output_dir, output_metrics_path, output_metrics_overall_path, use_llm=True, extract_answer=True, domain_fields=None, api_base_url=None, model_name=None):
    # Initialize domain metrics dictionary
    #  filtered_data(完整数据), input_list(问题), output_list(模型输出)
    print(f"evaluate.py开始运行 run_evaluation, filtered_data={filtered_data}")
    print(f"evaluate.py开始运行 run_evaluation, input_list={input_list}")
    print(f"evaluate.py开始运行 run_evaluation, output_list={output_list}")
    print(f"evaluate.py开始运行 run_evaluation, filtered_data={task_type}")

    domain_metrics = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'em': [],
        'acc': [],
        'f1': [],
        'math_equal': [],
        'llm_equal': [],
        'pass@1': []
    })

    # Helper function to get domain from item  （按领域/难度分组的指标统计）
    def get_domain(item):
        if not domain_fields:  # 增加这一行，处理 None 或空列表的情况
            return 'Unknown'
        for field in domain_fields:
            if field in item and item[field] is not None:
                return item[field]
        return 'Unknown'

    if task_type == 'code':
        # Prepare samples and generations for codegen_metrics
        samples_list = []
        generations_list = []
        num_valid_answer = 0

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:  # 若 result 是字符串 → item['Output'] = result
                item['Output'] = result
            else:  # 若 result 是对象 → item['Output'] = result.outputs[0].text
                item['Output'] = result.outputs[0].text

            if item['Output'] == '':
                item['Pred_Answer'] = ''
                item['Question'] = input_prompt
                item['Metrics'] = {'pass@1': 0}
                item['Results'] = {}
                item['Final_metadata'] = {}
                continue

            pred_code = extract_answer_fn(item['Output'], mode='codegen', extract_answer=extract_answer)
            if pred_code != '':
                num_valid_answer += 1

            public_test_cases = json.loads(item.get("test_cases", "{}"))

            inputs = public_test_cases.get("inputs", [])
            outputs = public_test_cases.get("outputs", [])

            sample = {
                "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

            samples_list.append(sample)
            generations_list.append([pred_code])
            item['Pred_Answer'] = pred_code
            item['Question'] = input_prompt

        # # Call codegen_metrics with pass@1
        # metrics, results, final_metadata = codegen_metrics(
        #     samples_list,
        #     generations_list,
        #     k_list=[1],  # Evaluate the top 1 generated result
        #     num_process_evaluate=10,   # Parallel evaluation
        #     timeout=10,  # Set timeout to 10 seconds
        #     debug=False,  # Enable debug mode
        # )

        # # Extract pass@1
        # pass_at_1 = metrics.get('pass@1', 0.0)
        # detail_pass_at_1 = metrics['detail']['pass@1']

        # for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
        #     item['Metrics'] = {'pass@1': pass1}
        #     item['Results'] = res
        #     item['Final_metadata'] = meta

        # Compute overall pass@1
        overall_metrics = {
            'pass@1': 0.0, # pass_at_1,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        }

        # Add domain-specific metrics collection
        for item in filtered_data:
            domain = get_domain(item)
            domain_metrics[domain]['total'] += 1
            domain_metrics[domain]['pass@1'].append(0.0)

    elif task_type in ['math', 'choose', 'qa']:
        # Evaluation for math/qa tasks
        avg_em, avg_acc, avg_f1, avg_math, avg_llm = [], [], [], [], []
        num_valid_answer = 0
        
        # Lists to store data for batch LLM evaluation
        questions_for_llm = []
        labeled_answers_for_llm = []
        pred_answers_for_llm = []
        items_for_llm = []

        for item, input_prompt, result in tqdm(zip(filtered_data, input_list, output_list), total=len(input_list)):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text

            if item['Output'] == '':
                item['Pred_Answer'] = ''
                item['Question'] = input_prompt
                item['Metrics'] = {
                    'em': 0,
                    'acc': 0,
                    'f1': 0,
                    'math_equal': 0,
                    'llm_equal': 0 if use_llm else None
                }
                avg_em.append(0)
                avg_acc.append(0)
                avg_f1.append(0)
                avg_math.append(0)
                if use_llm:
                    avg_llm.append(0)
                continue

            # Get the labeled answer from the item 获取标准答案。 优先级: item['answer'] → item['Correct Choice'] → item['answer_letter']
            labeled_answer = item.get('answer', '')  # Use get() to safely access the answer field
            if 'Correct Choice' in item and item['Correct Choice'] is not None:
                labeled_answer = item['Correct Choice']
            elif 'answer_letter' in item and item['answer_letter'] is not None:
                labeled_answer = item['answer_letter']
            # 调用 evaluate_predictions() → 得到 (metric, pred_answer)
            metric, pred_answer = evaluate_predictions(
                output=result, 
                labeled_answer=labeled_answer,
                mode=task_type,
                use_llm=use_llm,
                question=input_prompt,
                extract_answer=extract_answer
            )
            
            # 赋值
            item['Pred_Answer'] = pred_answer
            item['Metrics'] = metric
            item['Question'] = input_prompt

            # Store data for batch LLM evaluation
            if use_llm:  # 若 use_llm=True: 批量调用 llm_evaluate_equivalence_batch()
                questions_for_llm.append(input_prompt)
                labeled_answers_for_llm.append(labeled_answer)
                pred_answers_for_llm.append(pred_answer)
                items_for_llm.append(item)

            # Determine the validity of the predicted answer
            my_method_valid = (pred_answer != '')

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

        # Perform batch LLM evaluation if needed
        if use_llm and questions_for_llm:
            # llm_results = asyncio.run(llm_evaluate_equivalence_batch(
            llm_results = await llm_evaluate_equivalence_batch(
                questions=questions_for_llm,
                labeled_answers=labeled_answers_for_llm,
                pred_answers=pred_answers_for_llm,
                extract_answer=extract_answer,
                api_base_url=api_base_url,
                model_name=model_name
            )
            
            # Update metrics with LLM results
            for item, (llm_result, llm_response) in zip(items_for_llm, llm_results):
                item['Metrics']['llm_equal'] = int(llm_result)
                item['Metrics']['llm_response'] = llm_response
                avg_llm.append(int(llm_result))

        # Compute overall metrics
        overall_metrics = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
            'math_equal': np.mean(avg_math) if len(avg_math) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        }
        
        # Add LLM evaluation metric if available
        if len(avg_llm) > 0:
            overall_metrics['llm_equal'] = np.mean(avg_llm)

        for item, metric in zip(filtered_data, [item['Metrics'] for item in filtered_data]):
            domain = get_domain(item)
            domain_metrics[domain]['total'] += 1
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])
            if 'llm_equal' in metric:
                domain_metrics[domain]['llm_equal'].append(metric['llm_equal'])

    # After the main evaluation loop and before saving metrics, add:
    # Calculate domain-specific metrics
    domain_metrics_final = {}
    for domain, metrics in domain_metrics.items():
        domain_metrics_final[domain] = {
            'total': metrics['total'],
            'em': np.mean(metrics['em']) if len(metrics['em']) > 0 else 0.0,
            'acc': np.mean(metrics['acc']) if len(metrics['acc']) > 0 else 0.0,
            'f1': np.mean(metrics['f1']) if len(metrics['f1']) > 0 else 0.0,
            'math_equal': np.mean(metrics['math_equal']) if len(metrics['math_equal']) > 0 else 0.0,
        }
        if metrics['llm_equal']:
            domain_metrics_final[domain]['llm_equal'] = np.mean(metrics['llm_equal'])
        if metrics['pass@1']:
            domain_metrics_final[domain]['pass@1'] = np.mean(metrics['pass@1'])

    # Add domain metrics to overall metrics
    overall_metrics['domain_metrics'] = domain_metrics_final
    
    print(overall_metrics)
    
    # Save prediction results and metrics
    print(f"run_evaluate.py开始 Save prediction results and metrics 每条数据的详细结果，路径={output_metrics_path}")
    with open(os.path.join(output_dir, output_metrics_path), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    # ===== 新增：将 filtered_data 追加到 all.jsonl =====
    print(f"run_evaluate.py开始 将 filtered_data 追加到 all.jsonl")
    all_jsonl_path = os.path.join(output_dir, 'all.jsonl')
    with open(all_jsonl_path, mode='a', encoding='utf-8') as jsonl_file:
        for item in filtered_data:
            simplified_item = {
                "id": item.get("id"),
                "answer": item.get("answer"),
                "Pred_Answer": item.get("Pred_Answer"),  # 若 extract_answer=False，直接返回完整输出；否则提取最后3行
                "Output": item.get("Output"),
            }
            jsonl_file.write(json.dumps(simplified_item, ensure_ascii=False) + '\n')
    # ===================================================

    print(f"run_evaluate.py开始 Save prediction results and metrics总体指标，路径={output_metrics_overall_path}")
    with open(os.path.join(output_dir, output_metrics_overall_path), mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model outputs.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--task', type=str, required=True, choices=['code', 'math', 'choose', 'qa', 'llm'], help='Task type for evaluation')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM for equivalence evaluation')
    parser.add_argument('--extract_answer', action='store_true', help='Extract answer from output')
    parser.add_argument('--api_base_url', type=str, default=None, help='Base URL for LLM API')
    parser.add_argument('--model_name', type=str, default=None, help='Model name for LLM evaluation')
    args = parser.parse_args()

    # Define the list of domain field names to check (in order of priority)
    DOMAIN_FIELDS = ['Level', 'level', 'category', 'High-level domain', 'difficulty_level', 'field', 'problem_topic']

    output_path = args.output_path
    output_metrics_path = output_path.replace('.json', '.metrics.json')
    output_metrics_overall_path = output_path.replace('.json', '.metrics.overall.json')

    # Load main output data
    with open(output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    # Prepare input_list and output_list for run_evaluation
    input_list = []
    output_list = []
    filtered_data = []
    
    if isinstance(data, dict):
        # Convert dict to list if data is a dictionary
        for key, item in data.items():
            if isinstance(item, dict):  # Ensure item is a dictionary
                filtered_data.append(item)
                input_list.append(item.get('question'))
                output_list.append(item.get('result'))
    else:
        # If data is already a list
        filtered_data = data
        input_list = [item.get('Question', item.get('question')) for item in data]
        output_list = [item.get('Output', item.get('result')) for item in data]

    # Run evaluation with domain fields
    run_evaluation(
        filtered_data=filtered_data,  # Pass the properly structured data
        input_list=input_list,
        output_list=output_list,
        task_type=args.task,
        output_dir=output_path,
        output_metrics_path=output_metrics_path,
        output_metrics_overall_path=output_metrics_overall_path,
        use_llm=args.use_llm,
        api_base_url=args.api_base_url,
        model_name=args.model_name,
        extract_answer=args.extract_answer,
        domain_fields=DOMAIN_FIELDS  # Pass the domain fields to run_evaluation
    )

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
