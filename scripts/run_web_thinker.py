# run_web_thinker.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict, Set
import argparse
import random
import asyncio
import aiohttp

from openai import AsyncOpenAI

from search.bing_search import (
    bing_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    fetch_page_content_async,
    extract_snippet_with_context,
    bing_web_search_async,
    google_serper_search_async,
    extract_relevant_info_serper
)
from evaluate.evaluate import (
    run_evaluation, 
    extract_answer_fn
)
from prompts.prompts import (
    get_deep_web_explorer_instruction, 
    get_web_page_reader_instruction,
    get_search_intent_instruction,
    get_click_intent_instruction,
    get_multiqa_search_o1_instruction, 
    get_task_instruction_openqa, 
)
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("/share/project/llm/QwQ-32B")
# # tokenizer = AutoTokenizer.from_pretrained("/share/project/llm/DeepSeek-R1-Distill-Qwen-32B")
# aux_tokenizer = AutoTokenizer.from_pretrained("/share/project/llm/Qwen2.5-72B-Instruct")


# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
# BEGIN_CLICK_INTENT = "<|begin_click_intent|>"
# END_CLICK_INTENT = "<|end_click_intent|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

error_indicators = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]

invalid_search_queries = [
    "and end with",
    "search query",
    "query",
    "your query here",
    "your query",
    "your search query",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-o1 for various datasets and models.")
    parser.add_argument('--single_question', type=str, default=None, help="Single question to process instead of dataset")
    parser.add_argument('--dataset_name', type=str, required=False, default='custom', help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=False, default='test', help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")

    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument('--min_p', type=float, default=0.05, help="Minimum p sampling parameter.")
    parser.add_argument('--top_k_sampling', type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help="Repetition penalty. If not set, defaults based on the model.")
    parser.add_argument('--max_tokens', type=int, default=81920, help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.")

    parser.add_argument('--max_search_limit', type=int, default=20, help="Maximum number of searches per question.")
    parser.add_argument('--top_k', type=int, default=10, help="Maximum number of search documents to return.")
    # keep_links 只影响"点击链接后抓取的网页正文"里是否保留页面内的超链接，并不影响第一次搜索结果的 URL 展示——因为 format_search_results 是用 json.dumps(doc_info) 输出的，doc_info 里始终带 url 字段，模型无论 keep_links 设不设都能从搜索结果里看到 URL 并发起
    parser.add_argument('--keep_links', action='store_true', default=False, help="Whether to keep links in fetched web content")
    parser.add_argument('--use_jina', action='store_true', help="Whether to use Jina API for document fetching.")
    parser.add_argument('--jina_api_key', type=str, default='None', help="Your Jina API Key to Fetch URL Content.")
    parser.add_argument('--bing_subscription_key', type=str, default=None, help="Bing Search API subscription key.")
    parser.add_argument('--bing_endpoint', type=str, default="https://api.bing.microsoft.com/v7.0/search", help="Bing Search API endpoint.")
    parser.add_argument('--serper_api_key', type=str, default=None, help="Google Serper API key.")
    parser.add_argument('--search_engine', type=str, default="bing", choices=["bing", "serper"], help="Search engine to use (bing or serper). Default: bing")
    parser.add_argument('--eval', action='store_true', help="Whether to run evaluation after generation.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for generation. If not set, will use current timestamp as seed.")
    parser.add_argument('--api_base_url', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--aux_api_base_url', type=str, required=True, help="Base URL for the auxiliary model API endpoint")
    parser.add_argument('--model_name', type=str, default="QwQ-32B", help="Name of the model to use")
    parser.add_argument('--aux_model_name', type=str, default="Qwen2.5-32B-Instruct", help="Name of the auxiliary model to use")
    parser.add_argument('--concurrent_limit', type=int, default=32, help="Maximum number of concurrent API calls")
    parser.add_argument('--lora_name', type=str, default=None, help="Name of the LoRA adapter to load")
    parser.add_argument('--lora_path', type=str, default=None, help="Path to the LoRA weights")
    parser.add_argument('--tokenizer_path', type=str, default="/share/project/llm/QwQ-32B", help="Path to the main tokenizer")
    parser.add_argument('--aux_tokenizer_path', type=str, default="/share/project/llm/Qwen2.5-32B-Instruct", help="Path to the auxiliary tokenizer")
    parser.add_argument('--api_key', type=str, default="empty", help="API key for the main model")
    parser.add_argument('--aux_api_key', type=str, default="empty", help="API key for the auxiliary model")
    return parser.parse_args()

# Initialize tokenizers
args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
aux_tokenizer = AutoTokenizer.from_pretrained(args.aux_tokenizer_path)


def extract_between(text, start_marker, end_marker):  # 从一段文本中，提取指定起始标记和结束标记之间的内容。返回的是文本中“最后一次出现”的起始标记和结束标记之间的内容（而不是通常默认的第一次出现）
    """Extracts text between two markers in a string."""
    print(f"extract_between开始执行")
    try:
        pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
        # Run pattern matching with timeout
        matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
        if matches:
            return matches[0][::-1].strip()
        return None
    except Exception as e:
        print(f"---Error:---\n{str(e)}")
        print(f"-------------------")
        return None

def format_search_results(relevant_info: List[Dict]) -> str:
    print(f"开始执行format_search_results，relevant_info：{json.dumps(relevant_info, indent=4, ensure_ascii=False)}")
    """Format search results into a readable string"""
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        doc_info['title'] = doc_info['title'].replace('<b>','').replace('</b>','')
        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')
        formatted_documents += f"***Web Page {i + 1}:***\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
        # formatted_documents += f"Title: {doc_info['title']}\n"
        # formatted_documents += f"URL: {doc_info['url']}\n"
        # formatted_documents += f"Snippet: {doc_info['snippet']}\n\n"
        # if 'page_info' in doc_info:
        #     formatted_documents += f"Web Page Information: {doc_info['page_info']}\n\n\n\n"
    print(f"处理完毕format_search_results，formatted_documents：{json.dumps(formatted_documents, indent=4, ensure_ascii=False)}")
    return formatted_documents


# 带重试机制的异步 LLM 调用。负责与 LLM API（通过 AsyncOpenAI 客户端）进行通信
async def generate_response(
    client: AsyncOpenAI,  # AsyncOpenAI 客户端实例
    prompt: str,  # 输入文本（用户 prompt）
    semaphore: asyncio.Semaphore,  # asyncio.Semaphore，控制并发数
    generate_mode: str = "chat",  # "chat" 或 "completion"
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.0,
    top_k: int = 1,
    min_p: float = 0.0,
    model_name: str = "QwQ-32B",
    stop: List[str] = [END_SEARCH_QUERY],  # 停止词列表，遇到即停止生成。这让模型在"想要搜索"时停下来，等待系统处理搜索后再继续。
    retry_limit: int = 3,  # 最大重试次数（默认 3）
    bad_words: List[str] = [f"{END_SEARCH_RESULT}\n\n{tokenizer.eos_token}"],
) -> Tuple[str, str]:
    """Generate a single response with retry logic"""
    print(f"开始执行generate_response")
    for attempt in range(retry_limit):
        try:
            async with semaphore:   # 获取并发许可（受 concurrent_limit 控制）
                if generate_mode == "chat":  # 【模式分支】首次调用（主模型开始回答）
                    messages = [{"role": "user", "content": prompt}]
                    if 'qwq' in model_name.lower() or 'deepseek' in model_name.lower() or 'r1' in model_name.lower():
                        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        formatted_prompt = aux_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if ('deepseek' in model_name.lower() or 'r1' in model_name.lower()) and "<think>\n" not in formatted_prompt:  # 特殊处理 DeepSeek: 如果 prompt 不以 "思考" 结尾，追加 "思考\n"  为了兼容 DeepSeek-R1 的思考模式，强制追加思考触发词。
                        formatted_prompt = formatted_prompt + "<think>\n"
                else:  # 【模式分支】后续调用（模型已经输出了搜索标记，需要续写） # # 直接原样使用
                    formatted_prompt = prompt  

                response = await client.completions.create(  # 调用 API
                    model=model_name,
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    extra_body={   # vLLM 特有参数
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        # 'bad_words': bad_words,
                        # 'min_p': min_p
                    },
                    timeout=3600,  # 1小时超时
                )
                return formatted_prompt, response.choices[0].text
        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            # print(prompt)
            if "maximum context length" in str(e).lower():
                # If length exceeds limit, reduce max_tokens by half
                max_tokens = max_tokens // 2   # 自动减半
                print(f"token超过了，自动减半  Reducing max_tokens to {max_tokens}")
            if attempt == retry_limit - 1:
                print(f"彻底失败 Failed after {retry_limit} attempts: {e}")
                return "", ""
            await asyncio.sleep(1 * (attempt + 1))
    return "", ""


# 让主模型（如 QwQ-32B）在获得初始搜索结果后，能够自主决定：
# 是否需要发起新的搜索查询（search）
# 是否需要点击某个链接深入阅读（click）
# 通过多轮交互，模型可以像人类一样"浏览网页"，逐步收集信息，最终形成完整的答案。
async def generate_deep_web_explorer(
    client: AsyncOpenAI,    # 主模型客户端（QwQ-32B）
    aux_client: AsyncOpenAI,   # 辅助模型客户端（Qwen2.5-32B-Instruct）
    search_query: str,   # 当前搜索查询（触发深度探索的原始查询）
    document: str,  # 初始搜索结果（已格式化的文档列表）
    search_intent: str,  # 搜索意图（由辅助模型生成）
    args: argparse.Namespace, 
    search_cache: Dict,  # 搜索缓存字典
    url_cache: Dict,   # URL 内容缓存字典
    semaphore: asyncio.Semaphore,  # 并发控制信号量
) -> Tuple[str, List[Dict], str]:
    """
    Generate deep web exploration with multiple search and click operations
    Returns the output, list of interaction records, and initial prompt
    """
    # 这是真正触发 Deep Web Explorer 循环的地方。系统不会直接把原始的搜索结果塞回给主模型，而是启动一个独立的交互循环，在这个循环中：
    print(f"开始执行generate_deep_web_explorer, search_intent={search_intent}")
    prompt = get_deep_web_explorer_instruction(search_query=search_query, search_intent=search_intent, search_result=document)  # 将当前的搜索词、搜索到的网页列表（包含标题、URL、摘要）以及提示词 get_deep_web_explorer_instruction 传递给模型。
    output = ""
    original_prompt = ""
    total_tokens = len(prompt.split())  # Track total tokens including prompt
    MAX_TOKENS = 30000
    MAX_INTERACTIONS = 10  # Maximum combined number of searches and clicks  # 最大交互次数（搜索+点击）
    clicked_urls = set()  # Track clicked URLs   # 已点击的 URL（防重复）
    executed_search_queries = set()  # Track executed search queries  # 已执行的搜索（防重复）
    total_interactions = 0   # # 当前交互计数
    finished = False  # 是否完成
    first_generation = True  # 是否首次生成

    while True:
        # Generate next response
        formatted_prompt, response = await generate_response(
            client=client if 'qwq' in args.model_name.lower() else aux_client,
            model_name=args.model_name if 'qwq' in args.model_name.lower() else args.aux_model_name,
            prompt=prompt,
            semaphore=semaphore,
            generate_mode="chat" if first_generation else "completion",  # 首次生成用 chat 模式（需要 apply_chat_template）。后续用 completion 模式（直接续写）
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            stop=[END_SEARCH_QUERY, END_CLICK_LINK],  # 遇到这两个标记就停止
        )
        print("=" * 60)
        print(f"开始执行generate_deep_web_explorer, formatted_prompt={formatted_prompt}")
        print("=" * 60)

        print("=" * 60)
        print(f"开始执行generate_deep_web_explorer, response={response}")
        print("=" * 60)
      
        if first_generation:
            original_prompt = formatted_prompt
            prompt = formatted_prompt
        
        output += response.replace('</think>\n','')
        total_tokens = len(prompt.split()) + len(response.split())
        first_generation = False

        if total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS:
            break

        # 模型决策：在这个循环中，模型（通常是辅助模型 Qwen2.5-32B-Instruct 扮演探索者的角色）会分析当前的搜索结果，决定下一步的行动：
        # Check for search query。如果模型认为当前结果不够，需要换个关键词搜索，它会输出新的 <|begin_search_query|>。提取查询文本，调用搜索 API
        if response.rstrip().endswith(END_SEARCH_QUERY):
            new_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            print(f"generate_deep_web_explorer里面的 new_query={new_query}")
            total_interactions += 1
            # 过滤无效查询
            if new_query is None or END_SEARCH_QUERY in new_query or len(new_query) <= 5 or new_query in invalid_search_queries:
                print(f"generate_deep_web_explorer里面的 new_query 失败了，因为 new_query is None or END_SEARCH_QUERY in new_query or len(new_query) <= 5 or new_query in invalid_search_queries")
                continue
            # 防重复
            if new_query:
                if new_query in executed_search_queries:
                    print(f"new_query in executed_search_queries, 告诉LLM你已经搜索过了")
                    # If search query was already executed, append message and continue
                    search_result = f"\n{BEGIN_SEARCH_RESULT}\nYou have already searched for this query. Please use the previously found information.\n{END_SEARCH_RESULT}\n\nOkay,"
                    output += search_result
                    prompt += output
                    total_tokens += len(search_result.split())
                    continue

                executed_search_queries.add(new_query)  # Add query to executed set
                
                # Execute search  # 执行搜索（带缓存）
                if new_query in search_cache:
                    results = search_cache[new_query]
                    print(f"new_query（{new_query}）在缓存中，直接取结果：{json.dumps(results, indent=4, ensure_ascii=False)}")
                else:
                    try:
                        if args.search_engine == "bing":
                            print(f"搜索引擎是 bing")
                            results = await bing_web_search_async(new_query, args.bing_subscription_key, args.bing_endpoint)
                        elif args.search_engine == "serper":
                            print(f"搜索引擎是 google")
                            results = await google_serper_search_async(new_query, args.serper_api_key)
                        else: # Should not happen
                            results = {}
                        search_cache[new_query] = results
                    except Exception as e:
                        print(f"Error during search query '{new_query}' using {args.search_engine}: {e}")
                        results = {}
                print(f'- Searched for "{new_query}" using {args.search_engine}')

                # 提取结构化信息
                if args.search_engine == "bing":
                    relevant_info = extract_relevant_info(results)[:args.top_k]
                elif args.search_engine == "serper":
                    relevant_info = extract_relevant_info_serper(results)[:args.top_k]
                else: # Should not happen
                    relevant_info = []

                formatted_documents = format_search_results(relevant_info)
                
                # Append search results  # 将结果追加到对话历史
                search_result = f"\n{BEGIN_SEARCH_RESULT}\n{formatted_documents}\n{END_SEARCH_RESULT}\n"
                output += search_result
                prompt += output
                total_tokens += len(search_result.split())
                
        # Check for click link。点击链接：如果模型认为某个网页包含需要的信息，它会输出。提取 URL，爬取页面，用辅助模型总结
        elif response.rstrip().endswith(END_CLICK_LINK):
            url = extract_between(response, BEGIN_CLICK_LINK, END_CLICK_LINK) # 提取该 URL
            print(f"generate_deep_web_explorer里面  辅助模型 想要点击的url={url}")
            # click_intent = extract_between(response, BEGIN_CLICK_INTENT, END_CLICK_INTENT)
            total_interactions += 1
            # # 用辅助模型生成点击意图
            _, click_intent = await generate_response(
                client=aux_client,
                model_name=args.aux_model_name,
                max_tokens=1000,
                prompt=get_click_intent_instruction(output),
                semaphore=semaphore,
            )
            print(f"generate_deep_web_explorer里面  通过 generate_response 生成 click_intent={click_intent}")

            # 防重复点击
            if url and click_intent:
                if url in clicked_urls:
                    # If URL was already clicked, append message
                    click_result = f"\n{BEGIN_CLICK_RESULT}\nYou have already clicked this URL.\n{END_CLICK_RESULT}\n\nOkay,"
                    output += click_result
                    prompt += output
                    total_tokens += len(click_result.split())
                    continue

                clicked_urls.add(url)  # Add URL to clicked set
                print(f"- Clicking on URL: {url} with intent: {click_intent}")
                
                # 爬取页面内容  # Fetch and process page content
                if url not in url_cache:
                    try:  # 这里获取的是用户/模型明确想要深入查看的页面内容。
                        content = await fetch_page_content_async(
                            [url],   # # 单个 URL
                            use_jina=args.use_jina, 
                            jina_api_key=args.jina_api_key, 
                            keep_links=args.keep_links
                        )
                        content = content[url]   # 获取单条结果
                        # Only cache content if it doesn't contain error indicators
                        # 错误检测
                        has_error = (any(indicator.lower() in content.lower() for indicator in error_indicators) and len(content.split()) < 64) or content == ''
                        if not has_error:
                            url_cache[url] = content
                    except Exception as e:
                        print(f"Error fetching URL {url}: {e}")
                        content = ""
                else:
                    content = url_cache[url]

                # Check if content has error indicators
                has_error = any(indicator.lower() in content.lower() for indicator in error_indicators) or content == ''
                
                # 内容处理
                if has_error:
                    # If content has error, use it directly as summary
                    summary = "Unable to fetch the page content. You can try other links."
                else:
                    # Use web page reader to summarize content。然后用辅助模型作为 Web Page Reader 总结内容
                    reader_prompt = get_web_page_reader_instruction(click_intent, content)
                    _, summary = await generate_response(
                        client=aux_client,
                        prompt=reader_prompt,
                        semaphore=semaphore,
                        max_tokens=3600,
                        model_name=args.aux_model_name,
                    )

                print(f"用辅助模型作为 Web Page Reader 总结内容summary={summary}")
                # Append click results
                click_result = f"\n{BEGIN_CLICK_RESULT}\n{summary}\n{END_CLICK_RESULT}\n"
                output += click_result
                prompt += output
                total_tokens += len(click_result.split())
        
        else:  # 不以 END_SEARCH_QUERY 或 END_CLICK_LINK 结尾，说明模型已经"满意"了，直接结束。
            finished = True
            break

    # Add max limit message if needed
    # token 超过 30000 或者 交互次数超过 10 次，强制结束
    if not finished and (total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS):
        output += f"\n{BEGIN_CLICK_RESULT}\nYou have reached the limit for clicking links.\n{END_CLICK_RESULT}\n\nOK, I will now provide the final information based on my collected information.\n\n**Final Information:**"
        prompt += output
        _, final_response = await generate_response(
            client=client if 'qwq' in args.model_name.lower() else aux_client,
            model_name=args.model_name if 'qwq' in args.model_name.lower() else args.aux_model_name,
            prompt=prompt,
            semaphore=semaphore,
            generate_mode="completion",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=512,
            repetition_penalty=1.2,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
        )
        output += final_response  # 完整的交互历史（包含所有搜索、点击、结果、最终答案）

    return output, original_prompt  # 最原始的 prompt（用于记录和调试）


# 单条问题的完整推理链处理。负责驱动一条问题从初始 prompt 到最终答案的完整生命周期
async def process_single_sequence(
    seq: Dict,  # 序列字典，包含 prompt, output, finished, history
    client: AsyncOpenAI,  # 	主模型客户端（QwQ-32B）
    aux_client: AsyncOpenAI,  # 辅助模型客户端（Qwen2.5-32B-Instruct）
    semaphore: asyncio.Semaphore,  # 并发控制信号量
    args: argparse.Namespace,
    search_cache: Dict,  # 全局搜索缓存
    url_cache: Dict,
    batch_output_records: List[Dict],
) -> Dict:
    """Process a single sequence through its entire reasoning chain with MAX_TOKENS limit"""
    
    # 初始化 token 计数器，初始值设为 prompt 的 token 数（简单用 split() 作为近似）
    MAX_TOKENS = 40000
    total_tokens = len(seq['prompt'].split())
    
    # Initialize web explorer interactions list
    seq['web_explorer'] = []  # 深度探索记录
    
    # First response uses chat completion  【Step 1: 首次生成】
    formatted_prompt, response = await generate_response(
        client=client,
        model_name=args.model_name,
        prompt=seq['prompt'],
        semaphore=semaphore,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k_sampling,
        min_p=args.min_p,
        stop=[END_SEARCH_QUERY],
    )
    print(f"process_single_sequence里面的 首次生成 formatted_prompt={formatted_prompt}")
    print(f"process_single_sequence里面的 首次生成 response={response}")

    # Update token count and sequence fields
    tokens_this_response = len(response.split())
    total_tokens += tokens_this_response
    
    seq['output'] += response.replace('</think>\n', '')
    seq['history'].append(response.replace('</think>\n', ''))
    seq['original_prompt'] = formatted_prompt
    seq['prompt'] = formatted_prompt + response.replace('</think>\n', '')
    
    # # 拼接为下一轮输入 【Step 2: 主循环】
    while not seq['finished']:
        print(f"process_single_sequence里面的 seq={seq}")
        # Check if sequence is finished
        if not seq['output'].rstrip().endswith(END_SEARCH_QUERY):  #  检查输出是否以 <|end_search_query|> 结尾？
            seq['finished'] = True  # 否 → seq['finished'] = True, break  # 模型已给出最终答案
            break
        
        # 是 → 进入搜索处理流程  
        search_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)  # 【提取搜索查询】
        print(f"process_single_sequence里面的  extract_between处理后的 search_query={search_query}")
        seq['search_count'] += 1

        if seq['search_count'] < args.max_search_limit and total_tokens < MAX_TOKENS:  #【合法性检查】
            if search_query is None or len(search_query) <= 5 or END_SEARCH_QUERY in search_query or search_query in invalid_search_queries: # 不合法的query
                print(f"process_single_sequence里面的  extract_between处理后的不合法 search_query={search_query}")
                continue

            if search_query in seq['executed_search_queries']:  # # 已搜索过，提示模型使用已有信息
                # If search query was already executed, append message and continue
                append_text = f"\n\n{BEGIN_SEARCH_RESULT}You have already searched for this query.{END_SEARCH_RESULT}\n\nOkay,"
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)
                total_tokens += len(append_text.split())
                continue

            _, search_intent = await generate_response(
                client=aux_client,
                model_name=args.aux_model_name,
                max_tokens=1000,
                prompt=get_search_intent_instruction(seq['output']),
                semaphore=semaphore,
            )
            print(f"process_single_sequence里面的 generate_response生成的  search_intent={search_intent}")


            # 执行搜索和后续操作（同原逻辑）
            if search_query in search_cache:
                results = search_cache[search_query]
            else:
                try:
                    if args.search_engine == "bing":
                        results = await bing_web_search_async(search_query, args.bing_subscription_key, args.bing_endpoint)
                    elif args.search_engine == "serper":
                        print(f"开始google_serper_search_async，search_query={search_query}")
                        results = await google_serper_search_async(search_query, args.serper_api_key)
                    else: # Should not happen
                        results = {}
                    search_cache[search_query] = results
                except Exception as e:
                    print(f"Error during search query '{search_query}' using {args.search_engine}: {e}")
                    results = {}
            print(f'Searched for: "{search_query}" using {args.search_engine}')

            if args.search_engine == "bing":
                relevant_info = extract_relevant_info(results)[:args.top_k]
            elif args.search_engine == "serper":
                relevant_info = extract_relevant_info_serper(results)[:args.top_k]
                print(f"serper搜索结果经过 extract_relevant_info_serper后的结果 relevant_info ={json.dumps(relevant_info, indent=4, ensure_ascii=False)}")
            else: # Should not happen
                relevant_info = []

            # Process documents
            urls_to_fetch = []
            for doc_info in relevant_info:
                url = doc_info['url']
                if url not in url_cache:
                    urls_to_fetch.append(url)

            if urls_to_fetch:
                try:
                    contents = await fetch_page_content_async(
                        urls_to_fetch,   # 本次搜索返回的所有 URL
                        use_jina=args.use_jina, 
                        jina_api_key=args.jina_api_key, 
                        keep_links=args.keep_links
                    )
                    for url, content in contents.items():
                        # Only cache content if it doesn't contain error indicators
                        has_error = (any(indicator.lower() in content.lower() for indicator in error_indicators) and len(content.split()) < 64) or len(content) < 50 or len(content.split()) < 20
                        if not has_error:
                            url_cache[url] = content
                        # else:
                        #     print(f'---Fetching Error\n{content}')
                except Exception as e:
                    print(f"Error fetching URLs: {e}")

            # Get web page information for each result
            for doc_info in relevant_info:
                url = doc_info['url']
                if url not in url_cache:
                    raw_content = ""
                else:
                    raw_content = url_cache[url]
                    is_success, raw_content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=2000)  #  raw_content刚爬取的网页全文,# Bing/Serper API 返回的摘要,# 前后各取 2000 字符
                    print(f"run_search_o1.py里面的 relevant_info循环 里面的 extract_snippet_with_context结果：success={is_success}，filtered_context={json.dumps(raw_content, indent=4, ensure_ascii=False)}")

                # Check if content has error indicators
                has_error = any(indicator.lower() in raw_content.lower() for indicator in error_indicators) or raw_content == ""
            
                if has_error:
                    # If content has error, use it directly as summary
                    doc_info['page_info'] = "Can not fetch the page content."
                else:
                    # Use raw content directly as page info
                    doc_info['page_info'] = raw_content  # 这就是最终给 LLM 看的正文
                    # # Use detailed web page reader to process content
                    # reader_prompt = get_detailed_web_page_reader_instruction(search_query, search_intent, raw_content)
                    # _, page_info = await generate_response(
                    #     client=aux_client,
                    #     prompt=reader_prompt,
                    #     semaphore=semaphore,
                    #     max_tokens=4000,
                    #     model_name=args.aux_model_name,
                    # )
                    # doc_info['page_info'] = page_info

            formatted_documents = format_search_results(relevant_info) # 【格式化搜索结果】

            # Generate deep web exploration with interactions  【深度网页探索】★ 关键步骤
            analysis, explorer_prompt = await generate_deep_web_explorer(
                client=client,
                aux_client=aux_client,
                search_query=search_query,
                search_intent=search_intent,
                document=formatted_documents,
                args=args,
                search_cache=search_cache,
                url_cache=url_cache,
                semaphore=semaphore,
            )

            extracted_info = extract_answer_fn(analysis, mode='summary')  # 【提取关键信息】

            # Store web explorer input/output with all interactions  【记录探索过程】
            seq['web_explorer'].append({
                "search_query": search_query,
                "Input": explorer_prompt,
                "Output": analysis,
                "Extracted_info": extracted_info
            })
            
            # Update sequence with search results  【追加到主对话历史】
            append_text = f"\n\n{BEGIN_SEARCH_RESULT}{extracted_info}{END_SEARCH_RESULT}\n\n"
            seq['prompt'] += append_text
            seq['output'] += append_text
            seq['history'].append(append_text)
            
            seq['executed_search_queries'].add(search_query)
            total_tokens += len(append_text.split())
            
            # Subsequent responses use completion mode  【继续生成】（completion 模式续写）
            _, response = await generate_response(
                client=client,
                model_name=args.model_name,
                prompt=seq['prompt'],
                semaphore=semaphore,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k_sampling,
                min_p=args.min_p,
                stop=[END_SEARCH_QUERY],
                generate_mode="completion"
            )
            
            # Update token count and sequence fields
            tokens_this_response = len(response.split())
            total_tokens += tokens_this_response
            
            seq['output'] += response.replace('</think>\n', '')
            seq['history'].append(response.replace('</think>\n', ''))
            seq['prompt'] += response.replace('</think>\n', '')
            continue

        else:  # 搜索次数或 token 超限
            append_text = f"\n\n{BEGIN_SEARCH_RESULT}You have reached the search limit. You are not allowed to search.{END_SEARCH_RESULT}\n\n"
            seq['prompt'] += append_text
            seq['output'] += append_text
            seq['history'].append(append_text)
            
            _, final_response = await generate_response(
                client=client,
                prompt=seq['prompt'],
                semaphore=semaphore,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=1.1,
                top_k=args.top_k_sampling,
                min_p=args.min_p,
                model_name=args.model_name,
                generate_mode="completion",
                bad_words=[f"{END_SEARCH_RESULT}\n\n{tokenizer.eos_token}", f"{END_SEARCH_QUERY}{tokenizer.eos_token}"]
            )
            
            seq['output'] += final_response
            seq['history'].append(final_response)
            seq['finished'] = True
            break
    
    return seq


# vLLM 支持在运行时动态加载 LoRA 适配器，无需重启服务即可切换不同的微调权重。
# 加载：在推理开始前，将特定 LoRA 权重挂载到基础模型上
async def load_lora_adapter(api_base_url: str, lora_name: str, lora_path: str) -> bool:
    """Load a LoRA adapter with the specified name and path"""
    try:
        lora_load_url = f"{api_base_url}/load_lora_adapter"
        lora_payload = {
            "lora_name": lora_name,
            "lora_path": lora_path
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(lora_load_url, json=lora_payload) as response:
                return response.status == 200
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        return False

# 卸载：在任务完成后，释放 LoRA 占用的 GPU 显存
async def unload_lora_adapter(api_base_url: str, lora_name: str) -> bool:
    """Unload a LoRA adapter with the specified name"""
    try:
        unload_url = f"{api_base_url}/unload_lora_adapter"
        unload_payload = {"lora_name": lora_name}
        async with aiohttp.ClientSession() as session:
            async with session.post(unload_url, json=unload_payload) as response:
                return response.status == 200
    except Exception as e:
        print(f"Error unloading LoRA adapter: {e}")
        return False


async def main_async():
    # Set random seed # 1. 设置随机种子
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate API keys based on selected search engine  # 2. 校验搜索引擎 API 密钥
    if args.search_engine == "bing" and not args.bing_subscription_key:
        print("Error: Bing search engine is selected, but --bing_subscription_key is not provided.")
        return
    elif args.search_engine == "serper" and not args.serper_api_key:
        print("Error: Serper search engine is selected, but --serper_api_key is not provided.")
        return
    elif args.search_engine not in ["bing", "serper"]: # Should be caught by choices, but good to have
        print(f"Error: Invalid search engine '{args.search_engine}'. Choose 'bing' or 'serper'.")
        return

    if args.jina_api_key == 'None':
        jina_api_key = None

    # Modified data loading section
    # 分支 A：单问题模式
    if args.single_question:
        # Create a single item in the same format as dataset items
        filtered_data = [{
            'Question': args.single_question,
        }]
        args.dataset_name = 'custom'  # Set dataset name to custom for single questions
    else:  # 分支 B：批量数据集模式
        # Original dataset loading logic
        if args.dataset_name == 'supergpqa':
            data_path = f'./data/SuperGPQA/{args.split}.json'
        elif args.dataset_name == 'webwalker':
            data_path = f'./data/WebWalkerQA/{args.split}.json'
        elif args.dataset_name == 'browsecomp':
            data_path = f'./data/BrowseComp/{args.split}.json'
        elif args.dataset_name == 'openthoughts':
            data_path = f'./data/OpenThoughts/{args.split}.json'
        elif args.dataset_name == 'webthinker':
            data_path = f'./data/WebThinker/{args.split}.json'
        elif args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'gaia', 'hle', 'limo', 'bamboogle', 'seal0', 'XbenchDS']:
            data_path = f'./data/{args.dataset_name.upper()}/{args.split}.json'
        elif args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            data_path = f'./data/QA_Datasets/{args.dataset_name}.json'
        else:
            data_path = f'./data/{args.dataset_name}.json'
        
        print('-----------------------')
        print(f'Using {args.dataset_name} {args.split} set.')
        print('-----------------------')

    # ---------------------- Caching Mechanism 缓存加载（与 bing_search.py 的桥梁）----------------------
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, f'{args.search_engine}_search_cache.json') # 缓存 {query: search_results}，避免对同一查询重复调用 Bing/Serper API
    if args.keep_links:
        url_cache_path = os.path.join(cache_dir, 'url_cache_with_links.json')
    else:
        url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches
    search_cache = json.load(open(search_cache_path)) if os.path.exists(search_cache_path) else {}
    url_cache = json.load(open(url_cache_path)) if os.path.exists(url_cache_path) else {}

    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # Define output directory
    if 'qwq' in args.model_name.lower():
        model_short_name = 'qwq'
        if 'webthinker' in args.model_name.lower():
            model_short_name = f'webthinker{args.model_name.split("webthinker")[-1]}'
    elif 'deepseek' in args.model_name.lower():
        if 'llama-8b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-8b'
        elif 'llama-70b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-70b'
        elif 'qwen-1.5b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-1.5b'
        elif 'qwen-7b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-14b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-14b'
        elif 'qwen-32b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-32b'
        if 'webthinker' in args.model_name.lower():
            model_short_name = f'webthinker{args.model_name.split("webthinker")[-1]}'
    else:
        model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')

    # output_dir = f'./outputs/{args.dataset_name}.{model_short_name}.webthinker'
    output_dir = f'./outputs/{args.dataset_name}.{model_short_name}.webthinker'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the OpenAI client  # 主模型客户端（如 QwQ-32B）
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
    )
    # Initialize auxiliary client   辅助模型客户端（如 Qwen2.5-32B-Instruct）
    aux_client = AsyncOpenAI(
        api_key=args.aux_api_key,
        base_url=args.aux_api_base_url,
    )
    
    if not args.single_question:
        # Load and prepare data
        with open(data_path, 'r', encoding='utf-8') as json_file:
            filtered_data = json.load(json_file)

        if args.subset_num != -1:
            indices = list(range(len(filtered_data)))
            selected_indices = random.sample(indices, min(args.subset_num, len(indices)))
            filtered_data = [filtered_data[i] for i in selected_indices]

    # Prepare sequences  序列准备（构建推理链）
    active_sequences = []
    for item in filtered_data:
        question = item['Question']
        instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
        user_prompt = get_task_instruction_openqa(question)

        prompt = instruction + user_prompt  # user_prompt：具体的问题
        item['prompt'] = prompt
        active_sequences.append({
            'item': item,
            'prompt': prompt,
            'output': '',
            'finished': False,
            'history': [],
            'search_count': 0,
            'executed_search_queries': set(),
        })

    # Initialize batch output records
    batch_output_records = []
    start_time = time.time()

    # Create semaphore for concurrent API calls
    semaphore = asyncio.Semaphore(args.concurrent_limit)

    # Load LoRA adapter if specified
    if args.lora_name and args.lora_path:
        print(f"Loading LoRA adapter '{args.lora_name}' from {args.lora_path}")
        success = await load_lora_adapter(args.api_base_url, args.lora_name, args.lora_path)
        if not success:
            print("Failed to load LoRA adapter")
            return
        else:
            print("LoRA adapter loaded successfully")

    try:
        # Process all sequences concurrently
        tasks = [  # 为每个问题创建 process_single_sequence 任务
            process_single_sequence(
                seq=seq,
                client=client,
                aux_client=aux_client,
                semaphore=semaphore,
                args=args,
                search_cache=search_cache,
                url_cache=url_cache,
                batch_output_records=batch_output_records
            )
            for seq in active_sequences
        ]

        # Run all sequences concurrently with progress bar  # 并发执行，带进度条
        with tqdm(total=len(tasks)) as pbar:
            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result
            
            tracked_tasks = [track_progress(task) for task in tasks]
            completed_sequences = await asyncio.gather(*tracked_tasks)
    finally:
        # Unload LoRA adapter if it was loaded
        if args.lora_name:
            print(f"Unloading LoRA adapter '{args.lora_name}'")
            await unload_lora_adapter(args.api_base_url, args.lora_name)
            print("LoRA adapter unloaded successfully")

    total_time = time.time() - start_time

    if args.eval:  # 评估模式：运行评估脚本
        # Prepare output list and save results
        DOMAIN_FIELDS = ['Level', 'level', 'category', 'High-level domain', 'difficulty_level', 'field', 'problem_topic']
        output_list = [seq['output'] for seq in completed_sequences]
        print("开始运行run_evaluation")
        # run_evaluation(filtered_data, [seq['original_prompt'] for seq in completed_sequences], output_list, args.dataset_name, output_dir, total_time, args.split)

        output_metrics_path = 'result.metrics.json'
        output_metrics_overall_path = 'result.metrics.overall.json'
        await  run_evaluation(filtered_data, [seq['original_prompt'] for seq in completed_sequences], output_list, 'math', output_dir, output_metrics_path, output_metrics_overall_path, 
                              use_llm=True, extract_answer=True, domain_fields = DOMAIN_FIELDS, api_base_url='http://localhost:1826/v1', model_name='qwen3.5-9b')
    else:
        t = time.localtime()
        random_num = str(random.randint(0, 99)).zfill(2)
        result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{random_num}.json'

        for item, seq in zip(filtered_data, completed_sequences):
            item['prompt'] = seq['original_prompt']
            item['Output'] = seq['output']
            item['WebExplorer'] = seq['web_explorer']  # Updated field name
            
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    # Save caches
    save_caches()
    print("Process completed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
