import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Union
from urllib.parse import urljoin, urlparse
import aiohttp
import asyncio
import chardet
import random
from aiohttp_socks import SocksConnector


# ----------------------- Set your WebParserClient URL -----------------------
WebParserClient_url = None  # 当常规爬虫方式（requests + BeautifulSoup 或 Jina AI）失败或遇到限制时，作为降级方案（fallback）调用远程解析服务。


# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)
proxies = {"http": "socks5h://127.0.0.1:1824", "https": "socks5h://127.0.0.1:1824"}

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

class WebParserClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化Web解析器客户端
        
        Args:
            base_url: API服务器的基础URL，默认为本地测试服务器
        """
        self.base_url = base_url.rstrip('/')
        
    def parse_urls(self, urls: List[str], timeout: int = 120) -> List[Dict[str, Union[str, bool]]]:
        """
        发送URL列表到解析服务器并获取解析结果
        
        Args:
            urls: 需要解析的URL列表
            timeout: 请求超时时间，默认20秒
            
        Returns:
            解析结果列表
            
        Raises:
            requests.exceptions.RequestException: 当API请求失败时
            requests.exceptions.Timeout: 当请求超时时
        """
        endpoint = urljoin(self.base_url, "/parse_urls")
        response = requests.post(endpoint, json={"urls": urls}, timeout=timeout, proxies=proxies)
        response.raise_for_status()  # 如果响应状态码不是200，抛出异常
        
        return response.json()["results"]


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))  # Precision = |snippet_words ∩ sentence_words| / |sentence_words|
    recall = intersection / float(len(true_set))  # Recall    = |snippet_words ∩ sentence_words| / |snippet_words|
    return 2 * (precision * recall) / (precision + recall)  # F1 = 2 * P * R / (P + R)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 3000) -> Tuple[bool, str]:
    # 在网页全文中找到这段摘要对应的原始句子，并截取该句子前后一定长度的上下文返回。
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    print(f"google_search.py里面的 extract_snippet_with_context() 传入的len(full_text)={len(full_text)}")
    print(f"google_search.py里面的 extract_snippet_with_context() 传入的snippet={snippet}")
    print(f"google_search.py里面的 extract_snippet_with_context() 传入的context_chars={context_chars}")
    try:
        full_text = full_text[:100000]  # 全文截断到 100,000 字符（防止过长处理）

        snippet = snippet.lower() # 转小写、去除标点 → 分词得到 snippet_words（集合）
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        #  全文分句（使用 nltk.sent_tokenize）
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize  

        for sentence in sentences:  # 遍历每句话，计算 F1 Score
            key_sentence = sentence.lower()  # 句子同样：小写 → 去标点 → 分词
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)  
            if f1 > best_f1: # 保留 F1 > 0.2 且分数最高的句子（best_sentence）
                best_f1 = f1
                best_sentence = sentence
        # 是否找到 best_sentence?
        if best_sentence: # 是 → 定位句子在全文中的位置
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            # if end_index - start_index < 2 * context_chars:
            #     end_index = min(len(full_text), start_index + 2 * context_chars)
            context = full_text[start_index:end_index]  # 截取 [pos - context_chars, pos + len(sentence) + context_chars]
            print(f"google_search.py里面的 extract_snippet_with_context() 找到匹配snippet的句子！！！成功返回best_sentence：{best_sentence}\n返回的content：{context}")
            return True, context  # 返回 (True, context)
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            print(f"❌❌❌google_search.py里面的 extract_snippet_with_context() 找不到匹配snippet的句子，失败了！！！！！！！,退化为前 N 字符")
            return False, full_text[:context_chars * 2]  # 返回 (False, full_text[:context_chars*2])  # 退化为前 N 字符
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None, keep_links=False):
    """  当 use_jina=False 时，它使用传统的 requests + BeautifulSoup 方案获取网页内容，并内置了多级降级策略（错误检测 → WebParserClient 备用 → 纯文本提取）。
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        jina_api_key (str): API key for Jina.
        snippet (Optional[str]): The snippet to search for.
        keep_links (bool): Whether to keep links in the extracted text.

    Returns:
        str: Extracted text or context.
    """
    try:
        print(f'google_search.py里面的 extract_text_from_url()开始爬取url={url},use_jina={use_jina},jina_api_key={jina_api_key},snippet={snippet}')
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers, proxies=proxies,).text
            # Remove URLs  。Jina 走 r.jina.ai 拿 markdown 并用正则去掉其中的 URL
            response.raise_for_status()  # 4xx/5xx 会抛异常
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
            print(f"google_search.py里面的 extract_text_from_url()里面的 jina爬取 text={text} ")
        else:  # （use_jina=False 分支）
            if 'pdf' in url:  # 如果 URL 包含 "pdf"，则调用 extract_pdf_text 函数处理 PDF 文件。
                return extract_pdf_text(url)

            try:  # 发送 HTTP 请求
                response = session.get(url, timeout=30, proxies=proxies)
                response.raise_for_status()
                
                # 添加编码检测和处理
                if response.encoding.lower() == 'iso-8859-1':  # 如果响应编码是 iso-8859-1，则尝试从内容中检测正确的编码。
                    # 尝试从内容检测正确的编码
                    response.encoding = response.apparent_encoding
                
                try:  # 先用 lxml 解析器，失败则回退到 html.parser。
                    soup = BeautifulSoup(response.text, 'lxml')
                except Exception:
                    soup = BeautifulSoup(response.text, 'html.parser')

                # Check if content has error indicators 。检查页面是否包含错误指示（如 "limit exceeded"、"Please turn on Javascript" 等），且内容少于64个词，或内容为空。
                has_error = (any(indicator.lower() in response.text.lower() for indicator in error_indicators) and len(response.text.split()) < 64) or response.text == ''
                if has_error:  # 检测到错误
                    print(f"extract_text_from_url 爬虫 检测到错误，降级到WebParserClient")
                    if WebParserClient_url is None:  # WebParserClient_url 是否已设置?  否 → 直接返回错误字符串
                        # If WebParserClient is not available, return error message
                        return f"Error extracting content: {str(e)}"
                    # If content has error, use WebParserClient as fallback
                    client = WebParserClient(WebParserClient_url)  # 是 → 调用远程解析服务 (POST /parse_urls)
                    results = client.parse_urls([url])
                    if results and results[0]["success"]:
                        text = results[0]["content"]
                    else:
                        error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                        return f"WebParserClient error: {error_msg}"
                else:  # 内容提取（两种模式）
                    if keep_links:  # 遍历所有元素，保留文本和链接的 Markdown 格式 [text](url)：
                        # Clean and extract main content
                        # Remove script, style tags etc
                        for element in soup.find_all(['script', 'style', 'meta', 'link']):
                            element.decompose()

                        # Extract text and links
                        text_parts = []
                        for element in soup.body.descendants if soup.body else soup.descendants:
                            if isinstance(element, str) and element.strip():
                                # Clean extra whitespace
                                cleaned_text = ' '.join(element.strip().split())
                                if cleaned_text:
                                    text_parts.append(cleaned_text)
                            elif element.name == 'a' and element.get('href'):
                                href = element.get('href')
                                link_text = element.get_text(strip=True)
                                if href and link_text:  # Only process a tags with both text and href
                                    # Handle relative URLs
                                    if href.startswith('/'):
                                        base_url = '/'.join(url.split('/')[:3])
                                        href = base_url + href
                                    elif not href.startswith(('http://', 'https://')):
                                        href = url.rstrip('/') + '/' + href
                                    text_parts.append(f"[{link_text}]({href})")

                        # Merge text with reasonable spacing
                        text = ' '.join(text_parts)
                        # Clean extra spaces
                        text = ' '.join(text.split())
                    else:  # soup.get_text() 提取纯文本（默认）
                        text = soup.get_text(separator=' ', strip=True)
            except Exception as e:
                if WebParserClient_url is None:
                    # If WebParserClient is not available, return error message
                    return f"Error extracting content: {str(e)}"
                # If normal extraction fails, try using WebParserClient
                client = WebParserClient(WebParserClient_url)
                results = client.parse_urls([url])
                if results and results[0]["success"]:
                    text = results[0]["content"]
                else:
                    error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                    return f"WebParserClient error: {error_msg}"

        print(f"google_search.py里面的 extract_text_from_url()里面 爬虫到 url:{url}  fetch结果：{len(text)}")
        if snippet: # 是否提供了 snippet?
            success, context = extract_snippet_with_context(text, snippet)
            print(f"bing_search.py里面的 extract_text_from_url 循环 里面的 extract_snippet_with_context结果：success={success}，context={json.dumps(context, indent=4, ensure_ascii=False)}")
            if success: # 成功 → 返回定位后的上下文
                print(f"google_search.py里面的 extract_text_from_url()里面的 fetch全文，【extract_snippet_with_context抽取】后的结果：{context}")
                return context  # 失败 → 返回完整文本
            else: # 没提供snippet  返回 text[:20000]（前 2 万字符）
                return text
        else:
            # If no snippet is provided, return directly
            return text[:20000]  # 如果没有 snippet，返回前 20000 字符
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(urls, max_workers=32, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None, show_progress=False, keep_links=False):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        jina_api_key (str): API key for Jina.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.
        show_progress (bool): Whether to show progress bar with tqdm.
        keep_links (bool): Whether to keep links in the extracted text.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, snippets.get(url) if snippets else None, keep_links): url
            for url in urls
        }
        completed_futures = concurrent.futures.as_completed(futures)
        if show_progress:
            completed_futures = tqdm(completed_futures, desc="Fetching URLs", total=len(urls))
            
        for future in completed_futures:
            url = futures[future]
            try:
                data = future.result()
                # results[url] = data
                # ========== 新增：检查是否抓取失败 ==========
                print(f"jina爬取结果：data={data}")
                if data.code == 402:
                    print(f"[Warning] jina 爬取失败 Failed to fetch {url}: {data}")
                elif data and not data.startswith("Error"):
                    print(f"jina爬取成功！！！！！！！！！！！,data={data}")
                    results[url] = data
                else:
                    # failed_urls.append(url)
                    print(f"[Warning] jina 爬取失败 Failed to fetch {url}: {data}")
                # ============================================
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            # time.sleep(0.1)  # Simple rate limiting
    return results

def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20, proxies=proxies):
    """
    Perform a search using the Bing Web Search API with a set timeout.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int or float or tuple): Request timeout in seconds.
                                         Can be a float representing the total timeout,
                                         or a tuple (connect timeout, read timeout).

    Returns:
        dict: JSON response of the search results. Returns empty dict if all retries fail.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=timeout, proxies=proxies)
            response.raise_for_status()  # Raise exception if the request failed
            search_results = response.json()
            return search_results
        except Timeout:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search request timed out ({timeout} seconds) for query: {query} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Timeout occurred, retrying ({retry_count}/{max_retries})...")
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search Request Error occurred: {e} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Request Error occurred, retrying ({retry_count}/{max_retries})...")
        time.sleep(1)  # Wait 1 second between retries
    
    return {}  # Should never reach here but added for completeness


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    print(f"开始执行 extract_pdf_text")
    try:
        response = session.get(url, timeout=20, proxies=proxies)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = full_text
        print(f"extract_pdf_text成功，返回内容：cleaned_text={cleaned_text}")
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

# 将 Bing Web Search API 返回的原始 JSON 响应，解析成结构化的文档列表，供后续流程使用。
def extract_relevant_info(search_results):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    print(f'开始执行 bing _search.py里面的 extract_relevant_info ')
    if 'webPages' in search_results and 'value' in search_results['webPages']:  # 检查 'webPages' 键是否存在
        for id, result in enumerate(search_results['webPages']['value']):  # 遍历 search_results['webPages']['value'] 中的每个 result
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': result.get('name', ''),
                'url': result.get('url', ''),
                'site_name': result.get('siteName', ''),
                'date': result.get('datePublished', '').split('T')[0],
                'snippet': result.get('snippet', ''),  # Remove HTML tags
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    print(f'extract_relevant_info执行完毕得到的 useful_info={json.dumps(useful_info, indent=4, ensure_ascii=False)}')
    return useful_info




async def bing_web_search_async(query, subscription_key, endpoint, market='en-US', language='en', timeout=20, proxies=proxies):
    """
    Perform an asynchronous search using the Bing Web Search API.

    Args:
        query (str): Search query.
        subscription_key (str): Subscription key for the Bing Search API.
        endpoint (str): Endpoint for the Bing Search API.
        market (str): Market, e.g., "en-US" or "zh-CN".
        language (str): Language of the results, e.g., "en".
        timeout (int): Request timeout in seconds.

    Returns:
        dict: JSON response of the search results. Returns empty dict if all retries fail.
    """
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    params = {
        "q": query,
        "mkt": market,
        "setLang": language,
        "textDecorations": True,
        "textFormat": "HTML"
    }

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = session.get(endpoint, headers=headers, params=params, timeout=timeout, proxies=proxies)
            response.raise_for_status()
            search_results = response.json()
            return search_results
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Bing Web Search Request Error occurred: {e} after {max_retries} retries")
                return {}
            print(f"Bing Web Search Request Error occurred, retrying ({retry_count}/{max_retries})...")
            time.sleep(1)  # Wait 1 second between retries

    return {}

class RateLimiter:
    def __init__(self, rate_limit: int, time_window: int = 60):
        """
        初始化速率限制器
        
        Args:
            rate_limit: 在时间窗口内允许的最大请求数
            time_window: 时间窗口大小(秒)，默认60秒
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """获取一个令牌，如果没有可用令牌则等待"""
        async with self.lock:
            while self.tokens <= 0:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.rate_limit,
                    self.tokens + (time_passed * self.rate_limit / self.time_window)
                )
                self.last_update = now
                if self.tokens <= 0:
                    await asyncio.sleep(random.randint(5, 30))  # 等待xxx秒后重试
            
            self.tokens -= 1
            return True

# 创建全局速率限制器实例
jina_rate_limiter = RateLimiter(rate_limit=130)  # 每分钟xxx次，避免报错

async def extract_text_from_url_async(url: str, session: aiohttp.ClientSession, use_jina: bool = False, 
                                    jina_api_key: Optional[str] = None, snippet: Optional[str] = None, 
                                    keep_links: bool = False) -> str:
    """Async version of extract_text_from_url"""
    print(f'开始执行 async def extract_text_from_url_async')
    print(f'bing_search.py里面的 async def extract_text_from_url_async()开始爬取url={url},use_jina={use_jina},jina_api_key={jina_api_key},snippet={snippet}')
    try:
        if use_jina:
            # 在调用jina之前获取令牌
            await jina_rate_limiter.acquire()
            
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            async with session.get(f'https://r.jina.ai/{url}', headers=jina_headers) as response:
                text = await response.text()
                if not keep_links:
                    pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                    text = re.sub(pattern, "", text)
                text = text.replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
                print(f"google_search.py里面的 extract_text_from_url()里面的 jina爬取 text={text} ")
        else:
            if 'pdf' in url:
                # Use async PDF handling
                text = await extract_pdf_text_async(url, session)
                print(f"是pdf，进入extract_pdf_text_async函数处理,返回 text[:10000]={text[:10000]}")
                return text[:10000]

            async with session.get(url) as response:
                # 检测和处理编码
                content_type = response.headers.get('content-type', '').lower()
                if 'charset' in content_type:
                    charset = content_type.split('charset=')[-1]
                    html = await response.text(encoding=charset)
                else:
                    # 如果没有指定编码，先用bytes读取内容
                    content = await response.read()
                    # 使用chardet检测编码
                    detected = chardet.detect(content)
                    encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
                    html = content.decode(encoding, errors='replace')
                
                # 检查是否有错误指示
                has_error = (any(indicator.lower() in html.lower() for indicator in error_indicators) and len(html.split()) < 64) or len(html) < 50 or len(html.split()) < 20
                # has_error = len(html.split()) < 64
                if has_error:
                    print(f"extract_text_from_url 爬虫 检测到错误，降级到WebParserClient")
                    if WebParserClient_url is None:
                        # If WebParserClient is not available, return error message
                        return f"Error extracting content: {str(e)}"
                    # If content has error, use WebParserClient as fallback
                    client = WebParserClient(WebParserClient_url)
                    results = client.parse_urls([url])
                    if results and results[0]["success"]:
                        text = results[0]["content"]
                    else:
                        error_msg = results[0].get("error", "Unknown error") if results else "No results returned"
                        return f"WebParserClient error: {error_msg}"
                else:
                    try:
                        soup = BeautifulSoup(html, 'lxml')
                    except Exception:
                        soup = BeautifulSoup(html, 'html.parser')

                    if keep_links:
                        # Similar link handling logic as in synchronous version
                        for element in soup.find_all(['script', 'style', 'meta', 'link']):
                            element.decompose()

                        text_parts = []
                        for element in soup.body.descendants if soup.body else soup.descendants:
                            if isinstance(element, str) and element.strip():
                                cleaned_text = ' '.join(element.strip().split())
                                if cleaned_text:
                                    text_parts.append(cleaned_text)
                            elif element.name == 'a' and element.get('href'):
                                href = element.get('href')
                                link_text = element.get_text(strip=True)
                                if href and link_text:
                                    if href.startswith('/'):
                                        base_url = '/'.join(url.split('/')[:3])
                                        href = base_url + href
                                    elif not href.startswith(('http://', 'https://')):
                                        href = url.rstrip('/') + '/' + href
                                    text_parts.append(f"[{link_text}]({href})")

                        text = ' '.join(text_parts)
                        text = ' '.join(text.split())
                    else:  # 直接使用 BeautifulSoup 的 get_text() 方法提取纯文本，用空格分隔，去除首尾空白。
                        text = soup.get_text(separator=' ', strip=True)

        # print('---\n', text[:1000])
        print(f"google_search.py里面的 extract_text_from_url()里面 爬虫到 url:{url}  fetch结果：{len(text)}")
        if snippet:  # 调用 extract_snippet_with_context 函数，在全文找到与 snippet 最匹配的句子及其上下文（前后各 3000 字符）。
            success, context = extract_snippet_with_context(text, snippet)
            print(f"google_search.py里面的 async def extract_text_from_url_async()里面的 fetch全文，【extract_snippet_with_context抽取】后的结果：success={success}，filtered_context={json.dumps(context, indent=4, ensure_ascii=False)}")
            return context if success else text
        else:
            return text[:50000]

    except Exception as e:
        return f"Error fetching {url}: {str(e)}"

# 并发批量爬取多个 URL
async def fetch_page_content_async(urls: List[str], use_jina: bool = False, jina_api_key: Optional[str] = None, 
                                 snippets: Optional[Dict[str, str]] = None, show_progress: bool = False,
                                 keep_links: bool = False, max_concurrent: int = 32) -> Dict[str, str]:
    """Asynchronously fetch content from multiple URLs."""
    print(f"开始执行 fetch_page_content_async")
    async def process_urls():
        # connector = aiohttp.TCPConnector(limit=max_concurrent)
        connector = SocksConnector.from_url('socks5://127.0.0.1:1824', limit=max_concurrent)  # ← 改成 SocksConnector
        timeout = aiohttp.ClientTimeout(total=240)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
            tasks = []
            for url in urls:
                task = extract_text_from_url_async(
                    url, 
                    session, 
                    use_jina, 
                    jina_api_key,
                    snippets.get(url) if snippets else None,
                    keep_links
                )
                tasks.append(task)
            
            if show_progress:
                results = []
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching URLs"):
                    result = await task
                    results.append(result)
            else:
                results = await asyncio.gather(*tasks)
            
            # return {url: result for url, result in zip(urls, results)}  # 返回字典而不是协程对象
            # ========== 新增：判断每个 URL 的爬取状态并输出日志 ==========
            results_dict = {}
            success_count = 0
            fail_count = 0
            
            for url, result in zip(urls, results):
                # 判断成功/失败的标准（与同步版 fetch_page_content 保持一致）
                print(f"jina爬取结果：result={result}")
                if getattr(result, 'code', None) == 402:
                    print(f"[Warning] jina 爬取失败 Failed to fetch {url}: {result}")
                if result and not result.startswith("Error"):
                    print(f"[✅ jina爬取成功！！！！！！！！！！！ ] {url} | 内容长度: {len(result)} 字符")
                    success_count += 1
                else:
                    print(f"[❌ [Warning] jina 爬取失败 Failed to fetch] {url} | 原因: {result[:200] if result else '空内容'}")
                    fail_count += 1
                results_dict[url] = result
            
            print(f"爬取完毕: 成功 {success_count} / 失败 {fail_count} / 总计 {len(urls)}")
            # ============================================================
            return results_dict# ← 必须 return，否则外部拿到 None

    return await process_urls()  # 确保等待异步操作完成

async def extract_pdf_text_async(url: str, session: aiohttp.ClientSession) -> str:
    """
    Asynchronously extract text from a PDF.

    Args:
        url (str): URL of the PDF file.
        session (aiohttp.ClientSession): Aiohttp client session.

    Returns:
        str: Extracted text content or error message.
    """
    print(f"开始执行 extract_pdf_text_async")
    try:
        async with session.get(url, timeout=30, proxies=proxies) as response:  # Set timeout to 20 seconds
            if response.status != 200:
                return f"Error: Unable to retrieve the PDF (status code {response.status})"
            
            content = await response.read()
            
            # Open the PDF file using pdfplumber
            with pdfplumber.open(BytesIO(content)) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
            
            # Limit the text length
            cleaned_text = full_text
            return cleaned_text
    except asyncio.TimeoutError:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def google_serper_search(query: str, api_key: str, timeout: int = 20):
    """
    Perform a search using the Google Serper API.

    Args:
        query (str): Search query.
        api_key (str): API key for Google Serper API.
        timeout (int or float or tuple): Request timeout in seconds.

    Returns:
        dict: JSON response of the search results. Returns empty dict if request fails.
    """
    print(f"开始执行 google_serper_search")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=timeout, proxies=proxies)
            response.raise_for_status()  # Raise exception if the request failed
            search_results = response.json()
            print(f"google_search.py里面的 google_web_search() 调用API结果 serper_data：{json.dumps(search_results, indent=4, ensure_ascii=False)}")
            return search_results
        except Timeout:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Google Serper API request timed out ({timeout} seconds) for query: {query} after {max_retries} retries")
                return {}
            print(f"Google Serper API Timeout occurred, retrying ({retry_count}/{max_retries})...")
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Google Serper API Request Error occurred: {e} after {max_retries} retries")
                return {}
            print(f"Google Serper API Request Error occurred, retrying ({retry_count}/{max_retries})...")
        time.sleep(1)  # Wait 1 second between retries
    
    return {}

def extract_relevant_info_serper(search_results):
    """
    Extract relevant information from Google Serper search results.

    Args:
        search_results (dict): JSON response from the Google Serper API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    print(f'开始 extract_relevant_info_serper')
    if 'organic' in search_results:
        for i, result in enumerate(search_results['organic']):
            # Try to extract domain for site_name, or leave empty
            site_name = ''
            try:
                site_name = urlparse(result.get('link', '')).netloc
                print(f"extract_relevant_info_serper link-{result.get('link', '')} 解析出来的 site_name={site_name}")
            except Exception:
                pass

            info = {
                'id': i + 1,
                'title': result.get('title', ''),
                'url': result.get('link', ''),
                'site_name': site_name, # Serper doesn't directly provide siteName, try to parse from URL
                'date': result.get('date', ''), # Serper might not always provide date
                'snippet': result.get('snippet', ''),
                'context': ''  # Reserved field
            }
            useful_info.append(info)
    print(f"extract_relevant_info_serper 返回的useful_info: {json.dumps(useful_info, indent=4, ensure_ascii=False)}")
    return useful_info

async def google_serper_search_async(query: str, api_key: str, timeout: int = 20):
    """
    Perform an asynchronous search using the Google Serper API.

    Args:
        query (str): Search query.
        api_key (str): API key for Google Serper API.
        timeout (int): Request timeout in seconds for each attempt.

    Returns:
        dict: JSON response of the search results. Returns empty dict if all retries fail.
    """
    print(f"开始进行 async def google_serper_search_async")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers_serper = {  # Use a different name to avoid conflict with global headers
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    max_retries = 5  # Consistent with bing_web_search_async
    retry_count = 0
    
    # Create a timeout object for aiohttp
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    # async with aiohttp.ClientSession() as session:
    connector = SocksConnector.from_url('socks5://127.0.0.1:1824')
    async with aiohttp.ClientSession(connector=connector) as session:
        while retry_count < max_retries:
            try:
                async with session.post(url, headers=headers_serper, data=payload, timeout=client_timeout) as response:
                    response.raise_for_status()  # Raise AIOHTTPError for bad status (4xx or 5xx)
                    search_results = await response.json()
                    print(f"async def google_serper_search_async 的搜索结果 {json.dumps(search_results, indent=4, ensure_ascii=False)}")
                    return search_results
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Google Serper API request timed out ({timeout} seconds) for query: {query} after {max_retries} retries")
                    return {}
                print(f"Google Serper API Timeout occurred, retrying ({retry_count}/{max_retries})...")
            except aiohttp.ClientError as e: # Covers ConnectionError, ClientResponseError, etc.
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Google Serper API Request Error occurred: {e} after {max_retries} retries")
                    return {}
                print(f"Google Serper API Request Error occurred ({e}), retrying ({retry_count}/{max_retries})...")
            
            if retry_count < max_retries:
                await asyncio.sleep(1)  # Wait 1 second between retries (non-blocking)
    
    return {}

# ------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    # Define the query to search
    query = "Structure of dimethyl fumarate"

    # --- CHOOSE SEARCH TYPE ---
    # search_type = "bing"
    search_type = "serper"  # or "bing"

    search_results = {}
    extracted_info = []

    if search_type == "bing":
        # Set your API key for Bing Web Search API
        BING_SUBSCRIPTION_KEY = "YOUR_BING_SUBSCRIPTION_KEY"
        bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
        
        # Perform the search
        print("Performing Bing Web Search...")
        search_results = bing_web_search(query, BING_SUBSCRIPTION_KEY, bing_endpoint)
        
        print("Extracting relevant information from Bing search results...")
        extracted_info = extract_relevant_info(search_results)

    elif search_type == "serper":
        # Set your API key for Google Serper API
        SERPER_API_KEY = "。。。"

        print("Performing Google Serper Search...")
        search_results = google_serper_search(query, SERPER_API_KEY)

        print("Extracting relevant information from Google Serper search results...")
        extracted_info = extract_relevant_info_serper(search_results)
        print(extracted_info)
    else:
        print(f"Unknown search_type: {search_type}. Please choose 'bing' or 'serper'.")
        exit()
    
    if not extracted_info:
        print("No search results to process.")
        exit()

    print("Fetching and extracting context for each snippet...")
    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_url(info['url'], use_jina=False)  # Get full webpage text
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(full_text, info['snippet'])
            print(f"run_search_o1.py里面的 relevant_info循环 里面的 extract_snippet_with_context结果：success={success}，context={json.dumps(context, indent=4, ensure_ascii=False)}")
            if success:
                info['context'] = context
            else:
                info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
        else:
            info['context'] = f"Failed to fetch full text: {full_text}"

    print("Your Search Query:", query)
    print("Final extracted information with context:")
    print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
