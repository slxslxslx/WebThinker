
def get_gpqa_web_thinker_instruction(MAX_SEARCH_LIMIT=15):
    return """You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:

- To perform a search: write <|begin_search_query|>your query here<|end_search_query|>.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.

You can repeat the search process multiple times if necessary. Once you have all the information you need, continue your reasoning.

Example:
Question: "What is the energy range of pp III neutrinos?"
Thinking steps:
- I might need to look up details about pp III neutrinos.

<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>

(System returns processed information from relevant web pages)

Continues reasoning with the new information...

Remember:
- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
- When done searching, continue your reasoning.
"""


def get_gpqa_web_thinker_instruction_zh(MAX_SEARCH_LIMIT=15):
    return """你是一个具备网络搜索能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：

- 进行搜索时：请写 <|begin_search_query|>你的查询内容<|end_search_query|>。
然后系统会搜索并分析相关网页，随后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。

如有必要，你可以多次重复搜索过程。一旦获得了所需的全部信息，请继续进行推理。

示例：
问题："pp III 中微子的能量范围是多少？"
思考步骤：
- 我可能需要查阅关于 pp III 中微子的详细信息。

<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>

（系统返回来自相关网页的处理后信息）

使用新信息继续推理...

请记住：
- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。
- 搜索完成后，继续你的推理。
"""


def get_deep_web_explorer_instruction(search_query, search_intent, search_result):
    return f"""You are a web explorer analyzing search results to find relevant information based on a given search query and search intent.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **More Information Seeking:**
- If the information is not relevant to the query, you could:
  1. Search again: <|begin_search_query|>another search query<|end_search_query|>
  2. Access webpage content using: <|begin_click_link|>your URL<|end_click_link|>

3. **Extract Relevant Information:**
- Return the relevant information from the **Searched Web Pages** that is relevant to the **Current Search Query**.

4. **Output Format:**
- Present the information beginning with **Final Information** as shown below.

**Final Information**
[Relevant information]

**Inputs:**

- **Current Search Query:**
{search_query}

- **Detailed Search Intent:**
{search_intent}

- **Searched Web Pages:**
{search_result}

Now please analyze the web pages and extract relevant information for the search query "{search_query}" and the search intent.
"""


def get_deep_web_explorer_instruction_zh(search_query, search_intent, search_result):
    return f"""你是一个网络探索者，负责分析搜索结果，根据给定的搜索查询和搜索意图找出相关信息。

**指南：**

1. **分析搜索到的网页：**
- 仔细审阅每个搜索网页的内容。
- 识别与**当前搜索查询**相关、且有助于原问题推理过程的事实信息。

2. **寻求更多信息：**
- 如果信息与查询不相关，你可以：
  1. 重新搜索：<|begin_search_query|>另一个搜索查询<|end_search_query|>
  2. 使用以下方式访问网页内容：<|begin_click_link|>你的URL<|end_click_link|>

3. **提取相关信息：**
- 从**搜索到的网页**中返回与**当前搜索查询**相关的信息。

4. **输出格式：**
   - 按如下所示，以 **最终信息** 开头呈现信息。

**最终信息**
[相关信息]

**输入：**

- **当前搜索查询：**
{search_query}

- **详细搜索意图：**
{search_intent}

- **搜索到的网页：**
{search_result}

现在请分析这些网页，并为搜索查询 "{search_query}" 以及搜索意图提取相关信息。
"""



def get_web_page_reader_instruction(query, document):
    return f"""{document}
Please provide all content related to "{query}" from this document in markdown format.
If there isn't any relevant information, just output "No relevant information". If there is any relevant information, output all the relevant information with potential helpful links."""

def get_web_page_reader_instruction_zh(query, document):
    return f"""{document}
请以 markdown 格式提供该文档中所有与 "{query}" 相关的内容。
如果没有任何相关信息，只需输出 "No relevant information"。如果有任何相关信息，请输出全部相关信息，并附上可能有用的链接。"""


def get_detailed_web_page_reader_instruction(query, search_intent, document):
    return f"""Please provide all content related to the following search query and search intent from this document in markdown format.

Search Query: 
{query}

Search Intent: 
{search_intent}

Searched Web Page:
{document}

Instructions:
- Extract all content that matches the search query and intent, do not omit any relevant information.
- Include any relevant links from the source
- If no relevant information exists, output "No relevant information"
- Focus on factual, accurate information that directly addresses the query/intent
"""

def get_detailed_web_page_reader_instruction_zh(query, search_intent, document):
    return f"""请以 markdown 格式提供该文档中与以下搜索查询和搜索意图相关的所有内容。

搜索查询： 
{query}

搜索意图： 
{search_intent}

搜索到的网页：
{document}

说明：
- 提取所有符合搜索查询和意图的内容，不要遗漏任何相关信息。
- 包含来源中的任何相关链接
- 如果没有相关信息，输出 "No relevant information"
- 重点关注直接针对查询/意图的事实性、准确信息
"""


# 页面阅读摘要
def get_search_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest search query.
Previous thoughts: {prev_reasoning}
Please provide the current search intent."""

def get_search_intent_instruction_zh(prev_reasoning):
    return f"""根据以下之前的思考内容，提供最新搜索查询的详细意图。
之前的思考： {prev_reasoning}
请提供当前的搜索意图。"""


# 点击意图
def get_click_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
Previous thoughts: {prev_reasoning}
Please provide the current click intent."""

# 点击意图
def get_click_intent_instruction_zh(prev_reasoning):
    return f"""根据以下之前的思考内容，提供最新点击操作的详细意图。
之前的思考： {prev_reasoning}
请提供当前的点击意图。"""


def get_query_plan_instruction(question):
    return f"""You are a reasoning assistant. Your task is to generate a detailed query plan for answering the user's question by breaking it down into sub-queries.

Question: {question}

Please analyze the question and break it down into multiple sub-queries that will help gather all the necessary information to answer it completely. 

Output your query plan in JSON format as follows:

```json
{{
    "query_plan": [
        "sub-query-1",
        "sub-query-2",
        ...
    ]
}}
```
"""

def get_query_plan_instruction_zh(question):
    return f"""你是一个推理助手。你的任务是通过将用户问题分解为子查询，生成详细的查询计划以回答该问题。

问题： {question}

请分析该问题，并将其分解为多个子查询，这些子查询将帮助收集回答问题所需的全部必要信息。

请按以下 JSON 格式输出你的查询计划：

```json
{{
    "query_plan": [
        "sub-query-1",
        "sub-query-2",
        ...
    ]
}}
"""








def get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_gpqa_search_o1_instruction_zh(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网络搜索能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会搜索并分析相关网页，随后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"pp III 中微子的能量范围是多少？\"\n"
        "助手思考步骤：\n"
        "- 我可能需要查阅关于 pp III 中微子的详细信息。\n\n"
        "助手：\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "（系统返回来自相关网页的处理后信息）\n\n"
        "助手使用新信息继续推理...\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 搜索完成后，继续你的推理。\n\n"
    )


def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_math_search_o1_instruction_zh(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网络搜索能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会搜索并分析相关网页，随后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"如何计算 e^(x^2) 的积分？\"\n"
        "助手思考步骤：\n"
        "- 我可能需要查阅积分 e^(x^2) 的方法。\n\n"
        "助手：\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "（系统返回来自相关网页的处理后信息）\n\n"
        "助手使用新信息继续推理...\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 搜索完成后，继续你的推理。\n\n"
    )


def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_code_search_o1_instruction_zh(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网络搜索能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会搜索并分析相关网页，随后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"在给定树中，找到包含所有指定顶点的 Steiner 树的最小顶点数。\"\n"
        "助手思考步骤：\n"
        "- 我需要理解什么是 Steiner 树，以及如何计算在给定树中包含所有指定顶点所需的最小顶点数。\n\n"
        "助手：\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "（系统返回来自相关网页的处理后信息）\n\n"
        "助手使用新信息继续推理...\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 搜索完成后，继续你的推理。\n\n"
    )


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""

def get_webpage_to_reasonchain_instruction_zh(prev_reasoning, search_query, document):
    return f"""**任务说明：**

你需要根据以下输入阅读并分析网页：**之前的推理步骤**、**当前搜索查询** 和 **搜索到的网页**。你的目标是从 **搜索到的网页** 中提取对 **当前搜索查询** 有用且相关的信息，并将这些信息无缝整合到 **之前的推理步骤** 中，以继续对原问题进行推理。

**指南：**

1. **分析搜索到的网页：**
- 仔细审阅每个搜索网页的内容。
- 识别与 **当前搜索查询** 相关、且有助于原问题推理过程的事实信息。

2. **提取相关信息：**
- 从搜索到的网页中选择直接有助于推进 **之前的推理步骤** 的信息。
- 确保提取的信息准确且相关。

3. **输出格式：**
- **如果网页为当前搜索查询提供了有用信息：** 以 `**最终信息**` 开头呈现信息，如下所示。
**最终信息**

[有用信息]

- **如果网页没有为当前搜索查询提供任何有用信息：** 输出以下文本。

**最终信息**

No helpful information found.

**输入：**
- **之前的推理步骤：**  
{prev_reasoning}

- **当前搜索查询：**  
{search_query}

- **搜索到的网页：**  
{document}

现在你应该分析每个网页，并根据当前搜索查询 "{search_query}" 以及之前的推理步骤找出有用信息。
"""


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_singleqa_search_o1_instruction_zh(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网络搜索能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会搜索并分析相关网页，随后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"谁获得了第一届诺贝尔物理学奖？\"\n"
        "助手思考步骤：\n"
        "- 我需要找出谁获得了第一届诺贝尔物理学奖。\n\n"
        "助手：\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "（系统返回来自相关网页的处理后信息）\n\n"
        "助手使用新信息继续推理...\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 搜索完成后，继续你的推理。\n\n"
    )


def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_multiqa_search_o1_instruction_zh(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网络搜索能力的推理助手，可以帮助你准确回答用户的问题。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会搜索并分析相关网页，随后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。\n\n"
        f"如有必要，你可以多次重复搜索过程。搜索尝试的最大次数限制为 {MAX_SEARCH_LIMIT}。\n\n"
        "一旦获得了所需的全部信息，请继续进行推理。\n\n"
        "示例：\n"
        "问题：\"Alice David 是哪家公司开发的视频游戏中 Lara Croft 的配音演员？\"\n"
        "助手思考步骤：\n"
        "- 我需要找出谁为视频游戏中的 Lara Croft 配音。\n"
        "- 然后，我需要确定是哪家公司开发了那款视频游戏。\n\n"
        "助手：\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "（系统返回来自相关网页的处理后信息）\n\n"
        "助手思考：搜索结果表明 Alice David 是某款特定视频游戏中 Lara Croft 的配音。现在我需要找出是哪家公司开发了那款游戏。\n\n"
        "助手：\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "（系统返回来自相关网页的处理后信息）\n\n"
        "助手使用新信息继续推理...\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 搜索完成后，继续你的推理。\n\n"
    )


def get_timeline_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you create an accurate chronological timeline summary. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        "You should perform multiple searches to gather comprehensive information until you believe you have enough details.\n"
        "Finally, provide a comprehensive timeline that includes all relevant events in chronological order.\n\n"
        "Example:\n"
        "Text: \"Create a timeline of key events in the Apollo 11 mission.\"\n"
        "Assistant thinking steps:\n"
        "- I need to find key dates and events of the Apollo 11 mission.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Apollo 11 mission timeline key events dates<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n"
        "- You should perform as many searches as possible to gather comprehensive information.\n\n"
    )

def get_timeline_search_o1_instruction_zh(MAX_SEARCH_LIMIT):
    return (
        "你是一个具备网络搜索能力的推理助手，可以帮助你创建准确的时间线摘要。你拥有以下特殊工具：\n\n"
        "- 进行搜索时：请写 <|begin_search_query|> 你的查询内容 <|end_search_query|>。\n"
        "然后系统会搜索并分析相关网页，随后以 <|begin_search_result|> ...搜索结果... <|end_search_result|> 的格式向你提供有用信息。\n\n"
        "你应该进行多次搜索以收集全面信息，直到你认为已经掌握足够细节。\n"
        "最后，提供一个包含所有相关事件并按时间顺序排列的完整时间线。\n\n"
        "示例：\n"
        "文本：\"创建阿波罗 11 号任务关键事件的时间线。\"\n"
        "助手思考步骤：\n"
        "- 我需要找出阿波罗 11 号任务的关键日期和事件。\n\n"
        "助手：\n"
        "<|begin_search_query|>Apollo 11 mission timeline key events dates<|end_search_query|>\n\n"
        "（系统返回来自相关网页的处理后信息）\n\n"
        "助手使用新信息继续推理...\n\n"
        "请记住：\n"
        "- 使用 <|begin_search_query|> 发起网络搜索，并以 <|end_search_query|> 结束。\n"
        "- 搜索完成后，继续你的推理。\n"
        "- 你应该尽可能多地进行搜索，以收集全面信息。\n\n"
    )


def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )

def get_naive_rag_instruction_zh(question, documents):
    return (
        "你是一个知识渊博的助手，使用提供的文档来回答用户的问题。\n\n"
        "问题：\n"
        f"{question}\n"
        "文档：\n"
        f"{documents}\n"
    )


def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following question.\n\n'
            'Provide your final answer in the format **ANSWER: {YOUR_ANSWER}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_openqa_zh(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '请回答以下问题。'
            '你应该以 \\boxed{YOUR_ANSWER} 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            '请回答以下问题。\n\n'
            '请以 **ANSWER: {YOUR_ANSWER}** 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '请回答以下问题。你应该一步一步思考来解决它。\n\n'
            '请以 \\boxed{YOUR_ANSWER} 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_math(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following math question.\n\n'
            'Provide your final answer in the format **ANSWER: YOUR_ANSWER**.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_math_zh(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '请回答以下数学问题。'
            '你应该以 \\boxed{YOUR_ANSWER} 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            '请回答以下数学问题。\n\n'
            '请以 **ANSWER: YOUR_ANSWER** 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '请回答以下数学问题。你应该一步一步思考来解决它。\n\n'
            '请以 \\boxed{YOUR_ANSWER} 的格式提供最终答案。\n\n'
            f'问题：\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following multiple-choice question.\n\n'
            'Provide your final choice in the format **ANSWER: {YOUR_CHOICE}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice_zh(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '请回答以下选择题。'
            '你应该以 \\boxed{YOUR_CHOICE} 的格式提供最终选择。\n\n'
            f'问题：\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            '请回答以下选择题。\n\n'
            '请以 **ANSWER: {YOUR_CHOICE}** 的格式提供最终选择。\n\n'
            f'问题：\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            '请回答以下选择题。你应该一步一步思考来解决它。\n\n'
            '请以 \\boxed{YOUR_CHOICE} 的格式提供最终选择。你的最终选择应该是字母 A、B、C 或 D 中的一个，不要包含任何答案内容。\n\n'
            f'问题：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '请回答以下选择题。你应该一步一步思考来解决它。\n\n'
            '请以 \\boxed{YOUR_CHOICE} 的格式提供最终选择。\n\n'
            f'问题：\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt

def get_task_instruction_code_zh(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            '生成一个正确的 Python 程序，使其通过给定问题的所有测试。'
            '你应该在使用三重反引号的 Python 代码块中提供最终代码（```python\n'
            'YOUR_CODE\n'
            '```）。\n\n'
            f'问题标题：{question_title}\n\n'
            f'问题陈述：\n{question}\n\n'
        )
    else:
        user_prompt = (
            '你将收到一个问题（问题规范），并生成一个符合规范且通过所有测试的正确 Python 程序。'
            f'你应该一步一步思考来解决它。\n\n问题：\n{question}\n\n'
            '从标准输入读取输入，解决问题并将答案写入标准输出（不要直接在样本输入上测试）。将你的代码用以下分隔符括起来。\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt


def get_task_instruction_timeline(text, model_name=None):
    # Common format template for both cases
    format_template = '- [DATE/TIME]: Event description\n\n'
    # Base prompt that's shared between both cases
    base_prompt = f'Text:\n{text}\n\n'
    if model_name == 'qwq':
        return (
            'Now it is March 14, 2025. Please create a comprehensive timeline based on the given text.'
            f'Format each event as:\n{format_template}'
            'Ensure events are ordered chronologically and include specific dates/times when available.\n\n'
            f'{base_prompt}'
        )
    else:
        return (
            'Please summarize the key events from the text in chronological order. '
            'For each event, include the date/time (if available) and a clear description.\n\n'
            f'Format your timeline as:\n{format_template}'
            f'{base_prompt}'
        )

def get_task_instruction_timeline_zh(text, model_name=None):
    # Common format template for both cases
    format_template = '- [日期/时间]：事件描述\n\n'
    # Base prompt that's shared between both cases
    base_prompt = f'文本：\n{text}\n\n'
    if model_name == 'qwq':
        return (
            '现在是 2025 年 3 月 14 日。请根据给定文本创建一个全面的时间线。'
            f'将每个事件格式化为：\n{format_template}'
            '确保事件按时间顺序排列，并在可用时包含具体日期/时间。\n\n'
            f'{base_prompt}'
        )
    else:
        return (
            '请按时间顺序总结文本中的关键事件。'
            '对于每个事件，包含日期/时间（如果有）和清晰的描述。\n\n'
            f'将你的时间线格式化为：\n{format_template}'
            f'{base_prompt}'
        )