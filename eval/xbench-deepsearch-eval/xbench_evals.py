# 该脚本是一个自动化评估工具，用于在加密的问答数据集上运行大语言模型（如 DeepSeek-V3），收集多轮生成和评分结果，便于后续比较模型性能或进行投票集成。
import base64
import csv
import argparse
from tqdm import tqdm
from eval_grader import eval_and_grade_question  # 使用指定的模型（默认 deepseek-v3）对问题生成响应、提取答案、评分，并记录各种指标（如是否超长、内容过滤、错误信息等）


def xor_decrypt(data, key):
    """
    XOR decrypt data with a key
    """
    key_bytes = key.encode('utf-8')
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-v3")
    parser.add_argument("--dataset", type=str, default="data/ScienceQA.csv")
    parser.add_argument("--n-repeats", type=int, default=5) # 指定对每个问题重复评估的次数（默认 5 次）
    args = parser.parse_args()
    n_repeats = args.n_repeats

    with open(args.dataset, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        questions = []
        for question in reader:
            key = question["canary"]
            question["prompt"] = xor_decrypt(base64.b64decode(question["prompt"]), key).decode('utf-8')
            question["answer"] = xor_decrypt(base64.b64decode(question["answer"]), key).decode('utf-8')
            questions.append(question)

    header = ["id", "prompt", "type", "answer"]
    for n_repeat in range(n_repeats):
        n = str(n_repeat + 1)
        header.append("response-" + n)  # 模型生成的原始回答文本
        header.append("extracted-answer-" + n)  # 从回答中提取出的最终答案（通过正则或 LLM Judge 解析）
        header.append("score-" + n)  # 单次回答是否正确（1 = 正确，0 = 错误）.如果提取的 extracted-answer 与正确答案完全一致 → 直接得 1 分。否则调用 LLM Judge（gemini-2.0）判断是否一致（支持数值微小误差等），根据 Judge 的“结论”字段（正确/错误）给分。
        header.append("score-reason-" + n)  # 评分理由（来自 LLM Judge 的解释，或直接匹配成功的说明）
        header.append("exceed-length-" + n)  # 是否因长度超限而被截断（Y/空）
        header.append("content-filter-" + n)  # 是否触发安全过滤（Y/空）
        header.append("error-" + n)  # 是否发生 API 错误（Y/空）
    header.append("avg_score") # 多次重复的平均正确率（0~1 之间）
    header.append("best_of_n") # 多次重复中最佳单次得分（0 或 1）
    header.append("majority_vote_answer")  # 	多次回答中出现次数最多的答案
    header.append("majority_vote_score")  # 多数投票答案是否正确
    header.append("avg_cost (RMB)")  # 单次 API 调用的平均成本（人民币）
    header.append("avg_time（s)")  # 	单次响应时间的平均值（秒）

    csv_filename = f"{args.model}_results.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for question in tqdm(questions):
            result = eval_and_grade_question(question, model=args.model, n_repeats=n_repeats)
            writer.writerow(result)


if __name__ == "__main__":
    main()
