import os
import json
from openai import OpenAI
from Prompt_Pos import Doctor_prompt, Patient_prompt, Tongjing
import re


# ====== 基础配置 ======
client = OpenAI(
    api_key="sk-f01f4e0e0f3a410e80feb5df479e5a2a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = "qwen3-235b-a22b-thinking-2507"
OUTPUT_JSONL = "Dataset/Pos_Dataset/Dialogue_Dataset.jsonl"


# ====== 写入一行JSON ======
def append_jsonl(record: dict, path: str = OUTPUT_JSONL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# 构建某个证型的对话场景
def build_syndrome_background(syndrome_name: str) -> str:
    return (
        f"【场景约束】本条样本的目标证型：『{syndrome_name}』。\n"
        f"- 仅从该证型在 Tongjing 中的“一级/二级/三级症状”里进行描述与提问；禁止跨证型取词。\n"
        f"- 若患者表述出现库外或他证型症状，请提示“该描述不在可选范围”，并给出本证型内的候选症状供选。\n"
        f"- 最后一轮医生仅做证型预测与方案给出，若与目标证型高度符合则以该证型为主。"
    )


# ====== 生成一轮对话 ======
def gen_dialogue(role: str, history: list[str], judge: bool = False, target_syndrome: str = "") -> str:
    # 确认当前说话角色
    if role == "doctor":
        system_prompt = Doctor_prompt
        Play_role = "医生"
    else:
        system_prompt = Patient_prompt
        Play_role = "患者"

    # 判断是否有历史对话
    if history:
        dialogue_text = "\n".join(history)
    else:
        # 首轮在对话顶部加入“目标证型”场景约束
        if target_syndrome:
            target_syndrome_prompt = build_syndrome_background(target_syndrome)
            dialogue_text = target_syndrome_prompt + "\n"
        else:
            dialogue_text = "（尚无对话）"

    # 根据上下文记录，编写提示
    speaker_prompt = f"""以下是问诊对话到目前为止（逐行）：
{dialogue_text}
现在轮到你发言：请仅以「{Play_role}」身份输出下一段内容，不要写对方的台词，不要加引号或前缀。"""

    # 医生最后一轮，强制预测提示
    if role == "doctor" and judge:
        speaker_prompt = speaker_prompt + """
【这是最后一次医生发言】请仅进行证型预测与方案给出：
1) 从 Tongjing 的证型集合中仅选择1个“可能证型”，并给出辨证要点（仅使用已出现且来自知识库的症状词）。
2) 依据所选证型，给出 Tongjing 中的“治疗方案”（可做简要转述，不得编造）。
禁止继续提问；禁止加入任何角色前缀；只输出内容本身。"""

    # 根据对话历史，调用大模型生成回复
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": speaker_prompt},
        ],
        extra_body={"enable_thinking": True},
    )
    content = completion.choices[0].message.content.strip()

    # —— 统一清洗：去掉开头的“患者/医生”前缀（更稳妥）
    content = _strip_role_prefix(content)

    return content


# 取出生成对话内容中可能出现的“患者：”，“医生：”
def _strip_role_prefix(text: str) -> str:
    """
    去掉模型输出开头可能带的角色前缀，如：
    患者: ... / 患者：... / 医生: ... / 医生：...
    仅清理开头一次，不动正文其他位置。
    """
    return re.sub(r'^(?:患者|医生)\s*[:：]\s*', '', text, count=1)


# ====== 模拟对话流程，生成一条样本 ======
def multi_round_dialogue(dialogue_id: int, rounds: int = 5, target_syndrome: str = ""):
    history = []
    pairs = []
    current_turn = 1

    while current_turn <= rounds:
        # 患者
        patient = gen_dialogue("patient", history, False, target_syndrome=target_syndrome)
        history.append(f"患者: {patient}")

        # 医生
        last_round = (current_turn == rounds)
        doctor = gen_dialogue("doctor", history, last_round, target_syndrome=target_syndrome)
        history.append(f"医生: {doctor}")

        # 当前轮次+1
        current_turn += 1
        # 添加问答对
        pairs.append({"user": patient, "bot": doctor})

    record = {
        "task": "TJ",
        "id": dialogue_id,
        "history": pairs
    }
    append_jsonl(record)
    # 方便后续继续读取对话历史
    return history


# ====== 批量生成对话数据集 ======
def generate_many(nums: int = 10, rounds: int = 5):
    # 取出痛经的所有证型
    syndromes = list(Tongjing.keys())
    syndromes_len = len(syndromes)
    for i in range(1, nums + 1):
        target = syndromes[(i - 1) % syndromes_len]  # 均匀循环
        print(f"==================== 生成第 {i}/{nums} 条样本（目标证型：{target}） ====================")
        multi_round_dialogue(i, rounds=rounds, target_syndrome=target)


if __name__ == "__main__":
    # 每个证型两条样本
    generate_many(nums=1, rounds=4)
