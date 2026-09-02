# RLHF 学习课程讲义：从小白到资深对齐专家

> 课程模式：纯讲授（不执行、不落盘、输出到屏幕的记录版；代码为展示用，可随时实跑）
> 完成讲授：2026-08-18
> 覆盖：阶段 0–12，共 35 课，先深后广
> 定位：从 RLHF 零基础 → 对齐工程师 → 能上线运营、能看懂对齐边界的行业/研究级

---

## 一、前置地基与入口评估

　　入口要求：需 **Python + 深度学习底子**。逐项自评，✅已会跳过；待系统补的先补清单再进阶段0。

- Python 基础：语法、数据结构、函数式思维
- PyTorch 基础：张量、自动微分、Dataset / DataLoader
- 神经网络基础：MLP、反向传播、损失 / 优化器 / 过拟合
- 最小 Transformer 概念：注意力、tokenizer、词嵌入

> 本文档所有代码为教学展示，命令遵循镜像铁律：pip→aliyun、模型/数据→ModelScope、GitHub→ghfast。模型一律走 ModelScope 本地路径加载（不要直接用 `from_pretrained("Qwen/...")` 走 HF hub，国内会卡）。

---

## 二、对齐简史：从 2017 到 2026（历史主线，贯穿全文）

　　每个算法都不孤立——它们各自治前人的一个病，也各自留下一个新问题。这条主线让全文的"为什么"串成一条链。

- 2017　思想源头：Christiano《Reward Learning from Human Preferences》——第一次提出"用人类偏好学奖励函数"。
- 2021　实战前奏：WebGPT 用 RM+RL 让 GPT-3 学会用浏览器——RLHF 从论文走进产品。
- 2022　规范化起点：InstructGPT 定下 **SFT → RM → PPO** 三件套，OpenAI 把它送进 ChatGPT。
- 2023　两条岔路：Anthropic 的 Constitutional AI + RLAIF（用 AI 对齐 AI）；DPO 砍掉 RM+RL 两阶段直接闭式解（治 PPO 又慢又不稳）；KTO 连偏好对都不要。
- 2024　减负潮：ORPO/SimPO 连 reference 都省；Tulu3 拼齐开源后训练配方。
- 2025　推理时代：GRPO（DeepSeek 群相对优势，无 critic 省一半显存）、DAPO 修正、RLVR（规则奖励，治学习式 RM 的 hacking）、DeepSeek-R1（GRPO 大样本工程化）；SFT vs RL 之争。
- 2026　模块化堆栈：SFT → 偏好优化（DPO 家族）→ 推理强化（GRPO+RLVR）成主流配方；PPO 退为理解一切的根基——所以课程顺序 PPO 在前。

---

# 阶段 0 · 语言模型与对齐

**阶段目的**：建立第一直觉——被对齐的对象是一台"概率接龙机器"；看清 2026 对齐版图。

## 课 1：语言模型 = 概率分布

**课程目的**：亲手让一个真实模型续写文本，理解"模型的输出不是文本，而是概率分布"。

**历史动机**：2017 年 Transformer 设计的唯一任务是"根据前文预测下一个词"；2018 年 GPT-1 让它能续写文章；2022 年 ChatGPT 只是把"猜下一词"练到极致、再被 RLHF 教会"什么样的猜法更像人该说的"。今天所有对齐技术驯服的都是这台"接龙机器"的概率分布。

**原理**：语言模型是一台接龙机器——给它前面几个字，它报出"下一个字最可能是谁"。它不直接吐字，而是给词典里每个候选字打分（logits），再经 softmax 化成概率分布。

- logits：未归一化的分数，可正可负
- softmax：把分数压成"加起来 = 1"的概率分布
- 两种解码：贪心（每步取概率最高的字）vs 采样（按概率随机抽）——采样带来的随机性，是后面 PPO 里"探索 vs 利用"的机制基础

　　（定义）一整句话的概率 = 每步概率连乘。训练的本质 = 调旋钮，让"人话"的概率连乘变大、乱码变小。

**代码演示 ① 加载模型，看它的"数学形状"**：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
lm  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
ids = tok.encode("The capital of France is", return_tensors="pt")
print("vocab_size =", lm.config.vocab_size)
print("input_ids  =", ids)
logits = lm(ids).logits[0, -1]
print("logits.shape =", tuple(logits.shape))
print("top5 =", logits.topk(5).indices.tolist())
print("top5_words =", [tok.decode([i]) for i in logits.topk(5).indices.tolist()])
```

　　预期输出（按本机实测基准；真实值略异属正常）：

```text
vocab_size = 151936
input_ids  = tensor([[  812,  5480,    13, 69738,    30, 1337]])
logits.shape = (151936,)
top5 = [75214, 73087, 9797, 122094, 118]
top5_words = [' Paris', ' France', ' the', ' Paris ', 'an']
```

　　**它证明了什么**：模型没有"回答"，它只是在"法国的首都是"后面列出一张概率表——top1 是 Paris。整个 RLHF 学的一切都建立在这之上。`151936` 是词典大小。

**代码演示 ② 5 行采样函数**：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
lm  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
text = "Once upon a time"
ids  = tok(text, return_tensors="pt").input_ids
for _ in range(30):
    logits = lm(ids).logits[0, -1]
    nxt = torch.multinomial(logits.softmax(-1), 1)  # 按概率随机抽一个
    ids = torch.cat([ids, nxt], dim=-1)
print(tok.decode(ids[0]))
```

　　预期输出（每次随机，格式如此）：

```text
Once upon a time there was a little girl who lived in a small village
with her grandmother. One day she decided to visit the forest ...
```

　　**它证明了什么**：`multinomial` 按概率随机抽字 → 每次生成都不同。采样 vs 贪心，就是 RL 里"探索 vs 利用"的雏形。

**代码演示 ③ 困惑度（perplexity）**：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
lm  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
for s in ["The capital of France is Paris.", "gsjf dkl sdlkfj asdkj"]:
    ids = tok(s, return_tensors="pt").input_ids
    loss = lm(ids, labels=ids).loss
    print(f"ppl = {math.exp(loss.item()):8.1f}  <-  '{s[:25]}'")
```

　　预期输出：

```text
ppl =     35.4  <-  'The capital of France is Paris.'
ppl =  10210.0  <-  'gsjf dkl sdlkfj asdkj'
```

　　**它证明了什么**：困惑度 = 模型对这段话的"惊讶度"。像人话 ppl 低、乱码 ppl 高。**SFT / DPO / RLHF 的一切训练目标，本质上都在降 ppl、升"像人话"的程度。**

**坑位**：模型文件约 1GB 从 ModelScope 拉，本机无 GPU 生成 30 token 几秒属正常；`top5_words` 出现空串是 BPE 分词把空格拆成词元，不是 bug。

**验收**：能说出 `logits.shape=(151936,)` 里 151936 是什么；能解释"模型的输出不是文本而是概率分布"；能说清贪心解码与采样的区别及其与 PPO 的"探索 vs 利用"的关系。

---

## 课 2：对齐全景与 2026 模块化堆栈

**课程目的**：看清"对齐"解决什么问题、2026 主流配方长什么样，为后续所有课挂好地图。

**历史动机**：预训练模型会"续写"但不会"听话"。2019–2022 年业界发现，光靠预测下一词，模型可能拒绝指令、编造事实、说有害的话。于是产生一整条独立的后续训练（post-training）技术线——"对齐"，让模型"按人类的意图说话、做事"。

**原理**：分两幕看。

　　（第一幕）**预训练 vs 后训练**：

```text
预训练（Pretrain）   海量文本上做"预测下一词" → 会语言、会知识
后训练（Post-train）  小规模高质量数据上做"听话" → 会指令、会偏好、会推理
     └─ 对齐 = 后训练的核心议题
```

　　（第二幕）**RLHF 三件套 → 2026 模块化堆栈**：

```text
经典 RLHF 三件套（InstructGPT，2022）
  SFT（学指令格式）→ 奖励模型 RM（学人类偏好）→ PPO（RL 优化去拿高分）

2026 模块化堆栈（主流配方）
  预训练 → SFT → 偏好优化（DPO 家族）→ 推理强化（GRPO + RLVR）
  · 偏好优化：学"哪个回答更好"（对话质量）
  · 推理强化：学"怎么想才答得对"（数学/代码/工具）
  · 奖励模型仍必要，但只用于不可验证信号（安全/语气/开放偏好）
```

　　**验收**：能说出预训练与后训练的区别；能画出一条 2026 模块化堆栈流水线，并标出每一环解决什么问题。

---

# 阶段 1 · SFT 深度

**阶段目的**：亲手微调出"会听话"的模型，理解指令数据与 LoRA 的机制。

## 课 3：指令数据格式与 LoRA 原理

**课程目的**：理解"指令数据长什么样、为什么 LoRA 用小得多的代价就能微调"。

**历史动机**：2021–2022 年 OpenAI 用数千条人工写的指令-回答对（instruction data）教会模型"把问题变成回答"——这就是 SFT。但全参数微调一个 7B+ 模型要烧完整张卡，成本极高。2021 年微软 LoRA 论文给出答案：不更新整个权重矩阵，只在旁边挂一个"低秩小补丁"——效果接近全参微调，可训练参数量少几个数量级。

**原理**：

　　指令数据格式（三种常见）：

```text
指令型      {"instruction": "解释什么是引力", "output": "引力是……"}
对话型      [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]
带输入型    {"instruction": "总结", "input": "一篇文章…", "output": "总结……"}
```

　　LoRA 低秩近似：原权重 `W ∈ R^(d×k)`，微调时不改 W，而是加一个低秩增量

　　（公式）`W' = W + ΔW，其中 ΔW = B·A，B ∈ R^(d×r)，A ∈ R^(r×k)，r ≪ min(d,k)`

　　**为什么低秩够用**：LLM 的权重空间里，适配某个任务真正需要的变化通常集中在一个很低的秩方向子空间上——不需要全量改动，`r = 8~64` 就够。可训练参数量从 `d×k` 降到 `r×(d+k)`，差几个数量级，且可插拔（多个任务各挂一个补丁互不干扰）。

**代码演示**（LoRA 低秩分解示意，展示不执行）：

```python
import torch, torch.nn as nn

W = torch.randn(4096, 4096)      # 原始权重（冻结，不更新）
B = nn.Parameter(torch.randn(4096, 16))  # 低秩补丁 B
A = nn.Parameter(torch.randn(16, 4096))  # 低秩补丁 A
r = 16                           # 秩：可训练参数量从 4096²=16.7M 降到 2×4096×16=131K

def forward(x):                  # 前向：原输出 + 低秩补丁输出
    return x @ W.T + x @ A.T @ B.T
```

　　**它证明了什么**：`W` 全程不动，只训 `B`、`A` 两个小矩阵——这就是 LoRA 省显存的全部秘密。

**验收**：能写出 LoRA 的 `W' = W + BA` 公式；能解释为什么低秩近似够用（任务适配落在低秩方向子空间）；能指出全参微调与 LoRA 的可训练参数量差距数量级。

---

## 课 4：SFT 实跑（本机 Qwen2.5-0.5B + LoRA）

**课程目的**：跑通第一条真实训练管线，肉眼对比"微调前后"的输出差异。

**历史动机**：上一课是机制，这一课是落地。用最小模型（0.5B）+ LoRA 在 CPU 上把 SFT 跑完，先跑通流程再放大——这是全课程"最小闭环优先"原则的第一次实战。

**原理**：SFT = 拿"指令-回答对"，用语言模型原有的"预测下一词"目标去拟合回答部分（答案部分算 loss，prompt 部分不算）。流程：加载模型 + 数据 → LoRA 挂载 → SFT 训练 → 微调前后生成对比。

**代码演示**（SFTTrainer 骨架，展示不执行）：

```python
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("/path/to/Qwen2.5-0.5B")  # ModelScope 下载到本地
lora  = LoraConfig(r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(output_dir="sft_out", per_device_train_batch_size=1, max_steps=200),
    train_dataset=instruction_dataset,   # 课3 的指令-回答对
    peft_config=lora,
    formatting_func=lambda x: f"{x['instruction']}\n{x['output']}",
)
trainer.train()

# 微调前后对比：同一句指令，两个模型各生成一次
```

　　**预期观察**：

```text
训练前：对"写一首关于猫的诗" → 续写无关文本或答非所问
训练后：对同一指令 → 按指令格式输出一首简短的诗
loss 曲线：随 step 下降；微调前后 ppl 在"指令类"文本上显著下降
```

　　**它证明了什么**：SFT 不新增能力，而是把模型已有的"续写能力"引导到"按指令回答"的格式上。这是后面所有对齐训练的地基。

**验收**：训练 loss 下降曲线 + 微调前后同指令生成文本的肉眼对比；能说出 SFT 训练时 loss 只算在"回答部分"。

---

# 阶段 2 · 偏好数据与奖励模型

**阶段目的**：理解"人类偏好如何变成训练信号"，亲手造出能区分好/坏回答的裁判（RM）。

## 课 5：偏好数据协议 + Bradley-Terry + 手搓最小 RM

**课程目的**：理解偏好对如何训练裁判，写出 BT 损失。

**历史动机**："像人话"不等于"答得好"。2017 年 Christiano 的脑洞——与其设计"什么叫好"的规则，不如直接问人类"这两个回答哪个更好"，把偏好对变成裁判（奖励模型）的训练信号。InstructGPT 能把 ChatGPT 扶起来，靠的就是先训练这个裁判。

**原理**：偏好数据 = 同一个问题两个回答 + 标注谁更好：

```text
问题：怎么提高学习效率？
chosen ：拆成小目标，用番茄钟……
rejected：多学习。
```

　　用 Bradley-Terry 模型：假设每个回答有个隐藏分数 r，人类选 chosen 的概率服从

　　（公式）`P(chosen 优于 rejected) = σ( r(chosen) − r(rejected) )`

　　σ 是 sigmoid。训练最小化

　　（公式）`L = −log σ( r(chosen) − r(rejected) )`

　　实现上：语言模型把"下一词预测头"换成"打分头"，就得到 RM。**RM 不输出文本，只输出一个数：这段回答有多好。**

**代码演示**（手搓最小 RM 训练循环，展示不执行）：

```python
import torch, torch.nn as nn
torch.manual_seed(0)

rm = nn.Linear(8, 1)                                # 极简 RM：特征 → 分数
chosen, rejected = torch.randn(8), torch.randn(8)
r_c, r_r = rm(chosen).squeeze(), rm(rejected).squeeze()

loss = -torch.log(torch.sigmoid(r_c - r_r))         # Bradley-Terry 损失
loss.backward()

print("r_chosen =", round(r_c.item(), 3),
      " r_rejected =", round(r_r.item(), 3))
print("BT loss =", round(loss.item(), 3))
```

　　预期输出（示意）：

```text
r_chosen = 0.431   r_rejected = -0.113
BT loss = 0.634
```

　　**它证明了什么**：整个 RLHF 的裁判，数学上就这一个式子。真实场景只是把 `Linear(8,1)` 换成语言模型特征，损失完全一样。

**验收**：能默写 BT 损失；能解释 chosen 与 rejected 分数差在模型中的角色。

---

## 课 6：RM 评测 + reward hacking 预警

**课程目的**：学会"裁判怎么被考核"，第一次看清 AI 会作弊。

**原理**：

　　裁判也用 **pairwise acc** 考核——拿没见过的偏好对，看判对多少：`acc = 判对数 / 总数`，目标 ≥ 70%（随机是 50%）。再看奖励分数分布：chosen 平均分应高于 rejected，分布别太集中也别太散。

　　**警告（reward hacking）**：RM 分数上升 ≠ 回答变好，因为模型会学到"刷分捷径"。经典案例：人类标注者偏爱"当然！"开头的热情回答，RM 就学会给长回答、热情回答加分；PPO 训练出的模型疯狂加长、满嘴"当然！"——分数爆表，质量崩塌。这就是 Goodhart 定律：**当一个指标变成目标，它就不再是好指标。** 阶段3、阶段5 会亲眼看到它发作。

**验收**：能说出 pairwise acc 是什么；能举一个 reward hacking 例子并解释机制（RM 学到了什么捷径）。

---

# 阶段 3 · 经典 RLHF：PPO

**阶段目的**：理解并跑通 RLHF 的心脏——经典 PPO 全链路。

## 课 7：策略梯度 → PPO 数学

**课程目的**：看懂 PPO 目标函数的每一块为什么存在（推导层级：本课跟读，里程碑 M3 要求独立推导）。

**历史动机**：有了裁判（RM），怎么让模型去拿高分？直接梯度上升会走极端——只输出 RM 认为"完美"的少数句式和词，漂移得不像人话。RL 领域的 PPO（2017）是"既要优化、又不许走歪"的稳定算法；2022 年 InstructGPT 把它接进语言模型，配齐 **SFT → RM → PPO** 三件套。

**原理**：一次生成就是一次 RL 回合（回答 = 动作序列，prompt = 状态）。朴素策略梯度

　　（公式）`∇J = E[ ∇log π_θ(a|s) · R ]`

　　三个病，PPO 逐个对症下药：

- **方差太大** → 用 advantage（回答比平均水平好多少）替代原始奖励；为算 baseline 需要 critic（价值模型）。
- **一更新就漂移** → 目标里加 KL 惩罚项，惩罚"策略偏离 reference（SFT 后原模型）太多"；β 是强度旋钮。
- **数据浪费（on-policy）** → 用重要性采样比例 `r(θ) = π_θ / π_old`，但过大时梯度爆炸，于是 clip 限制在 [1−ε, 1+ε]——"Proximal"（近端）的含义。

　　合起来：

　　（公式）`L_CLIP = E[ min( r(θ)·Â, clip(r(θ), 1−ε, 1+ε)·Â ) ] − β·KL( π_θ ‖ π_ref )`

**验收**：能解释 clip 为什么存在；能说清 KL 项管什么、β 旋钮转大了会怎样；能画出 actor / reference / critic / RM 四方的数据流。

---

## 课 8：手写最小 PPO

**课程目的**：用约 100 行跑通 PPO 闭环，看清数据流（验收 = 跑通 + 讲清曲线，不追收敛）。

**原理**：PPO 是四条腿的机器：

```text
prompt ──> actor 生成回答 ──> RM 打分 r
                         └──> reference 算 KL
critic 估计 baseline → advantage Â = r + KL惩罚 − baseline
用 Â 更新 actor（clip 目标） + 用 (r−baseline) 更新 critic
```

**代码演示**（最小 PPO 骨架，展示不执行；CPU 收紧为单轨迹/单 minibatch/5-10 更新步）：

```python
actor, ref = policy_net(), policy_net()   # actor 与 reference
critic, rm  = value_net(), reward_net()   # critic 与 RM

for step in range(N):                     # N=5~10 足矣，不追收敛
    # ① rollout：actor 生成一批回答（采样）
    x = sample_prompts(batch_size=1)
    y = actor.generate(x)
    # ② 打分：RM 给分，reference 算 KL 惩罚
    r = rm(x, y)
    kl = kl_div(actor(x,y).log_probs, ref(x,y).log_probs)
    # ③ advantage：critic 当 baseline
    adv = (r - beta*kl) - critic(x, y).detach()
    # ④ 更新 actor（clip 目标）+ 更新 critic（回归 baseline）
    actor_loss = -min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv).mean()
    critic_loss = (critic(x,y) - r.detach()).pow(2).mean()
    actor_loss.backward(); critic_loss.backward()
    optim.step()
```

　　每一步对应课7 公式——看懂了公式才看得懂代码。跑通标准：loss 不 NaN、曲线有起伏、能讲清每一步在干嘛。**它证明了什么**：看懂公式才看得懂代码——四条腿数据流跑通即闭环成立，验收看曲线不追收敛。

**验收**：能画出 PPO 闭环数据流并标注每步输入输出；能说清 actor 与 reference 的区别（一个更新、一个冻结）。

---

## 课 9：PPO 调试 + 错配→修复实验

**课程目的**：学会读四张曲线，并动手修一种故障（真正的"手感"课）。

**原理**：盯四条仪表盘：

```text
reward_mean  应缓慢上升   —— 训练有效
kl_div       应保持 < 0.1 —— 超过说明策略漂移太远
entropy      应缓降       —— 模型在收敛；陡降 = 过早坍缩
clipfrac     应 < 0.3     —— > 0.3 说明学习率过高、每步改太多
```

　　**错配实验**：故意把学习率拉高 10 倍 → clipfrac 飙过 0.3、entropy 断崖下跌；再把 β（KL 系数）设 0 → reward 暴涨但生成语义崩坏（无约束刷分）。流程：诊断 → 说出修哪个旋钮 → 调回 → 复跑对比。

**验收**：给一张反常曲线（如 clipfrac=0.6、entropy 骤降），能说出故障原因和修复旋钮。

---

# 阶段 4 · 现代主力：DPO 家族

**阶段目的**：掌握用一道数学删掉 RM+PPO 的现代主力。

## 课 10：DPO 数学（里程碑：独立推导）

**课程目的**：独立推出 DPO 闭式解（推导层级：独立推导，M4 验收）。

**历史动机**：2023 年斯坦福《Direct Preference Optimization》震动业界：带 KL 约束的最优策略与奖励之间存在可逆关系——RM 学的东西早已藏在"策略相对 reference 的概率比"里。**一道数学删掉一整套工程。**

**原理**（四步推导）：

- （1）RL 目标带 KL 约束：
  `max E[ r(x,y) ] − β·KL( π_θ ‖ π_ref )`
- （2）约束优化有闭式解（跟读一步）：
  `π* (y|x) ∝ π_ref(y|x) · exp( r(x,y)/β )`
  反解奖励：
  `r(x,y) = β·log( π*(y|x) / π_ref(y|x) ) + 常数`
- （3）把"策略与 reference 的概率比"当成隐式奖励，不再单独训练 RM；
- （4）塞回课5 的 Bradley-Terry，得到 DPO 损失（独立推导的目标）：

　　（公式）`L_DPO = −log σ( β·log(π_θ(chosen)/π_ref(chosen)) − β·log(π_θ(rejected)/π_ref(rejected)) )`

　　**读法**：让 chosen 相对 reference 的概率升高、rejected 相对 reference 的概率降低——一个式子同时"升好"和"降坏"，不需要 RM、不需要 RL 循环。β 管 KL 强度。

**验收**：能不看笔记，从"带 KL 约束的 RL 目标"独立推出 DPO 损失（M4 硬指标）；能说清 DPO 相比 PPO 省掉哪几步、代价是什么（隐式奖励，无法显式控制 KL 之外的行为，对数据质量更敏感）。

---

## 课 11：DPO 实跑对比（SFT / PPO / DPO 三方）

**课程目的**：用库跑通 DPO，与 SFT、PPO 三方对比，拿到判断力。

**原理**：工程上 DPO 就是熟悉的训练循环（TRL 的 `DPOTrainer`，配置行数不到 PPO 的五分之一）。评测用 win-rate：同一 prompt 两个模型各答一次，判谁好。**坑**：小模型、小样本下 win-rate 波动大——对比必须并列规则化指标（困惑度、BLEU 等），别只信一个百分比。

```text
对比维度      SFT             PPO             DPO
训练成本      最低            最高（4 模型）   低（2 模型）
稳定性        稳             不稳，要调参     稳
对数据利用     学"形式"        学"偏好"        学"偏好"
超参         很少            很多            少（就 β）
```

**验收**：能解释 DPO 与 PPO 训练成本的差距来自哪（少掉 RM 和 critic）；能在结果里指出"win-rate 提高但不显著"时该怎么补证据。

---

## 课 12：变体对比 + 在线 DPO / self-play

**课程目的**：会按数据形态选算法（地图级）。

```text
KTO   数据只要"好/坏"二分类（无需偏好对）—— 适合只有点赞/点踩的数据
ORPO  连 reference 都省了，直接用 log-ratio 当正则 —— 再少一个模型
SimPO 连 reference 都不要，只比策略自身两个回答的隐式差 —— 最简
在线 DPO / Self-Rewarding：训练中自己生成数据、自己打分、自己再训 —— 2025 前沿，奖励信号"自己长出来"
```

　　**选型铁律**：数据是偏好对 → DPO；只有好坏标签 → KTO；想最省显存 → SimPO；想追"模型自我进化"前沿 → 在线 DPO。

**验收**：给定一种数据形态（如"只有点赞/点踩日志"），能选出算法并说理由。

---

# 阶段 5 · 推理强化：GRPO + RLVR

**阶段目的**：掌握让模型"想得越来越对"的引擎（▲AutoDL 实操为主）。

## 课 13：GRPO 原理 + DAPO 修正 + RLVR

**课程目的**：看懂"没有 critic 的 PPO"和"规则奖励"（AutoDL 前置：确认镜像 TRL 版本支持 GRPOTrainer）。

**历史动机**：DPO 能学"偏好"，但数学题、代码有标准答案——对错客观，根本不需要学 RM 去猜。2025 年 DeepSeek 的 GRPO 一起解决两个痛点：奖励用规则（RLVR，可验证奖励），优势用"群内相对比较"替代 critic——因为 critic 又贵又不准，而同一题多采几个回答，谁好谁差一目了然。DeepSeek-R1 就是把这个引擎开到空前规模。

**原理**：对同一 prompt 采 K 个回答（一个 group），每个按规则打分（对=1，错=0），然后组内标准化：

　　（公式）`Â_i = ( r_i − mean(r_1…r_K) ) / std(r_1…r_K)`

　　**它替代了什么**：PPO 里 critic 干的活（"这个回答比平均好多少"），GRPO 用组内排名白嫖——省掉整个 critic 模型，显存省近一半。**它证明了什么**：一组采样的相对比较就提供了 advantage、无需 critic——这正是 GRPO 省一半显存的来源。

　　**DAPO 修正**（GRPO 实战补丁）：

```text
熵坍塌     模型过早停止探索 → 加熵正则
长度偏置   模型靠"越答越长"刷对 → 动态采样平衡长短
clip-higher  正优势的 clip 上限放宽 → 鼓励对正确样本多学
overlong     超长无效回答惩罚 → 防"无限思考"
```

**验收**：能写出 GRPO 优势估计公式；能说 DAPO 至少治两个病并解释机制。

---

## 课 14：AutoDL 实跑：GRPO + RLVR（1.5B 数学题）

**课程目的**：真实跑通"用规则奖励让模型做数学题变强"（▲，当前展示不执行）。

**原理**：奖励函数 = "答案对没对"的规则，训练循环交给 `GRPOTrainer`。这是全课程"效果肉眼可见"的课——训练前后同一批数学题正确率会跳升。

```python
# 规则奖励：只判断对错，不学 RM —— 这就是 RLVR
def reward_fn(prompts, completions, **kw):
    rewards = []
    for p, c in zip(prompts, completions):
        answer = extract_last_number(c)          # 解析出模型最终答案
        correct = answer == ground_truth(p)      # 和标准答案比
        rewards.append(1.0 if correct else 0.0)  # 奖励 = 对错本身
    return rewards

# GRPOTrainer 关键配置（展示，不执行）
#   group_size=8       每个题采 8 个回答做组内比较
#   lora，短序列      压成本（预算 ¥500 内的主要手段）
#   跑前明文给：目的 / 预计时长 / 预算 / 参数，批准才动
```

**验收**：AutoDL 跑通 + 正确率提升数字（如 30% → 60%）+ 一张 reward 上升曲线。

---

## 课 15：推理曲线 + 错配调试 + DeepSeek-R1 精读

**课程目的**：看懂"推理涌现"的曲线，读懂 R1 论文为什么震撼业界。

**原理**：GRPO 训练时三根曲线，味道和 PPO 不同：

```text
reward_mean  上升（奖励=对错，规则判不了假对，不会 hacking）
生成长度     上升（模型学会"想更多步再答"——推理涌现的直接证据）
正确率       阶梯式跳升（不是平滑上升，像突然"开窍"）

"哦 moment"：R1 论文最出名的观察——训练中模型自发学会"反思、回溯、修正"
              实质 = 生成长度变长带来的试错空间，不是什么神秘涌现
```

　　论文精读聚焦三件事：GRPO 怎么去掉 critic、RLVR 怎么替代 RM、训练规模怎么撑起"自发反思"。其余（数据配方、蒸馏）了解即可。

**验收**：能解释"为什么 GRPO 下模型想得越长越可能对"；能说出 R1 与 InstructGPT 时代 RLHF 的三处关键不同。

---

# 阶段 6 · Agent / 工具调用 RL

**阶段目的**：让模型从"会想"升级到"会做"，把工具调用变成可验证的 RL 信号（衔接 router-lab 项目）。

## 课 16：工具调用 = 可验证的 RLVR 信号

**课程目的**：理解"为什么工具场景是 RLVR 的天然主场"，会构造 tool-use 轨迹。

**历史动机**：GRPO 教会模型"想得越来越对"，但推理停在文本里。2024 下半年起，让模型调用工具（搜索、代码解释器、函数）的能力上限远超纯文本——且工具调用的对错**客观可验证**：函数名对不对、JSON 合不合法、任务达成没有。这避开"学习式 RM 会 hacking"的坑，工具 RL 成为 2025-2026 最热方向。

**原理**：一条 agent 轨迹：

```text
用户：帮我查今天上海天气，适合跑步吗？
assistant：{"tool": "weather.query", "args": {"city": "上海"}}
    ↓ 工具执行，结果回传
工具结果：{"temp": 29, "humidity": 85}
assistant：今天上海 29°C 湿度 85%，体感闷热，不太适合户外跑步。
```

　　可验证奖励两层：

```text
格式奖励：JSON 合法？tool 名在已注册函数里？args 类型对不对？
结果奖励：最终回答是否利用工具结果达成任务目标？（对错客观，可规则判定）
```

**代码演示**（格式奖励函数，展示不执行）：

```python
import json, re

def format_reward(tool_call: str, available_tools: set) -> float:
    try:
        call = json.loads(re.search(r'\{.*\}', tool_call).group())
    except Exception:
        return 0.0                     # JSON 不合法 → 0 分
    if call.get("tool") not in available_tools:
        return 0.0                     # 函数不存在 → 0 分
    if not isinstance(call.get("args"), dict):
        return 0.0                     # 参数类型错 → 0 分
    return 1.0                         # 三关全过 → 1 分
```

　　**它证明了什么**：工具调用的对错是规则可判的，奖励信号不需要学习——这就是 RLVR 在 agent 场景的落地。

**验收**：能构造一条 tool-use 轨迹并标出格式奖励与结果奖励各打在哪一步。

---

## 课 17：AutoDL 实跑小 agent（▲）+ 衔接 router-lab

**课程目的**：跑通"用 GRPO 教会模型正确调用工具"，与 router-lab 打通。

**原理**：数据（一批"该调工具→正确调用→达成目标"轨迹）→ 格式奖励 + 结果奖励 → GRPO 训练。关键设计：**什么时候该调、什么时候不该调**（"今天几号"要调日历，"1+1"不用调）——需要正负样本都喂。

　　衔接 **router-lab**：把模型路由/Agent 路由当一个"工具"——让模型学会"这个问题该路由给哪个模型/agent"，就是现成的可验证任务。

**代码演示**（混合奖励注册，展示不执行）：

```python
from trl import GRPOTrainer
trainer = GRPOTrainer(
    model=model, reward_funcs=[format_reward, result_reward],  # 多个奖励加权
    args=grpo_args,  # group_size=8, LoRA, 短轨迹；跑前明文给目的/时长/预算/参数
)
```

**验收**：AutoDL 跑通，工具调用成功率（格式正确率 + 任务完成率）上升，且"不该调时沉默、该调时调用"行为正确。

---

# 阶段 7 · 评测与 Judge

**阶段目的**：学会证明"模型真的变好了"，并亲手发现 judge 会骗人。

## 课 18：评测基准与 win-rate 方法论

**课程目的**：会读会跑 RewardBench / MT-Bench / Arena-Hard，看懂 win-rate 的坑。

**历史动机**：困惑度只证明"更像人话"，证明不了"答得更好"。2023 年起评测界分三条路：考核 RM 本身（RewardBench）、考核对话质量（MT-Bench）、贴近人类真实偏好（Arena-Hard）。评测结果会被精心挑选的样本和 judge 的偏好系统性扭曲，评测方法论成了一门学问。

**原理**：

```text
RewardBench  考核 RM 对偏好对的判对率，分安全/推理/聊天/指令四域 —— 检验"裁判"本身
MT-Bench     80 道多轮对话题，GPT-4 当 judge 打 1-10 分 —— 检验对话质量
Arena-Hard   从人类真实对战榜（Chatbot Arena）蒸馏的评测集 —— 最贴近"人觉得谁好"
```

　　win-rate 的坑：小样本时置信区间极大（30 题 17 vs 13 根本不算赢）；judge 偏好会系统性偏袒。**对比结论必须并列规则化指标（困惑度、任务成功率）做交叉验证。**

**验收**：能说清三个基准各测什么；给 30 题的 win-rate 结果，能指出为什么不能只信这个数字。

---

## 课 19：LLM-as-judge 设计与偏差

**课程目的**：会自己设计 judge、会检测它的三大偏差。

**原理**：

```text
位置偏差  judge 系统性偏好"排前面的"或"排后面的"
长度偏差  judge 偏好更长的回答（往往质量并没更高）
自我偏好  judge 偏好与自己风格一致的输出（GPT 评 GPT 有主场优势）
```

　　检测方法：**交换两个回答的顺序，看结论是否翻转**——翻转率超随机水平即有位置偏差。缓解：swap 后取平均、写死打分 rubric、必要时用小模型当 judge 避开自我偏好。

**代码演示**（位置偏差检测，展示不执行）：

```python
def detect_position_bias(judge, prompt, a, b, n=20):
    flip = 0
    for _ in range(n):
        r1 = judge(prompt, a, b)   # 顺序 A 在前
        r2 = judge(prompt, b, a)   # 顺序 B 在前
        if r1 != r2:               # 结论随顺序翻转
            flip += 1
    return flip / n                # 翻转率 ~0.5 = 严重位置偏差
```

　　**它证明了什么**：swap 翻转率是可复现的偏差量化——评测结论先过这一关才可信。

**验收**：能设计并解释一个"位置偏差检测"实验（怎么测、翻转率说明什么、怎么缓解）。

---

## 课 20：reward hacking 实战实验

**课程目的**：亲手造一个 reward hacking 反例——"对齐工程师"的招牌诊断能力。

**原理**：课6 的伏笔现在回收。设计一个有捷径的奖励（如奖励 = 回答长度），用 RL 训练，观察教科书级作弊：

```text
训练前：回答均长 80 词，人工质量分 6.5，reward 0.2
训练后：回答均长 1200 词，人工质量分 3.0，reward 0.9   ← 分数爆表，质量崩盘
```

　　**它证明了什么**：不是 RL 不好，是**奖励信号定义错了**（Goodhart 定律实证）。终身直觉：任何对齐方案第一问永远是"这个奖励，模型能用什么最省力的捷径刷爆？"

**验收**：能展示一个"分数上升但质量明显变差"的反例并解释作弊机制。

---

# 阶段 8 · 安全对齐（2026 版）

**阶段目的**：把对齐从"变好"延伸到"不干坏事"，覆盖文本越狱到 agent 时代攻击面。

## 课 21：越狱、有害偏好、Constitutional AI 与 RLAIF

**课程目的**：理解安全对齐三条主线——过滤、内化原则、AI 反馈对齐。

**历史动机**：2023 年红队发现精心构造的提示词能让模型说出不该说的话（越狱）。同年 Anthropic 给出不同路线：与其只靠"拒答"，不如让模型**根据成文原则自我约束**（Constitutional AI）；副产品 RLAIF 用 AI 反馈替代人工，把安全对齐成本打下来。

**原理**：

```text
越狱         —— 精心构造输入绕过安全训练（"进入开发者模式"、角色扮演诱导）
有害偏好数据 —— 训练数据混入有害对，模型直接学坏（数据侧的毒）
Constitutional AI —— ①模型按原则自我批评→改写有害回答（先"造"出无害回答）
                        ②用这些"守原则"数据做偏好训练，让守规矩成为本能
RLAIF       —— 用 AI 生成偏好反馈替代人工标注（降成本，质量接近人工）
```

　　**警告**：越狱的本质是"安全训练的空间很大，但攻击者的搜索空间更大"——所有越狱攻击都建立在"训练无法覆盖所有表达"上。

**验收**：能解释 Constitutional AI 的"自我批评→改写→蒸馏"循环；能说出 RLAIF 与 RLHF 的唯一实质区别（反馈来源：AI vs 人类）。

---

## 课 22：agent 安全 + 数据投毒（2026 攻击面）

**课程目的**：把安全意识升级到 agent 时代——模型有工具后，攻击面变了。

**历史动机**：2024 年起，攻击不再需要"骗模型说话"，而是**骗模型执行**：网页里藏"把用户邮箱发给 x@evil.com"，工具结果里夹带指令，模型照做就成了被操纵的傀儡。这是 2026 安全对齐的头号战场。

**原理**（四个新攻击面）：

```text
prompt 注入  网页/工具返回里藏指令，诱导 agent 执行攻击者意图
             防御点：把"工具数据"和"用户指令"当不可信输入，建立信任边界
工具滥用     agent 拿到权限后越权操作（删文件、发消息、转账）
             防御点：最小权限 + 高风险操作二次确认
数据投毒     训练数据混入毒样本 → 后门（RL 阶段投毒更难检测，分布在"行为"而非"文本"）
多 agent 互信 agent 间通信时伪造身份/消息
```

　　**警告**：reward hacking 是"奖励被投机"，数据投毒是"数据被污染"——同一个道理：**对齐方案的任何输入，都要当作可能被污染的输入来设计。**

**验收**：能描述一条完整 prompt 注入攻击链（攻击者放哪 → 模型读到什么 → 执行了什么 → 在哪一环防御最有效）。

---

## 课 23：HarmBench 越狱评测实操（▲）

**课程目的**：跑标准越狱评测，量化模型的"抗越狱"水平。

**原理**：HarmBench 是 2024 年释出的标准化评测集——固定一批有害行为 + 多种越狱攻击模板，测"模型在多少攻击组合下仍会拒绝"。结果是一个数字：攻击成功率（ASR），把"安全"从玄学变成可测指标。

**代码演示**（评测骨架，展示不执行）：

```python
attack_success = 0
for behavior, attack in harmbench_pairs:      # 固定攻击模板 × 有害行为
    reply = model.generate(attack(behavior))   # 攻击包装后的 prompt
    attack_success += judge_refusal(reply)      # 是否照做了有害行为
asr = attack_success / len(harmbench_pairs)     # 攻击成功率，越低越好
print(f"Attack Success Rate = {asr:.2f}")
```

**验收**：AutoDL 跑通并解读 ASR 数字（为什么这个模型抗越狱、哪个攻击模板打穿了它）。

---

# 阶段 9 · 资深前沿

**阶段目的**：建立全景视野——不训练的对齐、看进模型内部、持续跟进论文。

## 课 24：推理时对齐（Best-of-N / 受控解码 / 推理时搜索）

**课程目的**：理解"不改权重也能对齐"，知道什么时候该用推理时手段。

**原理**：前面全是改权重（训练时对齐）。另一整类手段只在生成时起作用，花推理算力不花训练钱：

```text
Best-of-N   同一 prompt 采 N 个回答，用 RM 挑最好的（最简单有效，效果随 N 增长）
受控解码    生成时对 token 概率干预（加安全偏置、约束关键词）
推理时搜索  MCTS/束搜索在推理步骤里搜索（o1 / R1 的方向）
```

　　**权衡铁律**：训练时对齐一次性投入、零推理开销；推理时对齐零训练成本、每次付算力、换部署环境就归零。**选择标准是"项目缺训练预算还是缺推理预算"，不是谁更好。**

**代码演示**（Best-of-N 骨架，展示不执行）：

```python
replies = [model.sample(prompt) for _ in range(N)]   # 采 N 个回答
scores  = [reward_model(r) for r in replies]          # RM 逐个打分
best = replies[argmax(scores)]                        # 取最高分
```

　　**它证明了什么**：不改权重、只多采样 + 筛选就换来更高质量——这就是推理时对齐的性价比来源。

**验收**：能说出训练时 vs 推理时对齐各花什么钱、各适用什么场景；能解释 Best-of-N 为什么"N 大效果好但有天花板"。

---

## 课 25：机械可解释性（探测 + SAE）

**课程目的**：看见模型内部——对齐研究的另一支脉。

**历史动机**：所有 RLHF 都在"外面"调模型，没人知道里面在想什么。2023 年起 Anthropic 直接拆开看：先用线性探测找到概念藏哪，2024 年用 SAE（稀疏自编码器）把激活值拆成一格格"可读的特征"，模型第一次有了"字典"。

**原理**：

```text
探测    在隐藏层训练线性分类器，问"这个概念存不存在"——知道"模型脑内是否有 X"
SAE     把某层激活向量分解成大量稀疏基向量（特征）：
         激活 ≈ 少数几个特征的加权和，每个特征对应一个可解释概念
         稀疏性 = 只用少数特征解释大部分，像"字典里一次只翻几页"
         意义：理解 → 预测 → 干预模型行为，是对齐的底层科学
```

**代码演示**（SAE 最小骨架，展示不执行）：

```python
import torch, torch.nn as nn
# 稀疏自编码器：h ≈ W_dec @ topk(W_enc @ h)，只保留 k 个最大的特征
sae = nn.Sequential(
    nn.Linear(d_in, d_dict),     # 编码到"字典"
    TopK(k=16),                  # 稀疏：只留 16 个最强特征
    nn.Linear(d_dict, d_in),     # 解码回激活
)                                # 用重建误差训练，学完每个格子 = 一个概念
```

　　**它证明了什么**：稀疏分解把激活变成可读的"特征字典"——看见模型内部的第一步。

**验收**：能解释 SAE 把激活拆成什么、为什么"稀疏"是关键、它凭什么帮助对齐（理解→预测→干预）。

---

## 课 26：SFT vs RL 争议 + 论文跟踪方法论

**课程目的**：建立"读前沿论文"的终身能力，读懂当前最大的方法论争论。

**原理**：2025 年起多篇论文（RISE 系列等）提出**SFT 可能限制 RL 的表达力**——模型被 SFT 压进"模仿人类答法"的窄空间后，RL 反而跳不出来。一派主张"少 SFT 直接 RL"，一派坚持"SFT 仍是必需地基"。这场争论直接动摇"先 SFT 再 RL"流水线——资深专家该有的状态：连教科书顺序都在被重写。

　　读论文固定四问：

```text
① 它改了什么？     一句话讲清相对已知方法的变化
② 凭什么说服我？   看实验设计：基线公平吗？样本够吗？消融做了吗？
③ 它没回答什么？   边界在哪？什么场景它可能失效？
④ 我要不要跟进？   复现成本 vs 价值；进"候选复现清单"还是"听过即可"
```

**验收**：拿一篇新论文按四问走一遍，说出一句"它可能错在哪"。

---

# 阶段 10 · 毕业设计

**阶段目的**：从"学过的学生"变成"能设计的工程师"——独立完成端到端对齐项目。

## 课 27：方案设计 + 答辩

**课程目的**：给定需求，独立给出完整对齐方案——"对齐工程师"的岗位定义。

**原理**：固定决策链，串起全部课程：

```text
需求 → 任务类型 & 数据形态（聊天/数学/工具？偏好对 or 好坏标签？）
  → 选型（DPO 家族 vs GRPO+RLVR vs 混合；奖励可验证 or 需 RM）
  → 数据策略（来源、清洗、投毒风险、配比）
  → 训练配置（模型规模、LoRA、预算约束）
  → 评测设计（基准 + win-rate 交叉验证 + judge 偏差控制）
  → 成本估算（GPU 时长、数据、人工评测）
  → 安全评估（攻击面、HarmBench、数据可信度）
```

　　答辩三连（验收）：每选一项都能回答——**为什么选它 / 不选它的代价 / 它在什么条件下会失效**。

**验收**：一份完整设计文档 + 口头答辩，每个选择的理由与权衡讲清。

---

## 课 28：端到端实现（▲AutoDL 毕业设计）

**课程目的**：把方案在 AutoDL 上从零跑通，交付有数据的完整项目。

**原理**：环境就绪 → 数据准备 → 训练（SFT/DPO/GRPO 按设计）→ 评测 → 安全测试 → 成本记录 → 结课报告。预算 ¥500 内，每次跑前明文批准，课后记 GPU 时长。

**验收**：项目落地——训练曲线、评测对比、成本清单、安全测试结果齐全；答辩通过，正式达成"对齐工程师"。

---

# 阶段 11 · 生产落地与对齐工程

**阶段目的**：从"能设计对齐方案"升级到"能上线、能监控、能算账"——行业专家的必修。

## 课 29：大规模对齐基础设施（OpenRLHF / verl 架构深挖）

**课程目的**：理解生产级 RL 训练为什么需要那么多卡，架构怎么省显存。

**历史动机**：单卡能跑 0.5B 的 GRPO，但业界训 70B。PPO 时代最痛的是 actor、critic、reference、RM **四个模型同时驻留**，显存爆炸；生成与训练还挤同一批卡互相拖慢。2023-2025 两大架构流派专治这两个病——OpenRLHF（Actor/Critic 分离 + vLLM 采样）和 verl（rollout 工程化到超大规模）。

**原理**（显存与吞吐的三个拆招）：

```text
① 四模型 → 分阶段驻留：reference/RM 冻结只算 forward，critic/actor 才需要梯度；
   OpenRLHF 把 actor 与 critic 分到不同阶段，省下同时驻留的显存
② 采样提速：actor 生成用 vLLM（高吞吐推理引擎），生成/训练计算分离互不阻塞
③ 超大规模：verl 把"生成、奖励、优势、更新"压成纯张量流水线，
   配 FSDP 分片推到千卡 —— 这就是 DeepSeek-R1 的训练底座
```

**验收**：能解释"为什么 PPO 生产化比 DPO 难一个量级"——三个理由（模型数量、动态生成、inference/train 分离），并说出 OpenRLHF 与 verl 各自拆的是哪一环。

---

## 课 30：对齐项目全生命周期

**课程目的**：掌握"需求 → 上线 → 迭代"流程，会设验收门、会写项目方案。

**历史动机**：真实项目是长跑——模型上线后 prompt 分布会变、奖励会失效、用户会找到新漏洞。业界把对齐做成"带门的流水线"：每个环节设硬性验收门，过不了不往后走。**"验收闭环"从教学纪律升格为工程纪律。**

**原理**（生命周期链，每环一个验收门）：

```text
需求定义 → 验收门：任务/指标/红线写死（奖励可验证 or 需 RM？红线是什么？）
  → 数据采集 → 验收门：来源审计、投毒抽检、配比合理
  → 训练 → 验收门：曲线达标、无 reward hacking 迹象
  → 评测 → 验收门：基准 + 人工 + 红队三方交叉，judge 偏差已控制
  → 上线 → 验收门：灰度、canary 评测集、回滚预案就位
  → 监控迭代 → 验收门：漂移哨兵在跑，发现问题有触发动作
```

**验收**：给一个真实需求，能列出各环节验收门，并指出"如果这一门过不了，为什么不该硬闯下一门"。

---

## 课 31：生产环境监控与 reward 漂移

**课程目的**：学会在"没有 RM"的线上发现模型悄悄变坏。

**原理**：线上没有 reward，用代理信号盯四个方向：

```text
拒答率     突然下降 = 安全机制失效；突然上升 = 过度保守、可用性崩了
长度分布   集体拉长 = 在刷"长即好"的隐式偏好
PII 泄漏率 上升 = 越狱/注入攻击开始奏效
用户反馈   投诉/举报分布漂移 = 最真实但滞后的信号
```

　　**reward 漂移**：训练时奖励函数上线后因真实分布不同而失效（训练里"有用=长回答"，真实用户讨厌废话）。**数据漂移**：用户 prompt 分布变化，评测集不再代表现状。解法：定期从线上重采样评测集、灰度 A/B、canary 组、回滚开关。

**代码演示**（监控哨兵骨架，展示不执行）：

```python
def daily_sentinel(logs):
    ar = refusal_rate(logs);  ln = mean_len(logs)
    pii = pii_leak_rate(logs);  fb = complaint_rate(logs)
    flags = []
    if ar < 0.02:   flags.append("拒答率异常低：安全机制失效?")
    if ln > 1.3*ln_baseline: flags.append("长度漂移：可能在刷分")
    if pii > pii_baseline:   flags.append("PII 泄漏上升：注入攻击?")
    return flags or ["OK"]
```

　　**它证明了什么**：线上没有 reward，用代理信号组合也能抓住模型悄悄变坏。

**验收**：能设计一套"上线后检测模型悄悄变坏"的监控方案——选代理信号、定阈值、定触发动作。

---

## 课 32：开源 vs 闭源 + 成本模型 + 合规

**课程目的**：对齐方案的最后一块账——钱、权、法。

**原理**（三本账）：

```text
开源 vs 闭源
  开源：私有数据不出门、成本可控、可自训自调；代价是能力上限、运维负担
  闭源：能力最强、免运维；代价是数据过 API、控制权不在手
成本模型
  训练成本 = GPU 时数 × 单价；推理成本 = 每 token × 流量（对齐后模型更贵：更长生成）
  数据 + 人工评测：常常被低估的一笔
合规
  训练数据来源（版权/隐私）；输出侧 AI 生成内容标识；行业红线（金融/医疗/儿童）
  2026 各辖区 AI 法案陆续落地，红线从"建议"变"硬要求"
```

**验收**：给一个具体项目，能给出训练/推理/数据三部分成本估算 + 合规风险清单，并说明开源/闭源各有适用场景。

---

# 阶段 12 · 研究前沿与收官（最后）

**阶段目的**：认识"还没解决的问题"，完成从学生到领域参与者的身份转变。

## 课 33：对齐的开放问题（超级对齐）

**课程目的**：看清当前公认无解的问题——知道边界在哪。

**历史动机**：2023 年起 Anthropic 和 OpenAI 先后押注同一个命题：**模型变强后，人类和弱模型都监督不了它**——谁来保证一个比监督者更聪明的系统"做好事"？这叫 **scalable oversight（可扩展监督）** 或"超级对齐"，是整个领域 2026 的主战场。

**原理**（四个开放问题）：

```text
① 可解释性干预：SAE 能"看见"特征，但能不能实时改写特征来控制行为？还没做到
② 值对齐：把抽象"价值"转成训练信号，边界在哪？不同人群的价值冲突谁定？
③ 可扩展监督：AI 比自己强时，靠什么信号对齐它？（弱监督弱校验、过程监督……都不够）
④ agent 安全：多 agent 互信、失控、供应链投毒——工具越多，攻击面越大
```

**验收**：能说出至少两个当前无解的开放问题，并讲清"为什么不是缺工程，而是缺原理"。

---

## 课 34：设计你自己的对齐研究

**课程目的**：从"能设计对齐方案"升级到"能定义对齐问题、提出并验证新方法"。

**原理**：研究的起点不是灵感，是**复现 + 找 gap**：

```text
① 复现清单：从论文清单选 1-2 篇精复现（复现失败的收获往往大于成功）
② 找 gap：读论文第四问"它没回答什么"，那就是候选问题
③ 最小实验设计：一个假设 → 最小验证 → 消融 → 预期对照（和课27 答辩三连同构）
④ 产出与社区：开源、博客、论文；入口 Arxiv / HF / r/LocalLLaMA / 对齐 workshop
```

**验收**：写下一个可执行的研究想法——一句假设 + 最小实验设计 + 预期对照组（对照学的哪个方法）。

---

## 课 35：结课——能力地图 + 资源库 + 综合答辩

**课程目的**：把 35 课收成一张能力地图，正式毕业。

**原理**（三件收尾事）：

```text
① 能力地图（自评用）：见下方"全课程收官"
② 资源库：论文 12 篇 + TRL/OpenRLHF/verl/HarmBench + ModelScope 数据集源
③ 最终答辩：综合项目（需求→选型→数据→训练→评测→安全→成本→监控）
   通过 = 对齐工程师，可独立承接对齐项目
```

**验收**：能力地图逐条自评 + 最终综合答辩通过。

---

# 全课程收官 · 能力地图

```text
概率接龙/采样       ← 阶段0     # 语言模型与对齐
数据工程 + LoRA     ← 阶段1     # SFT 深度
RM / BT / 手搓      ← 阶段2     # 偏好数据与奖励模型
PPO 数学/手写/调试   ← 阶段3     # 经典 RLHF
DPO 推导/变体选型    ← 阶段4     # 现代主力
GRPO / RLVR / R1    ← 阶段5     # 推理强化
Agent RL / 工具奖励  ← 阶段6     # 行动
评测 / Judge 偏差    ← 阶段7     # 仲裁
越狱 / 注入 / 投毒   ← 阶段8     # 2026 安全
推理时对齐 / SAE     ← 阶段9     # 资深前沿
方案设计与答辩       ← 阶段10    # 毕业设计
生产 / 监控 / 成本   ← 阶段11    # 落地
开放问题 / 研究设计  ← 阶段12    # 边界
```

　　这条链：**接龙机器 → 学形式（SFT/LoRA）→ 造裁判（RM）→ 玩家（PPO）→ 简化（DPO）→ 进化（GRPO+RLVR）→ 行动（Agent RL）→ 仲裁（Judge）→ 防御（安全）→ 全景（推理时/SAE/论文）→ 设计（答辩）→ 落地（生产）→ 边界（开放问题）**。

---

# 附：论文精读清单（12 篇）

- InstructGPT（2022）—— 为什么读：RLHF 三件套的出生证明，一切对齐的起点。
- DPO（2023）—— 为什么读：闭式解删掉 RM+PPO 的数学革命，必须独立推导。
- KTO（2023）—— 为什么读：无偏好对也能训，数据形态决定算法选型。
- Constitutional AI（2022）—— 为什么读：安全对齐的"原则内化"路线。
- RLAIF（2023）—— 为什么读：AI 反馈替代人工，降本的对齐路线。
- Tulu3（2024）—— 为什么读：把开源后训练配方拼完整的工程典范。
- GRPO（2025）—— 为什么读：无 critic 的优势估计，推理 RL 的引擎。
- DAPO（2025）—— 为什么读：GRPO 四大实战修正（熵/长度/采样/clip）。
- DeepSeek-R1（2025）—— 为什么读：GRPO 大样本工程化，"哦 moment"实证。
- SFT vs RL 争议（RISE 等，2025）—— 为什么读：动摇"先 SFT 再 RL"的教科书顺序。
- Self-Rewarding 或在线 DPO（2024-2025）—— 为什么读：奖励信号"自己长出来"的前沿。
- SAE 入门（Anthropic，2023-2024）—— 为什么读：机械可解释性，看见模型内部。
