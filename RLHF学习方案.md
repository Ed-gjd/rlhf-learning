# RLHF 学习方案 v2：从小白到资深对齐专家

> 依据：学习方法论与协作协议（开课前必读本文件 / 命令全明文 / 真实环境 / 每课验收 / 不重复 / 保持最新）
> 调研：2026-08-14 实测基准 + 2026-08-18 讲授修订（见文末参考来源）
> 定位：从「RLHF 零基础」→「对齐工程师」→「能设计对齐方案」→「能上线运营、能看懂对齐边界的行业/研究级」
> 核心思路：**先深后广**——阶段 0-10 一路钻机制（数学 + 最小实现 + 手搓 + 关键算法代码演示），阶段 11-12 展开广度与边界（生产落地 / 开放问题 / 毕业设计）
> 配套讲义：《RLHF学习课程讲义.md》（35 课完整内容，本文档为体系总纲）

---

## 〇、本机实测结论（2026-08-14 实测，方案以此为基准）

- Python 3.12.3 / torch 2.13.0（CPU-only，无 GPU）/ 内存 7.4G / C 盘剩 55G
- 已装：numpy、modelscope、torch；**未装**：transformers / datasets / peft / trl / accelerate → 需 pip 装（走 aliyun 镜像）
- 网络（沿用国内镜像方案）：pip→aliyun、模型→ModelScope、GitHub→ghfast
- **双轨计算（写死）**：本机 CPU 跑 ≤0.5B 小模型学机制；AutoDL GPU 跑 1.5B 跑真实效果（预算 ¥500 内）
- 2026 行业现状（调研摘要）：
  - 经典 RLHF（SFT→RM→PPO）已非 2026 主力，但**仍是理解一切对齐技术的根基**，必学
  - 2026 模块化 post-training 堆栈：`预训练 → SFT → 偏好优化(DPO 家族) → 推理强化(GRPO+RLVR)`
  - 奖励模型仍必要，但只用于**不可验证信号**（安全、语气、开放偏好）；代码/数学/工具调用用规则做奖励
  - 前沿：GRPO（无 critic 省一半显存）+ DAPO 修正 + RLVR（可验证奖励）；PPO 退为理解根基
  - 标准监控指标：`reward_mean`（上升）/ `kl_div`（<0.1）/ `entropy`（缓降）/ `clipfrac`（>0.3 = lr 过高）

---

## 一、前置地基与入口评估（占位层，不占课）

　　入口要求：需 **Python + 深度学习底子**。逐项自评，✅已会跳过 / 待系统补的先补再进阶段0。

- Python 基础（语法、数据结构、函数式思维）
- PyTorch 基础（张量、自动微分、Dataset / DataLoader）
- 神经网络基础（MLP、反向传播、损失/优化器/过拟合）
- 最小 Transformer 概念（注意力、tokenizer、词嵌入）

---

## 二、对齐简史：从 2017 到 2026（历史主线）

- 2017　思想源头：Christiano《Reward Learning from Human Preferences》——"用人类偏好学奖励函数"
- 2021　实战前奏：WebGPT 用 RM+RL 让 GPT-3 学会用浏览器
- 2022　规范化起点：InstructGPT 定下 **SFT → RM → PPO** 三件套
- 2023　两条岔路：Constitutional AI + RLAIF（用 AI 对齐 AI）；DPO 闭式解（治 PPO 的病）；KTO 无偏好对
- 2024　减负潮：ORPO/SimPO 连 reference 都省；Tulu3 拼齐开源配方
- 2025　推理时代：GRPO（无 critic）、DAPO 修正、RLVR（规则奖励）、DeepSeek-R1 工程化；SFT vs RL 之争
- 2026　模块化堆栈：SFT → DPO 家族 → GRPO+RLVR 成主流；PPO 退为理解根基（课程顺序 PPO 在前）

---

## 三、2026 工具栈（选新不选旧）

- **库**：transformers + peft（LoRA）+ **TRL**（SFTTrainer / DPOTrainer / GRPOTrainer 主学；PPOTrainer 了解原理）
- **实验/数据**：wandb（曲线可视化闭环）、argilla / distilabel（数据标注/合成，选学）
- **推理/评测**：vLLM（AutoDL 跑对齐模型）、HarmBench（越狱评测）、RewardBench / MT-Bench / Arena-Hard
- **框架（了解架构）**：OpenRLHF（Actor/Critic 分离 + vLLM）、verl（超大规模）、LLaMA-Factory（AutoDL UI，选学）
- **模型**：本机 gpt2(124M) / Qwen2.5-0.5B；AutoDL Qwen2.5-1.5B
- **镜像铁律**：pip→aliyun、模型/数据→ModelScope、GitHub→ghfast；**模型一律 ModelScope 下载到本地路径加载**，别走 HF hub

---

## 四、学习原则（写死，每课遵守）

1. **先深后广**：阶段 0-10 钻机制（公式推导 + 最小实现 + 手搓），阶段 11-12 展开广度；每课内部也"先机制后应用"
2. **命令全明文（最高优先）**：每个命令先在对话完整写出（含参数）→ 解释参数 → 我执行 → 贴完整输出 → 解释；纯讲授模式则"代码全明文展示 + 预期输出解释"
3. **命令结果必贴**：每课所有运行输出（loss / acc / 奖励曲线 / 生成文本）全部展示不省略
4. **验收带数据（硬指标）**：每课有可测通过标准，过不了不进入下一课（见 §八）
5. **反馈闭环（固定三件事）**：每课结尾 = ①验收数据贴出 → ②提问/挑刺 → ③点评反馈
6. **最小闭环优先**：先用最小模型跑通全流程（哪怕效果差），再放大；先手搓理解再上库
7. **双轨计算**：本机 CPU 学机制 + AutoDL GPU 看真实效果
8. **费用护栏（写死）**：AutoDL 每次跑前**明文给目的/预计时长/预算/参数**，批准才动；课后记 GPU 时长（上限 ¥500）
9. **卡点优先**：装不上 / 跑不通 / 显存不够，先解决再学内容，每个卡点写进当课"坑"
10. **vibe 循环**：直接讲 → 做 → 执行后解释 → 考判断（"这步能省吗？为什么不用 X？"）
11. **固定对比尾部**：多算法/多库对比用代码块逐条列，不用表格
12. **允许打断深挖**：你说"没懂 / 展开 / 结合实例 / 5 条讲清"，按指示重讲
13. **模型/数据一律 ModelScope 本地路径**：禁止直接 `from_pretrained("Qwen/...")` 走 HF hub
14. **推导层级按课标注**：跟读推导 / 独立推导；里程碑 M3/M4 要求独立推导 PPO 目标与 DPO 闭式解

---

## 五、环境就绪（开始前一次性做掉）

```bash
# ① 独立 venv（别污染系统 python）
python3 -m venv ~/rlhf-venv && source ~/rlhf-venv/bin/activate

# ② 装库（aliyun 镜像）
pip install -i https://mirrors.aliyun.com/pypi/simple/ \
    transformers datasets peft trl accelerate wandb

# ③ 验证
python -c "import transformers, peft, trl, datasets, accelerate; \
print('OK', transformers.__version__, trl.__version__)"

# ④ 模型与数据走 ModelScope 本地路径（一次做掉，之后不再卡）
modelscope download --model Qwen/Qwen2.5-0.5B
modelscope download --dataset iic/ultrachat                  # 指令数据（SFT）
modelscope download --dataset iic/ultrafeedback_binarized   # 偏好数据（RM/DPO）
```

> 卡点预警：本机 torch 2.13 是系统级的，venv 里 pip 会再装一份 torch（aliyun CPU 版 ~200MB）；嫌重可用 `--system-site-packages` 复用系统 torch。模型 ~1GB/个从 ModelScope 拉；数据源若有变动，以 ModelScope 当前可检索的国内直连版本为准。

---

## 六、阶段拆解（13 阶段 35 课，先深后广）

- **阶段0 语言模型与对齐（课1-2）**：LM=概率分布+采样+困惑度 ★；2026 模块化对齐全景+历史主线
- **阶段1 SFT 深度（课3-4）**：指令数据格式 + LoRA 低秩原理 ★；SFT 实跑（本机 0.5B + LoRA，前后对比）
- **阶段2 偏好数据与奖励模型（课5-6）**：偏好协议 + Bradley-Terry + 手搓最小 RM ★；RM 评测 + reward hacking 预警
- **阶段3 经典 RLHF：PPO（课7-9）**：策略梯度→PPO 数学（推导层级标定）★；手写最小 PPO（验收=跑通+讲清曲线）★；PPO 调试 + 错配→修复实验
- **阶段4 现代主力：DPO 家族（课10-12）**：DPO 数学（独立推导验收）★；DPO 实跑对比（win-rate 补规则化指标）；KTO/ORPO/SimPO + 在线 DPO/self-play
- **阶段5 推理强化：GRPO+RLVR（课13-15）**：GRPO 原理 + DAPO 修正 + RLVR（优势计算演示）★；AutoDL 实跑 1.5B 数学题 ▲；推理曲线 + 错配调试 + DeepSeek-R1 精读
- **阶段6 Agent/工具调用 RL（课16-17）**：工具调用 = 可验证 RLVR 信号 ★；AutoDL 实跑小 agent ▲（衔接 router-lab）
- **阶段7 评测与 Judge（课18-20）**：RewardBench/MT-Bench/Arena-Hard + win-rate 方法论；LLM-as-judge 设计与偏差检测 ★；reward hacking 实战实验
- **阶段8 安全对齐 2026（课21-23）**：越狱/有害偏好/Constitutional AI/RLAIF；agent 安全 + 数据投毒；HarmBench 越狱评测 ▲
- **阶段9 资深前沿（课24-26）**：推理时对齐（Best-of-N 等）★；机械可解释性（探测 + SAE 最小实现）★；SFT vs RL 争议 + 论文跟踪方法论
- **阶段10 毕业设计（课27-28）**：方案设计 + 答辩（需求→选型→数据→训练→评测→成本→安全）；AutoDL 端到端实现 ▲
- **阶段11 生产落地（课29-32）**：大规模基础设施（OpenRLHF/verl 架构）；项目全生命周期；生产监控与 reward 漂移；开源 vs 闭源 + 成本 + 合规
- **阶段12 研究前沿与收官（课33-35）**：对齐开放问题（可扩展监督）；设计自己的研究；结课能力地图 + 综合答辩

> ★ = 关键算法代码演示课（≥10 处贯穿）；▲ = AutoDL 实操课（≥5 处）。完整每课内容见《RLHF学习课程讲义.md》。

---

## 七、每课固定结构模板 + 课1 示范

### 7.1 每课固定结构

```
## 第N课：主题
- 课程目的（一句话，学完能干嘛）
- 历史动机（这算法从哪来、治前人的什么病）
- 原理（类比引入 + 公式，标注推导层级：跟读/独立推导）
- 关键算法/代码演示（可运行代码 + 预期输出 + "它证明了什么"）
- 坑（本课卡点 + 解法）
- 验收（可测硬指标，过不了不进下一课）
- 反馈（你提问 → 我点评）
```

### 7.2 课1 示范（模型加载走 ModelScope 本地路径）

**课程目的**：亲手让真实模型续写文本，理解"模型输出不是文本而是概率分布"。

**历史动机**：2017 Transformer 的基因就是"预测下一个词"；所有对齐技术驯服的都是这台"接龙机器"的概率分布。

**代码演示**：

```bash
modelscope download --model Qwen/Qwen2.5-0.5B   # 下载到 ~/.cache/modelscope，记录本地路径
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
# 注意：用 ModelScope 下载后的本地路径，不要直接 from_pretrained("Qwen/Qwen2.5-0.5B")
LOCAL = "/path/to/Qwen2.5-0.5B"                # 替换为实际本地路径
tok = AutoTokenizer.from_pretrained(LOCAL)
lm  = AutoModelForCausalLM.from_pretrained(LOCAL)
ids = tok.encode("The capital of France is", return_tensors="pt")
logits = lm(ids).logits[0, -1]
print("vocab_size =", lm.config.vocab_size)     # 151936
print("logits.shape =", tuple(logits.shape))    # (151936,) 每个候选字一个数
print("top5_words =", [tok.decode([i]) for i in logits.topk(5).indices.tolist()])
```

**它证明了什么**：模型只在"法国的首都是"后面列概率表——top1 是 Paris。整个 RLHF 建立在这之上。

**验收**：能说出 `151936` 是什么；能解释"输出是概率分布"；能说清贪心 vs 采样与 PPO"探索 vs 利用"的关系。

---

## 八、里程碑与验收总览（M0-M12）

- **M0**（阶段0）：环境通 + 理解 LM=概率分布 + 懂对齐全景 —— 三条命令跑通 + 能答"对齐在解决什么问题"
- **M1**（阶段1）：亲手微调出输出风格变化的模型 —— loss 下降 + 前后生成对比（肉眼可判）
- **M2**（阶段2）：手搓出区分好/坏回答的 RM —— held-out pairwise acc ≥ 阈值 + 奖励分布图
- **M3**（阶段3）：跑通经典 RLHF 全链路，**独立推导一遍 PPO 目标**，能解释 kl/熵/clipfrac 曲线
- **M4**（阶段4）：DPO 会用会选，**独立推导一遍 DPO 闭式解**，与 SFT base 的 win-rate 对比报告
- **M5**（阶段5）：GRPO+RLVR 让模型数学题变强 —— AutoDL 跑通 + 正确率提升数字 + 推理曲线
- **M6**（阶段6）：模型学会正确调用工具 —— 工具调用成功率上升，该调才调
- **M7**（阶段7）：能评测、能发现 judge 偏差、能造 reward hacking 反例
- **M8**（阶段8）：懂 2026 安全（越狱/注入/投毒），HarmBench 跑出 ASR 并解读
- **M9**（阶段9）：能用四问读新论文；能解释 SAE 原理与 Best-of-N 权衡
- **M10**（阶段10）：毕业答辩——完整选型决策 + 成本估算 + 端到端实现
- **M11**（阶段11）：能给真实项目做生命周期验收门、监控方案、成本与合规清单
- **M12**（阶段12）：最终综合答辩（需求→选型→数据→训练→评测→安全→成本→监控），达成"对齐工程师"

---

## 九、论文精读清单（12 篇）

InstructGPT / DPO / KTO / Constitutional AI / RLAIF / Tulu3 / GRPO / DAPO / DeepSeek-R1 / SFT vs RL 争议（RISE 等）/ Self-Rewarding 或在线 DPO / SAE 入门（Anthropic）。每阶段读 1-2 篇，读透不贪多；各篇"为什么读它"见讲义附录。

---

## 十、费用护栏（AutoDL，上限 ¥500）

- 本机 CPU：¥0（小模型学机制全部免费）
- AutoDL GPU（RTX 3090 档，约 ¥2/时，按需开机）：
  - 课14 GRPO 实跑 3-5 小时（group_size=8、短序列、LoRA 压成本）
  - 课15 调试 2-3 小时；课17 Agent RL 3-5 小时
  - 课20 hacking 实验 1-2 小时（可选）；课23 HarmBench 1-2 小时
  - 课28 毕业设计 5-10 小时 + 迭代缓冲
  - 合计约 ¥80-200，留 ¥500 上限缓冲
- **每次跑前明文给：目的 / 预计时长 / 预算 / 参数，你批准才动**；课后记 GPU 时长进进度存档
- 不买域名 / 不订阅付费 API：评测用小模型本地 + 公开榜

---

## 十一、待定项（开始前确认）

1. **现在就跑环境就绪清单？**（约 10-15 分钟：venv + 装库 + ModelScope 模型/数据下载）
2. **起点**：阶段0-1 可快进（课2-4 已在讲义中补齐，直接进入实操或跳过基础均可）
3. **课程仓库**：要不要建 `cc/rlhf-course/`（lessonNN_主题/ + materials/ + README，同 threejs/embodied 惯例）？是否推送 GitHub `Ed-gjd/rlhf-course`？
4. **AutoDL 本轮启用**：阶段5 起需要 GPU，先确认账号/凭证可用 + 镜像 TRL 版本支持 GRPOTrainer

---

## 十二、参考来源（2026 调研）

- [The Death of RLHF: A Practitioner's Guide to the New Post-Training Stack（2026 模块化堆栈）](https://pub.towardsai.net/the-death-of-rlhf-a-practitioners-guide-to-the-new-post-training-stack-84b2ff6d4e74)
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [RLHF for Chat Agents in 2026: PPO Is Out, GRPO Is In](https://callsphere.ai/blog/vw8g-rlhf-chat-agents-reward-model-grpo-2026)
- [RLHF Training Infrastructure on GPU Cloud: verl, OpenRLHF, TRL](https://www.spheron.network/blog/rlhf-training-infrastructure-verl-openrlhf-trl-gpu-cloud/)
- [LLM Fine-Tuning Guide 2026: LoRA, QLoRA, DPO, GRPO, RLHF（算法使用率数据）](https://futureagi.com/blog/llm-fine-tuning-guide-2025/)
- [RL Posttraining for Tool-Using Agents: GRPO, Async RL, and Reward Design in 2026](https://zylos.ai/en/research/2026-04-10-rl-posttraining-tool-using-agents-grpo-async-rl/)
- [DRA-GRPO（ACL 2026 Findings，GRPO 推理方向研究）](https://aclanthology.org/2026.findings-acl.685/)
- [Stanford CS336: Post-Training - RLVR（课程讲义）](https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-dIFAi8)
- 前沿补充：DeepSeek-R1、RISE（SFT vs RL 争议）、Anthropic SAE 系列（2023-2024）、HarmBench（2024）、RewardBench（2023）
