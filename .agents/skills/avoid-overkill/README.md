# Avoid Overkill

[English](#english) | [中文](#中文)

> **为什么有这个 Skill**：本 Skill 用于减少 **GPT-5.6 模型（及同类 Coding Agent）的防御性编程问题**。防御性编程指模型为防御想象出来的失败而添加无依据的校验、回退、重试、兼容路径、抽象和范围蔓延，以及在论文写作中产出自我削弱、防御性的表述。当模型可能过度防御、过度设计时，使用本 Skill 让改动与行文始终与结果和证据成比例。
>
> **Why this skill exists**: This skill exists to reduce **defensive programming in GPT-5.6 (and similar coding agents)** — the tendency to add speculative checks, silent fallbacks, retries, compatibility paths, abstractions, and scope creep to defend against imagined failures, plus defensive, self-weakening prose in academic writing. Use it whenever a model is likely to over-engineer or over-defend, to keep changes and prose proportional to the outcome and the evidence.

## English

A compact Codex skill that keeps engineering lean and academic writing centered on the strongest supported contribution.

### What it guides

- Solve the stated problem with the smallest general change; reuse existing implementations before rebuilding.
- Avoid speculative checks, silent fallbacks, test-specific patches, unnecessary abstractions, and scope creep.
- Treat normal GPU and numerical variation with tolerances and claim-relevant evidence.
- Write papers as claim-centered releases; give every experiment an argumentative role and state decisive advantages in prose.
- Remove self-weakening language, imagined reviewer objections, literal translations, and AI-styled prose.

### Structure

`SKILL.md` holds the core rules and routes each failure mode to one short file under `cases/`. Read only the case matching the active decision: ML nondeterminism, test patching, test overengineering, reuse-before-rebuild, scope creep, defensive paper writing, and academic terminology.

### Use

Place this directory in a Codex skill search path, or point Codex to its `SKILL.md`. Invoke it by name:

```text
Use $avoid-overkill to review this code or paper and keep it proportional, evidence-grounded, and direct.
```

### Usage tutorials (English)

**Tutorial 1 — Trim defensive checks in code**

Prompt:

```text
Use $avoid-overkill. Review this error handler: it retries 3 times, falls back to a cached value, logs 4 warning levels, and wraps every call site in a guard. Keep only what the contract requires.
```

What the skill does: names the observed failure (none reproduced), removes unrequested retries/fallbacks/guards, keeps the single necessary boundary, and stops when user-visible behavior is correct.

**Tutorial 2 — Stop test patching**

Prompt:

```text
Use $avoid-overkill. My test only passes after I added a mock for an internal helper. Is that right?
```

What the skill does: flags the test-specific mock as implementation-detail coverage, points the test at public behavior, and only changes a test after showing its contract is wrong.

**Tutorial 3 — Reuse before rebuild**

Prompt:

```text
Use $avoid-overkill. I wrote a custom retry-with-backoff class, but the project already depends on a library that does this. What should change?
```

What the skill does: prefers the repository/dependency/standard-library implementation, removes the handwritten duplicate, custom-builds only the unsupported part.

**Tutorial 4 — Claim-centered paper prose**

Prompt:

```text
Use $avoid-overkill. Review my paper's abstract and conclusion: remove self-weakening phrases like "only", "limited", "unfortunately", and make the contribution explicit without overclaiming.
```

What the skill does: opens with problem, gap, idea, hardest result; deletes evaluative negatives and imagined reviewer objections; states the supported advantage in prose; qualifies the exact boundary once.

## 中文

这是一个供 Codex 使用的短 Skill，约束两类倾向：工程上无依据地增加机制，论文中把最强贡献写成工作汇报或自我辩护。

### 核心约束

- **工程克制**：围绕明确问题做最小且可泛化的改动，不添加推测性校验、静默降级、测试补丁、无谓抽象和相邻改动。
- **证据相关**：用容差和会影响结论的证据处理 GPU 或数值波动，不追求无意义的逐字节一致。
- **优先复用**：先查仓库、已有依赖、标准库和官方实现，再决定是否自建。
- **发布会原则**：论文围绕最强且有证据支持的优势组织，不按尝试和失败的时间线写成工作汇报。
- **实验有职责**：每个实验都应支撑主效应、机制、边界、稳健性或必要权衡；没有论证作用就删。
- **表达主动**：正文明确说出关键优势，删除自我削弱、预写审稿意见、生硬直译和 AI 腔。

### 结构

`SKILL.md` 仅保留核心规则，并将具体场景分散到 `cases/`。使用时只读相关案例：机器学习非确定性、测试补丁、测试过度工程、代码复用、范围蔓延、防御性论文写作、专业术语。

### 安装与使用

**1. 本地安装**

克隆仓库并将其添加至 Codex 或 Coding Agent 的 Skill 搜索路径：

```bash
git clone https://github.com/6Kmfi6HP/avoid-overkill
```

**2. 远程同步**

直接在对话中让 Coding Agent 安装该 Skill：

```
请安装 https://github.com/6Kmfi6HP/avoid-overkill 中的 Skill。
```

### 使用教程（中文）

**教程 1 —— 去除代码里的防御性校验**

提问：

```text
使用 $avoid-overkill。审查这段错误处理代码：它重试 3 次、回退到缓存值、打了 4 层警告日志、每个调用点都套了防御判断。只保留契约真正要求的部分。
```

效果：Skill 会点名实际出现的失败（没有就删），移除未要求的重试/回退/防御，只保留必要的边界，并在用户可见行为正确后停止。

**教程 2 —— 停止测试打补丁**

提问：

```text
使用 $avoid-overkill。我的测试只有给内部 helper 加了一个 mock 才能通过，这样对吗？
```

效果：Skill 会指出这是针对实现细节的补丁式覆盖，把测试引导到公共行为上；只有证明测试契约本身错误时才改测试，且测试与实现同步更新。

**教程 3 —— 先复用再自建**

提问：

```text
使用 $avoid-overkill。我手写了一个带退避的重试类，但项目已有依赖库自带这个功能。应该怎么改？
```

效果：Skill 优先使用仓库/已有依赖/标准库/官方实现，删除手写重复代码，只自建真正缺失的部分。

**教程 4 —— 以最强主张为中心写论文**

提问：

```text
使用 $avoid-overkill。审阅我的摘要和结论：删掉 "only"、"limited"、"unfortunately" 这类自我削弱表达，在不夸大的前提下把贡献写明确。
```

效果：Skill 以问题—缺口—想法—最难结果开篇，删除评价性否定和预写审稿意见，在正文中写明有证据支持的优势，边界只限定一次。

### 防御性编程自查清单（GPT-5.6 场景）

当模型出现以下任何一条，都应提醒它使用本 Skill：

- 为防御"可能出错"而添加从未被触发的 `try/except`、重试、回退、默认值。
- 为了"更健壮"而给每个函数加参数校验、类型守卫、兼容分支。
- 为"未来扩展"而提前抽象、引入接口、拆模块。
- 测试里加针对实现的 mock / 特判来"让测试通过"。
- 论文里写"虽然我们的方法只是初步的""仍落后于 SOTA"之类的自我削弱。
