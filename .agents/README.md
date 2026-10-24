# 本项目 skills

安装位置为 `.agents/skills/<name>/SKILL.md`，在此项目下一轮会话中发现。使用 `$diagnosing-bugs`、`$avoid-overkill` 等名称；`$grill-me` 是现有 `grilling` 的兼容入口。若当前会话列表未刷新，可直接读取对应 SKILL.md。

|用途|skill|固定来源|
|---|---|---|
|故障复现、定位、回归|diagnosing-bugs|[mattpocock/skills](https://github.com/mattpocock/skills/tree/5beb35b040e3de97e54fa9e386d7e42d15c295c2/skills/engineering/diagnosing-bugs)|
|模块设计与删除重复逻辑|codebase-design|同上游 commit 的 `skills/engineering/codebase-design`|
|代码与需求分别审查|code-review|同上游 commit 的 `skills/engineering/code-review`|
|显式请求的架构审查|improve-codebase-architecture|同上游 commit 的 `skills/engineering/improve-codebase-architecture`|
|控制代码、测试、文件和防御性逻辑膨胀|avoid-overkill|[6Kmfi6HP/avoid-overkill](https://github.com/6Kmfi6HP/avoid-overkill/tree/3e6c6da904f857c1eae83ee580131ca4cbd58df7/skills/avoid-overkill)|
|显式请求的论文表达检查|defensive-writing-checker|[justairr/defensive-writing-checker](https://github.com/justairr/defensive-writing-checker/tree/b15b79b42be91278118e3c61d3efcbc4e6d6b51f/skills/defensive-writing-checker)|

新增六项通过 Codex skill-creator 的 frontmatter 校验。保留各 skill 的参考文件和上游 agents/openai.yaml；没有另行安装整套 Matt skills。

Codex 适配：架构 skill 移除不支持的 `disable-model-invocation` 字段，显式调用要求保留在 description；不存在的 Skill 工具改为读取本地相对链接。未安装的 domain-modeling 不再是硬依赖。code-review 支持用当前请求/本地设计作规范，审查未提交工作时使用 `git diff HEAD`，不要求先运行不存在的 setup 命令。原有 grill-me/grilling 不覆盖。

工程约束用 avoid-overkill；论文写作约束不得用于隐去失败结果、测试集边界或实际实验差距。日常实验记录如实保留。
