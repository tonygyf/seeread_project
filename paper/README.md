# paper/
- `main.tex` + `refs.bib`：IEEEtran（journal）。编译：`latexmk -pdf main.tex`（需要 `IEEEtran.cls/.bst`：Overleaf 自带；本地 `apt install texlive-publishers`）。
- 红色 `[TODO: ...]` 与 `XX.X` 是待填内容，**所有实验数字必须来自你自己复现的结果**。
- `refs.bib` 的条目根据两篇源论文的参考文献表整理，页码/卷期请逐条核对；DSC（`qi2023dsc`）我按 ICCV 2023 写，请核实。
- 换会议模板（CVPR/ACM MM/AAAI）：保留各 section 文本，替换 preamble、作者块与 `\bibliographystyle`。
