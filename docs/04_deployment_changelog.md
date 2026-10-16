# SeeRead Docker 部署变更记录

> 记录日期：2026-10-14。本文记录本次"彻底清理 + 按官方文档重部署 + 资源防护 + 代码审查修正"的全部变更，供后续 review。所有变更仅涉及**部署层**（`docker-compose.yml`、`.env`、`.env.example`），未修改模型、训练代码与数据协议。

## 1. 环境残留清理

| 对象 | 处置 | 说明 |
|---|---|---|
| `python:3.12-slim` 镜像 | 已删除 | 此前 ad-hoc 容器下载 TrOCR 时拉取，官方流程不需要 |
| `~/.cache/huggingface/.../trocr-base-printed` | 已删除 | root 属主（容器写入），经 ubuntu:24.04 容器以 root 删除；官方流程权重缓存放在 compose 命名卷 `hf-cache` |
| `~/.cache/torch/.../resnet50-11ad3fa6.pth` | 已删除 | 同上，官方流程放在命名卷 `torch-cache` |
| 项目工作区 | 无残留 | `git status` 干净，无 `__pycache__`/`.pyc`/`.env`/`egg-info` |
| 其他项目 Docker 资源（darktext、ollama、vllm、cgcwra-env、ds4 容器等） | **未触碰** | 不属于本项目 |

清理后验证：`docker images` 无 python:3.12-slim / seeread；`docker compose ls -a` 无 seeread 项目残留。

## 2. 按官方文档重部署

1. `.env`：由 `.env.example` 复制，按文档要求调整两项——`PUID=1001`/`PGID=1001`（匹配 host 用户 gyf 的 uid/gid，保证 `./data`、`./results` 挂载属主正确）；`CUDA_VISIBLE_DEVICES=0`（保留 GPU 1 空闲）。其余保持模板默认。
2. `docker compose build`：基础镜像 `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`（driver 595.91.07 / CUDA 13.2 向下兼容 12.4 runtime），产物 `seeread:latest`（10.1GB），构建成功无报错。
3. 模型权重按官方容器化流程重下至命名卷（TrOCR → `hf-cache`，ResNet50 → `torch-cache`），见本文第 5 节验证结果。

## 3. 资源防护（docker-compose.yml 新增，均对 `compose run` 生效）

| 风险 | 配置 | 值 | 依据 |
|---|---|---|---|
| 僵尸/孤儿进程 | `init: true` | tini 作 PID 1 | 自动回收 DataLoader worker |
| 内存泄漏拖垮主机 | `mem_limit` / `memswap_limit` | `${MEM_LIMIT:-128g}`，二者相等 | ≈ 主机 251G 的 50%，含 16g shm；禁超限 swap → 超限 OOM kill 容器而非主机 swap 抖动 |
| fork 炸弹 | `pids_limit` | 512 | tasks（进程+线程）估算 ≈100-130（4 workers × 8 OMP + 主进程），余量充足 |
| CPU 过载 | `cpus` | `${CPUS:-64}` | 112 核保留一半给主机 |
| 线程爆炸 | `OMP_NUM_THREADS`/`MKL_NUM_THREADS` | 8（`.env` 注入） | 否则 torch 按 112 核 × 每 worker 开线程 |
| core dump 撑爆磁盘 | `ulimits.core` | 0/0 | — |
| 句柄耗尽 | `ulimits.nofile` | 65536 | — |
| 日志无限增长 | `logging` | json-file 50m × 3 | — |

`pids_limit`/`ulimits`/`logging` 为策略类配置，保留硬编码；`mem_limit`/`cpus` 为主机相关参数，按项目既有插值模式（参照 `${PUID:-1000}`）参数化。

## 4. 代码审查记录（TRAE-code-review，双代理交叉验证）

| # | 级别 | 问题 | 处置 |
|---|---|---|---|
| 1 | minor/一致性 | `OMP_NUM_THREADS`/`MKL_NUM_THREADS` 放在 compose `environment`，偏离项目约定（运行时注入变量统一放 `.env` 经 env_file，见 `.env.example` 头部说明；同类变量 `CUDA_VISIBLE_DEVICES`、`TOKENIZERS_PARALLELISM` 均在 `.env`） | 已移入 `.env` 运行时段；`.env.example` 模板同步 |
| 2 | minor/对齐 | `mem_limit: 128g`/`cpus: 64` 硬编码且注释写死 "251G RAM"，未走项目既有 `${VAR:-default}` 插值模式（如 `${PUID:-1000}`） | 参数化为 `${MEM_LIMIT:-128g}`/`${CPUS:-64}`，默认行为不变；注释去机器特定值 |
| — | 误报剔除 | `pids_limit: 512` 误杀风险（任务数估算 ≈100-130，余量足）；128g 内存上限误 OOM 风险（模型小、batch 8，host 侧远低于上限）；与 `deploy.resources.devices` 混用兼容性（`docker compose config` 实测通过） | 不修改 |

## 5. 验证结果

- `docker compose config`：通过；解析值 `mem_limit=memswap_limit=137438953472`（128GiB）、`cpus=64`、`pids_limit=512`、`init=true`，`OMP/MKL_NUM_THREADS=8` 经 env_file 正确注入。
- `docker compose build`：成功，`seeread:latest` 生成，无报错。
- 权重下载与冒烟测试：见下节（执行后回填）。
