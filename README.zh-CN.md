# BioScope Studio 原型说明（中文）

BioScope Studio 是一个面向真实生物识别场景的工程化原型系统。它把 BioCLIP 检索先验、层级约束、多模态推理、干扰分析和人工标注回写整合成可控闭环。

当前版本重点解决三个痛点：

1. 大场景小目标导致的误检与漏检
2. 仅靠大模型自由推理带来的分类漂移
3. 现场业务中“可解释、可修正、可持续迭代”不足

## 原型的核心价值

- **检索增强而非纯猜测**：先由 BioCLIP 提供可追溯先验，再交给 Qwen 生成最终报告
- **分类可控**：最终输出受 BioCLIP 层级约束，降低离谱跨科/跨目漂移
- **复杂场景可用**：支持小目标定位、裁切增强、干扰因素分析
- **可持续进化**：支持人工标注回写向量库，形成数据闭环
- **可解释可审计**：中间证据、层级约束、干扰风险均可视化

## 最新能力总览（截至当前代码）

1. **BioCLIP + 本地 FAISS 检索**
2. **TreeOfLife 官方式先验**
   - 优先级：ToL 分类器 -> ToL 物种列表 -> 本地 metadata 回退
3. **完整层级参考展示**
   - 界 / 门 / 纲 / 目 / 科 / 属 / 种（含常见名）
4. **最终分类层级约束（关键）**
   - 默认阈值 0.6
   - 约束退化链：科 -> 目 -> 纲 -> 门 -> 界
   - 若界层级置信度不足则本轮不施加层级约束
5. **独立干扰分析 Agent（最终分类前）**
   - 路由规则：无可用物种候选抽象框或目标框过多时走全图，否则逐框
   - 覆盖干扰因素：少见姿态、遮挡、色度偏差、低分辨率、失焦模糊、运动模糊、曝光异常、目标过小、背景干扰、目标截断、层级约束冲突
6. **大场景小目标优化链路**
   - Qwen 两阶段定位（推理 -> JSON）
   - 可选 YOLO 辅助候选框
   - 融合候选框后裁切检索
7. **阶段一致性增强**
   - 阶段一仅作为“定位假设”，不允许直接给最终分类结论
   - 干扰分析接入 BioCLIP 先验和层级约束，冲突以风险形式记录，不再直接给相反最终结论
8. **物种中文化与别名归一化**
   - 物种搜索支持：学名 + 常见名 + 别名字典（示例：华南兔 -> Lepus sinensis）
   - 分析报告中的别名/俗名可归一化为标准学名
   - 标注入库时自动归一化为标准学名
9. **中英文双语 UI（默认中文）**
10. **长时推理稳健性**
    - 可配置超时、thinking budget、降级重试

## 系统工作流

1. 上传图像
2. （可选）定位与裁切预处理
3. BioCLIP 检索 + ToL 先验生成
4. 计算层级约束
5. 独立干扰分析（带层级约束上下文）
6. 生成最终受约束分析报告
7. 人工确认/修正并回写向量库

## 目录结构

- `app.py`：主应用编排与 UI
- `small_target_optimizer.py`：定位、融合、裁切、干扰分析 Agent
- `bioclip_model.py`：BioCLIP 与 ToL 相关能力（先验、层级约束、层级补全）
- `vector_store.py`：本地 FAISS 封装
- `build_index.py`：批量构建检索索引
- `export_tol_species_list.py`：导出 ToL 名录（CSV + TXT）
- `prepare_bioclip_local.py`：本地模型缓存准备
- `compare_small_target.py`：优化前后对比工具
- `run_demo.sh`：启动脚本（Conda `torch1`）
- `data/`：ToL 名录、别名字典、向量索引数据

## 环境变量示例

在项目根目录创建 `.env`：

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
DASHSCOPE_MODEL=qwen3.5-plus
DASHSCOPE_TIMEOUT_SECONDS=1800
DASHSCOPE_ENABLE_THINKING=1
DASHSCOPE_THINKING_BUDGET=8192

APP_DEFAULT_LANGUAGE=zh

SMALL_TARGET_OPTIMIZATION=1
SMALL_TARGET_USE_QWEN=1
SMALL_TARGET_USE_YOLO=1
SMALL_TARGET_MAX_CROPS=4
YOLO_ASSIST_MODEL_PATH=./models/ultralytics/yolov12/best_yolo12_s_动物_1024_randcopybg.pt

BIOCLIP_MODEL_ID=hf-hub:imageomics/bioclip
BIOCLIP_TOL_MODEL_ID=hf-hub:imageomics/bioclip
BIOCLIP_USE_TOL_CLASSIFIER=1
BIOCLIP_AUTO_EXPORT_TOL_SPECIES=1
BIOCLIP_SPECIES_LIST_PATH=./data/bioclip_tol_species.txt
BIOCLIP_SPECIES_CSV_PATH=./data/bioclip_tol_taxa.csv
BIOCLIP_SPECIES_ALIAS_PATH=./data/species_aliases.json
BIOCLIP_SPECIES_LIST_MAX_LABELS=0
BIOCLIP_TAXONOMY_CONSTRAINT_THRESHOLD=0.6

INTERFERENCE_BOX_LIMIT=10
INTERFERENCE_MAX_TARGETS=10

HF_HOME=/home/buluwasior/Works/bioscope_studio/models/hf_cache
BIOCLIP_OFFLINE=0
```

## 快速启动

### 1) 安装依赖

```bash
cd /home/buluwasior/Works/bioscope_studio
~/anaconda3/bin/conda run -n torch1 python -m pip install -r requirements.txt
```

### 2) 准备 BioCLIP 本地缓存（推荐）

```bash
cd /home/buluwasior/Works/bioscope_studio
~/anaconda3/bin/conda run -n torch1 python prepare_bioclip_local.py
```

可选离线模式：

```bash
export BIOCLIP_OFFLINE=1
```

### 3) 导出 ToL 名录（推荐）

```bash
cd /home/buluwasior/Works/bioscope_studio
~/anaconda3/bin/conda run --no-capture-output -n torch1 \
  python export_tol_species_list.py \
  --species-txt ./data/bioclip_tol_species.txt \
  --taxa-csv ./data/bioclip_tol_taxa.csv
```

### 4) 构建检索库（按需）

```bash
~/anaconda3/bin/conda run -n torch1 python build_index.py --sample-dir ./sample_images
```

### 5) 运行原型

```bash
cd /home/buluwasior/Works/bioscope_studio
./run_demo.sh
```

访问：`http://<server-ip>:8501`

## 别名字典示例

`data/species_aliases.json`

```json
{
  "Lepus sinensis": ["华南兔", "中国野兔", "Chinese hare", "South China hare"]
}
```

## 应用优势与潜力（面向用户沟通）

- **野外调查与生态监测**：在复杂背景和低质量图像中，仍可给出可解释、可约束的识别结果
- **保护地巡护与预警**：通过层级约束和干扰分析，降低误判带来的决策风险
- **科研辅助**：把“候选、证据、约束、风险”分层展示，便于专家快速审阅
- **教育科普与行业培训**：双语输出、清晰证据链，便于讲解“为什么是这个结果”
- **可持续迭代部署**：标注回写持续强化本地检索记忆，形成越用越准的业务闭环

## 参考资料

- BioCLIP 官方：https://imageomics.github.io/bioclip/
- 阿里云百炼模型文档入口：https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc/?type=model&url=3005961
- OpenAI 兼容 Qwen API：https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions
- 视觉理解文档：https://help.aliyun.com/zh/model-studio/vision
