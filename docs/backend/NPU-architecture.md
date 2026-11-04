# 完整模型 DSL 接入架构设计（B 路，Model API v3）

更新日期：2026-11-04。本文描述当前已实现的架构，不将未来能力表述为现有支持。

## 1. 目标与执行路线

通过 llama.cpp 复用 DSL 已有的完整模型 prefill/decode：llama.cpp 保留
Tokenizer、Context 入口、采样及文本输出，模型计算由 DSL Session 在 NPU 上完成。
模型专属知识集中在 DSL repo。新增符合现有接口的模型 DSL，不修改 llama 通用桥接逻辑。

当前采用 **B 路完整模型执行**，而不是 **A 路 GGML 算子/子图卸载**：

- B 路：`llama_decode()` → 通用桥接 → 完整模型 Session → NPU → logits。
- A 路：GGML 模型图 → 调度器 → 按算子或融合子图选择后端；本架构未交付此路线。
- 当前 GGML NPU 数值 `supports_op` 返回 false；设备注册不等于算子执行支持。
- 当前入口是定制 fork 中显式启用的完整模型路径，不是普通 `llama-cli` 自动执行任意 GGUF。

## 2. 总体架构与仓库边界

```text
llama.cpp repo
┌─────────────────────────────────────────────────────────────┐
│ 应用 / llama-npu-model                                      │
│ 模型元数据、Tokenizer、Context、采样、文本输出                │
│                         │ llama_decode()                    │
│ 通用完整模型桥接层                                         │
│ · 适配 llama 请求与输出                                     │
│ · 调用 DSL 注册/契约接口选择并验证执行实现                   │
│ · 验证资产完整性与 tokenizer 绑定                           │
│ · 创建 Session、检查执行边界、发布 logits                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ Model API v3（主机 C++17 接口）
libraries.npu.dsl repo    │
┌─────────────────────────▼───────────────────────────────────┐
│ 通用 Model API                                             │
│ Registry / Provider 描述符 / 契约校验 / 资产及 Session 接口   │
│                         │                                  │
│ 模型专属 Provider（当前真实实现：Qwen）                      │
│ 模型语义、权重表示、资产准备、Session 适配                    │
│                         │                                  │
│ 完整模型 DSL prefill / decode 与模型 Session 实现            │
└─────────────────────────┬───────────────────────────────────┘
                          │
frameworks.ai.npu.generic_compute repo
┌─────────────────────────▼───────────────────────────────────┐
│ NPU Runtime / 设备资源 / 提交与同步                          │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
                         NPU
```

**边界原则：llama 负责宿主适配，DSL 负责模型契约与完整计算，Runtime 负责设备执行。**
桥接层发起 Provider 查找，但注册表、通用匹配逻辑和模型专属约束由 DSL 提供。
Runtime 是独立的第三个 repo，不属于 DSL repo。

### 2.1 职责分配

| 所在仓库 | 模块 | 负责 | 不负责 |
| --- | --- | --- | --- |
| llama.cpp | 应用与 Context | 读取元数据、分词、调用 decode、采样、文本输出 | 解释 DSL 张量布局、实现模型 kernel |
| llama.cpp | 通用桥接 | 宿主选项转换、完整性验证、调用契约校验、执行边界与 logits 发布 | 硬编码 Qwen 维度或专属文件结构 |
| llama.cpp | 资产绑定工具 | 调用 Provider 准备/探测资产，计算绑定信息，发布新的绑定资产 | 转换或重新量化权重、证明数值正确性 |
| DSL | 通用 Model API | 注册、模型/格式/请求校验、资产和 Session 抽象 | 包含 llama 或模型生成 kernel 头文件 |
| DSL | 模型 Provider | 模型语义、表示约束、资产解析准备、完整 Session 适配 | tokenizer 与采样 |
| DSL | 完整模型实现 | prefill/decode、内部状态与模型执行资源 | GGML 图调度 |
| generic_compute | Runtime | 设备执行所需的资源、提交、同步等机制 | 模型匹配、tokenizer、采样 |

## 3. DSL 通用接口

接口定义位于 DSL 的 [model_api.h](../../../libraries.npu.dsl/model_api/model_api.h)。
当前 `interface_version = 3`，是主机 C++17 接口，不是稳定 C ABI 或动态插件协议。

| 接口 | 含义 |
| --- | --- |
| `registry` | 保存 Provider 描述符副本，拒绝重复 ABI，提供直接路由和契约匹配 |
| `provider` | 声明架构、模型容量、能力、语义约束、执行格式、请求限制及执行回调 |
| `asset_contract` | 完整性封装：固定顺序的文件名、大小上限、identity header |
| `asset_source` | 通用资产大小查询、范围读取和位置接口，不解释张量 |
| `asset_preparation` | Provider 验证后返回的执行格式证据与不可变自有 payload |
| `prepared_assets` | 保留资产源、payload、绑定身份和对应执行实现的受控句柄 |
| `session_options` | context、逻辑/物理 batch、序列数、输入输出及 cache 请求 |
| `effective_config` | 实际逻辑限制、分配容量、物理 batch 能力、固定分配标志和状态能力 |
| `session` | 完整模型 evaluate/reset、位置、故障状态、提交计数和实际配置 |

### 3.1 Provider 不是 checkpoint 专属实例

Provider 适配一类具有明确语义与表示约束的完整模型 DSL。
相同架构名称不保证兼容；相同契约的不同 checkpoint 可以复用同一 Provider。
用户提供模型和执行资产，不需要额外手选 Provider。

### 3.2 执行实现与句柄安全

`provider::set_execution()` 将准备和创建回调一起安装到私有、共享、不可变实现中。
未改变的描述符副本共享实现身份；替换任一回调都会产生新身份。

`prepared_assets` 绑定 ABI 和该实现身份。`create_session()` 在调用模型工厂前拒绝
空句柄、未验证句柄、错误 ABI/版本或不同实现准备的句柄，避免错误解释不透明 payload。
句柄同时保留资产源、payload 和回调捕获的生命周期；Session 如依赖这些资源，必须保留句柄。

这些约束用于受信任进程内代码的类型/生命周期安全，不构成恶意插件沙箱或发布者认证。

## 4. 三种独立身份与匹配规则

| 维度 | 表示 | 解决的问题 |
| --- | --- | --- |
| 接口兼容性与路由 | 接口版本、能力位、Provider/package ABI | 能否调用，路由到哪个实现 |
| 执行契约 | 模型语义、实际权重表示、I/O、请求与状态限制 | 该实现能否正确解释并执行输入 |
| 资产绑定 | 每文件 SHA256、package ID、tokenizer fingerprint | 当前使用的是否为绑定的那份资产 |

**找到 Provider ≠ 契约匹配；契约匹配 ≠ 数值验收通过。Hash 不是 checkpoint 白名单。**

### 4.1 路由

1. 存在 `npu.dsl.abi` 时直接路由到注册实现；未知或不兼容立即拒绝，不回退猜测。
2. 绑定工具处理未指定 ABI 的模型时，按 architecture 筛选，再验证候选完整契约与资产。
3. 无匹配则报告原因；多个匹配则报告歧义，不按注册顺序选择。
4. 当前运行时仍要求已绑定的 GGUF 与外部资产，不自动执行未绑定模型。

### 4.2 参数级兼容性

- 语义：实际 vocabulary、模型维度、attention/expert/RoPE 等 Provider 约束。
- 表示：`weights`、`scales`、`dense`、`head`、`layout`、`group_size`。
- 请求：context、逻辑/物理 batch、序列数、token 输入、logits 输出、cache 类型。
- 能力：顺序 token、最后位置 logits、reset、计数及永久故障语义。

六项格式元数据使用 `npu.dsl.format.*`；声明任意一项就必须完整提供并匹配六项。
全部缺失时仍需经过 Provider 资产准备得到有效格式证据，不能仅由 ABI 推断。
校验失败保留 `field / expected / actual` 诊断。

### 4.3 完整性与可信边界

宿主验证文件内容、绑定 identity 和实际 tokenizer 指纹后，才设置
`asset_identity.integrity_verified`。它是受信任宿主的进程内确认，不是签名认证。
新权重产生新的 hash/package ID，需要重新绑定；旧绑定必须拒绝变更后的资产。

Hash 与结构校验不能证明原始无类型字节的量化语义、checkpoint/tokenizer 等价性或数值准确性。
特别是 legacy Qwen 资产，错误但尺寸相同的打包仍可能通过结构与新 hash 检查，因此必须保留数值验收。

## 5. 资产绑定、初始化与执行流程

### 5.1 离线绑定

```text
模型元数据 / Tokenizer GGUF + 外部执行资产
    → 查找并验证候选 Provider
    → Provider 准备资产，验证布局并返回格式证据
    → 验证模型完整契约
    → 计算文件 hash、package ID、tokenizer 绑定
    → 发布新的 metadata/tokenizer GGUF 与 identity
```

绑定探测不创建执行 Session、不探测硬件。未经过宿主完整性确认的准备句柄不能通过
`create_session()` 执行。绑定工具不做权重转换或重新量化。

### 5.2 运行时初始化

```text
vocab_only 加载已绑定 GGUF + npu_model_path 指定外部资产
    → 根据 npu.dsl.abi 获取 Provider
    → 模型元数据与 llama 请求预检查
    → 验证 package/tokenizer identity、文件大小、路径及所有内容 hash
    → Provider prepare_assets：解析并验证执行资产
    → 携带格式证据再次验证模型契约
    → 规范化 session_options
    → create_session：校验句柄，调用工厂，检查 effective_config
    → 可用 Context / Session
```

通用桥接不要求 manifest/blob 格式。Qwen 显式使用可选的 `use_legacy_package()`
适配器，继续校验原有 manifest、生成张量 ABI、对齐、范围和层模式。

当前 `directory_assets()` 提供平面目录内声明文件的规范路径限制、大小/范围检查，
单次读取上限为 16 MiB。当前完整性封装仍是固定文件集合，不是任意远程/归档资源协议。
资产句柄不是锁定的文件快照，宿主必须保证从验证到执行期间外部文件不变。

### 5.3 每轮推理

```text
文本 → llama tokenizer → token IDs
    → llama_decode()
    → 桥接检查位置、token、context、逻辑/物理 batch 和状态
    → Session.evaluate()
    → 完整模型 DSL prefill / decode → Runtime → NPU
    → 最后 token 的完整 FP32 logits
    → llama 采样 → token → 文本，进入下一轮
```

接口同步执行。不在桥接内部静默拆分超限 batch，不自动 CPU 模型回退或重建 Session。
prefill/decode 的具体执行细节属于模型实现，不由 GGML 算子调度器逐项安排。

## 6. 配置、状态与错误语义

### 6.1 请求与实际配置

- context 为 0 表示 Provider 默认容量，不是无限容量。
- 逻辑 batch 规范化受 context 限制；物理 batch 受逻辑 batch 限制。
- llama 侧 `n_ubatch=0` 选择逻辑 batch；每次 decode 必须同时满足两个上限。
- 工厂收到明确配置，返回 `effective_config`；不一致的配置在 evaluate 前拒绝。
- 逻辑 context 不等于物理分配。当前 Qwen 即使请求 context=128，仍分配固定 65792 位置容量。
- 有效配置在 Session 生命周期内必须保持稳定。

### 6.2 状态归属

cache/recurrent state 由 Session 管理，宿主不解释其张量布局。
当前只表达并接受 append/reset；不提供通用 KV 拷贝、截断或序列化接口。
不支持的 llama 状态操作明确拒绝；无法直接返回错误的操作会锁存拒绝并阻止后续 decode。

### 6.3 故障与资源

参数错误必须发生在状态/计数被修改之前；桥接拒绝调用时使旧输出不可继续使用。
尝试设备提交计入计数，设备错误可能永久 poison Session。
reset 复位健康 Session 的位置/状态，但保留计数，不能恢复已 poison 的 Session。
Registry 用于启动阶段注册，不支持并发修改；Session 不声明线程安全。

## 7. 构建与扩展

| 目标 | 内容 |
| --- | --- |
| `npudsl_model_core` | 纯主机 C++17 通用实现，无 llama/Qwen/生成代码/Runtime 依赖 |
| `npudsl_model_api` | 注册入口与静态 Provider 工厂聚合 |
| `npudsl_model_qwen36` | 可选 Qwen 适配，依赖原有 `qwen36_session` |

`NPUDSL_MODEL_QWEN36` 默认 ON 保持兼容；OFF 不进入 Qwen 构建，llama 主机也无需
`GENERIC_COMPUTE_ROOT`。空注册表只代表核心可用，不代表能执行真实模型。

静态扩展使用 `NPUDSL_MODEL_PROVIDER_SOURCES`、`NPUDSL_MODEL_PROVIDER_FACTORIES`
和 `NPUDSL_MODEL_PROVIDER_LIBRARIES`。生成注册入口显式调用工厂，避免静态归档被裁剪。
独立 Provider 库依赖 core，不反向依赖聚合入口。

新增模型的步骤：

1. 在 DSL repo 实现 Provider 工厂与完整模型 Session。
2. 声明模型语义、权重表示、I/O、状态能力和请求上限。
3. 实现资产准备回调，返回有效格式证据和自有 payload。
4. 实现 Session 创建/配置回报，管理内部状态和资源生命周期。
5. 加入 DSL 侧静态构建注册，完成绑定/负向/生命周期测试。
6. 使用独立参考完成真实设备数值验收。

若新模型仍符合顺序 token 输入、最后位置 FP32 logits、append/reset，则无需修改 llama
桥接代码。多模态、多序列或新状态操作超出现有协议，需要扩展通用接口，不能仅增加 Provider。
静态注册扩展通常需要重新构建，不表示可在运行时热加载。
共享 llama 库应通过库内部的构建工厂注册；仅在可执行程序中注册的方式只验证了静态宿主。

## 8. 当前支持与明确限制

- 只有 Qwen3.6 Provider 有真实模型数值证据；测试 Provider 不代表第二个模型支持。
- 当前真实权重表示：主量化权重 symmetric signed INT4/group128，group-major FP16 scales，
  FP16/FP32 辅助张量，以及 symmetric signed INT8/group128 输出头；不能泛称任意 INT4。
- 当前使用 metadata/tokenizer GGUF + 外部 DSL 权重，不执行 GGUF 中任意量化权重。
- 不自动转换 Q4_K 等 GGUF 表示，不支持任意模型自动生成完整 DSL。
- 不交付动态插件、稳定 C ABI、多序列、多模态或通用状态保存恢复。
- 不以短提示回归声明长上下文认证，也不因架构重构声明性能提升。

已验证：核心 44 项检查、独立静态工厂扩展 45 项检查；Qwen-OFF 主机 17 项通过、1 项
真实模型测试预期跳过；默认配置 21/21 回归通过。真实 Qwen 审计的三行完整 logits 对独立
参考 mean relative L2 为 0.0065263350589211148，最大 0.011090676171006746，argmax 3/3
一致，reset/replay 位级一致。具体条件和证据见 [NPU-validation.md](NPU-validation.md)。

## 9. 代码索引

以下跨仓库链接按三个 repo 在同一父目录下部署的结构组织。

| 模块 | 文件 |
| --- | --- |
| llama 桥接 | [src/llama-npu.cpp](../../src/llama-npu.cpp)、[src/llama-npu.h](../../src/llama-npu.h) |
| Context 接入 | [src/llama-context.cpp](../../src/llama-context.cpp) |
| 应用与绑定工具 | [examples/npu-model/npu-model.cpp](../../examples/npu-model/npu-model.cpp)、[package-bind.cpp](../../examples/npu-model/package-bind.cpp) |
| DSL 通用接口与实现 | [model_api.h](../../../libraries.npu.dsl/model_api/model_api.h)、[model_api.cpp](../../../libraries.npu.dsl/model_api/model_api.cpp) |
| Qwen Provider | [qwen36_provider.cpp](../../../libraries.npu.dsl/model_api/qwen36_provider.cpp) |
| DSL 构建入口 | [model_api/CMakeLists.txt](../../../libraries.npu.dsl/model_api/CMakeLists.txt) |
| 详细契约 | [NPU-model-contract.md](NPU-model-contract.md) |
| 验证记录 | [NPU-validation.md](NPU-validation.md) |
| 部署说明 | [NPU-deployment.md](NPU-deployment.md) |