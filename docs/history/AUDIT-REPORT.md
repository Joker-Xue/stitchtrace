# StitchTrace 0.1.0 软件质量与科学用途证据报告

2026-10-19；私有研究候选，供独立审计。**已形成可安装软件、两后端实际捕获、可重复实验及完整手写基线。工程可行性证据成立；没有算法创新、人体/病理用途、用户效率收益或录用结论。** 最重要的负面结果是：原冻结BBBC039配置全部在原生几何检查处失败；补充配置成功必须单独呈现。

## 交付与范围

核心wheel：`dist/stitchtrace-0.1.0-py3-none-any.whl`；Python API和CLI提供验证、检查、无推理回放、测量。核心仅依赖NumPy和POSIX标准库；原生捕获为optional extras。数据/模型/测试/报告不打进wheel。正式项目与全部pilot独立，未覆盖旧成果，未创建公开仓库、上传Zenodo、投稿、付款或作本项目公开许可证承诺。

暂名通过当日PyPI精确查询404及GitHub公开仓库名称检索0项；检索证据在research/discovery.json、github.txt。这不是商标或完整全球名称检索。

贡献边界是跨两类实际拼接语义的过程记录和可审核回放：Cellpose全图连边/packed对象，StarDist局部责任区保留/拒绝/写入。持久化不会把局部拒绝说成生物对象消失；受控干预造成的输出变化不会称为精度改善。

## 工程验证

| 要求 | 实测证据 | 适用边界 |
|---|---|---|
| 可安装、最小核心 | wheel安装、新环境NumPy-only回放，CLI verify成功 | Python3.11/macOS ARM64实测；不宣称全平台CI |
| 版本/schema拒绝 | schema1；包版本、源码SHA、函数签名锁定；26项测试全部通过 | 支持固定Cellpose4.2.1.1、StarDist0.9.2，不自动接受新版本 |
| 原子提交/幂等/恢复 | 同文件系统原子替换、fsync、manifest后提交、目录rename；同contract重复调用无推理 | POSIX本地文件系统；未证明断电、NFS或多主机事务 |
| 进程死亡恢复 | 测试进程在写入缓存后os._exit(17)，锁释放，缓存复用成功 | 人工故障测试，不是上游现存bug |
| 实际模型中断恢复 | 第二tile故意失败；4tile重跑只推理3次；恢复完成后同任务零推理 | 开发fixture，不计作BBBC039独立样本 |
| 缓存失效 | 输入/参数/源代码/predictor契约；StarDist实际权重/config/threshold指纹；改变权重拒绝同目标复用 | Cellpose外部callable状态需调用者正确更新predictor_id |
| 并发隔离 | 两个实际StarDist调用输出一致；线程级hook锁、目标文件锁、异常恢复原函数 | 同进程串行，不是高并发性能优化；无对任意外部线程patch/cwd修改的隔离承诺 |
| 坏输入与路径 | 损坏/缺文件、路径穿越、symlink、重复JSON key、未知schema、坏spacing/labels、图闭包节点拒绝 | 哈希非真实性签名；不防攻击者重写全部清单；仍需可信本地文件系统 |
| 原版等价 | 4张真实图、8组Cellpose oracle、3组StarDist3D oracle与原生/独立基线的partition和measurements一致 | Cellpose真实神经推断未测试；3D全部标为labeloracle |
| 无模型回放 | 最终wheel在新环境，没有4个推理/后端包；OS拒绝网络和实际权重读取；15项回放成功 | macOS临时sandbox profile，本机路径需在其他机器重定位 |

原始证据：results/release-tests.log（26 passed）、recovery/summary.json、oracle-summary.json、sd3d-oracle-summary.json、offline-audit.json、offline-release.log、one-command-rerun-02.log。

测量按数组轴序、零原点、整数像素中心；支持各向异性spacing。实际BBBC039使用像素单位，无微米校准声明。干净3D控制2→2；不跨seam的错误输入1→1，assembly未新增该错误。名为“2d-clean”的宽跨seam输入在原生路径仍2→6，不能改称成功的真值恢复负对照：测试通过的是忠实记录与原版等价。2D链1→3，3D链1→1；窄对象两维度均1→3。全部在oracle-summary保留。

## 冻结实验、失败与补充回归

工程规格docs/SPEC.md与experiments/PLAN.md在实现前冻结，SHA在research/frozen-plan-hashes.json。按官方split文件名排序，验证集前2、测试集前2；4个明确文件名在research/frozen-sample.json。没有筛图、替换失败样本或逐图调阈值。

原配置：全图p1/99.8归一化，block320、min_overlap64、context96、预训练默认阈值。四图的原生、工具、完整手写流程**12/12在Block.cover overlap断言处失败，尚未推理**。results/finite包含所有失败与traceback；不归因于模型准确性，也不把已声明限制叫新上游bug。

补充工程回归在experiments/AMENDMENT-01.md单独冻结：一次性将block改为512（不超过最短图像边520的最大2次幂），其他参数不变；不做搜索。它是观察失败后的工程修复，不是预注册或独立确认性实验。4图原生/工具/基线全部完成；最终增加实际权重及实现指纹后，仅再跑4次工具捕获确认，没有改变选图/模型/拼接参数。

| 固定图ID | 原生对象数 | retain_all对象数 | 工具/完整基线原输出与测量 |
|---|---:|---:|---|
| validation-0 | 104 | 115 | 全部一致 |
| validation-1 | 94 | 99 | 全部一致 |
| test-0 | 150 | 164 | 全部一致 |
| test-1 | 31 | 33 | 全部一致 |

干预复用同一缓存，保持模型、context和局部预测不变，只改变责任区选取及原顺序非零写入。不能把更多对象解释为更准确，也不能把块尺寸/context改变导致的预测差异归因于拼接。本实验没有对原失败320与512做“精度提升”比较。

补充参考指标由现成skimage交集表及sklearn ARI计算；它们与本软件互补，不当弱基线。4图前景Dice约0.909–0.967、全像素ARI约0.903–0.954，仅描述预训练模型与参考的分歧。未排除训练集重叠：BBBC官方说明部分图像与BBBC038/DSB2018重叠，因此不称为训练独立的模型精度评估。

指标首次包装遗漏了稀疏表显式零消除，可能夸大split/merge；已修复并保留reference-metrics-first-pass.json与旧脚本。更正后只统计非零对象交集，未改图像/参数/输出。补充指标不是assembly因果诊断。所有指标详情见reference-metrics.json。

## 完整手写基线与成本

experiments/handwritten.py从**相同原生图像+模型或oracle输入**开始，自行进行内部捕获、源版本检查、局部输出/图/过滤记录持久化、独立哈希清单及离线重构；不导入stitchtrace，不读取工具bundle或专用输出。原60行末端notebook不再代表完整工作流成本。

基线104物理行、91行非空非注释；包9个模块549物理行/499行非空非注释；测试4文件185物理行。行数只描述实现规模，不是认知负担或效率指标。基线没有工具的原子恢复缓存、并发hook锁、完整类型/路径/契约错误策略；不因此假设人类时间收益。两者都能从足够的原生事实获得相同诊断信息。

以下是单次独立进程实测：秒数从模型加载完成后开始，含首次预测编译与阶段I/O；RSS为整个进程峰值，含导入/模型加载。不同阶段独立进程、运行顺序固定，OS缓存和首次编译可影响结果。没有重复统计或显著性检验，不能据此宣称加速。

| 图ID | 原生秒 | 工具秒 | 完整基线秒 | 工具峰值RSS MiB | 工具/基线存储 MiB |
|---|---:|---:|---:|---:|---:|
| validation-0 | 2.063 | 1.453 | 1.426 | 674.8 | 9.27 / 9.25 |
| validation-1 | 1.381 | 1.507 | 1.539 | 675.0 | 9.26 / 9.25 |
| test-0 | 1.709 | 1.705 | 1.889 | 675.5 | 9.31 / 9.30 |
| test-1 | 1.156 | 1.282 | 1.229 | 671.9 | 9.20 / 9.19 |

原生只保存最终输出约1.38MiB；原生不提供完整持久化历史，不能用这个存储差异说明工具浪费。工具与完整基线均承担记录开销。全部原始时间/内存/存储字段在performance-final.json及各*-status.json；完整流程重跑有独立日志，表格不混用其数值。

可测的工程价值是：实际决策长期可读、无需推理依赖/权重重建、固定干预测量、契约失效与坏依赖拒绝、故障后有效tile复用。可重复执行一个受检接口替代每次重新拼装这些机制，有合理科研用途；**实际用户节省时间仍未测量，不作声明**。标准化工程集成无需数学能力独有，但仍须作者和独立审计确认用途及维护性。

## 数据、环境与许可

BBBC039官方页面 https://bbbc.broadinstitute.org/BBBC039 声明CC0并给出标准划分；掩膜按官方示例取第一通道并connected-component labeling。下载images77915748B、masks2753811B、metadata17816B，加前期记录905397B共81592772B，小于150MiB。三个archive声明解压总148103808B，小于1GiB；实际只提取固定4对。依赖包流量不计入原始数据上限，明确单列为环境成本。

复用已核验官方StarDist小模型ZIP5320433B、权重5771480B，未再下载新模型、不训练；模型额外100MiB上限未触及。本项目内复制既有模型缓存用于自包含重跑，并保留官方模型许可证。模型hash为42202bd269c8106782316f1a2c75afb3f5ffa65e525c2e155ee0ced3a95da349。公开许可证选择仍待实际作者决定，详见LICENSE-OPTIONS.md及third-party资料。

实测macOS26.6.2 ARM64、Python3.11.15；M5/24GB为用户环境说明。低权限sysctl探针被拒绝，未将其当作独立硬件核验。依赖完整锁在requirements-capture.lock、requirements-core.lock；wheel安装/构建日志保留。核心NumPy-only避免离线回放携带约2.5GiB捕获环境。

## 可复现性、所有失败与维护边界

- 一键reproduce.py在新项目内子目录重建环境与结果，禁止覆盖；第二次完整流程通过。第一次因外层执行沙箱拒绝Dask本地socket，及恢复脚本仍引用外部开发fixture而失败；修正为项目内fixture并在允许本地socket的执行环境重跑。两次日志/摘要都保留，未改系统安全设置。
- 最后两项纯validator加固（非空contract、整数节点）在26项测试和最终新环境wheel离线回放验证；此前捕获源码完整快照在research/source-snapshots/capture-v1，逐文件SHA与捕获metadata匹配。不要仅靠相同0.1.0字符串推断每个开发阶段字节相同，以记录的实现hash为准。
- 最终离线审计曾误将锁文件当bundle目录，失败日志保留；修正枚举条件后15项成功。首次metadata解码读到macOS资源文件失败、首次smoke错误要求真值恢复、首次指标零值问题也均保留。
- raw模型输出/决策/bundle不含执行代码或pickle；工具核心不会加载权重。完整模型与原始下载档案作为独立本地研究材料，不进入bundle/wheel。
- 原生内部hook依赖维护者审阅上游改动。不能承诺任意版本/任意动态猴补丁兼容；锁定未支持版本而拒绝是明确代价。原生2D Cellpose跨seam行为可能明显不同于3D，此软件忠实记录而非自动修复。
- 没有Windows、多主机、NFS、断电、超大体积流式性能或任意并发外部应用安全证明；没有人类用户研究；没有真实3D神经模型或实际Cellpose神经预测验证。3D是明确labeloracle。
- 初版只提供两个固定干预，不改变预测context、不搜索阈值、不实现通用替代拼接算法库。报告对象变化，不自动判定错误。

**供审计决策：可以将此作为有实际工程功能的SoftwareX研究候选继续审阅；不能因短notebook而否定，也不能因测试通过而宣布论文贡献已获认可。** 当前主要下一步是外部代码/用途审计与维护范围确认，而非继续堆功能或扩大未经计划的数据。
