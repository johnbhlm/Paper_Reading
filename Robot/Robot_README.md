# Robot-Papers


# 1. Data
1. [ Open X-Embodiment ](https://robotics-transformer-x.github.io/)
2.  [RLBench](https://sites.google.com/view/rlbench) --- [Paper](https://arxiv.org/abs/1909.12271) --- [RLBench Code](https://github.com/stepjam/RLBench) 


# 2. Metrics
# 3. 比赛


# 4. Paper List
## 4.1 Survey

## 4.2 Papers

### 2024
1. Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation --- [CVPR 2024](https://arxiv.org/abs/2403.03890) --- [Project web](https://yusufma03.github.io/projects/hdp/) --- [Code](https://github.com/dyson-ai/hdp)
	>本文介绍了分层扩散策略（HDP），这是一种用于多任务机器人操作的分层代理。
HDP 将操纵策略分解为层次结构：预测远距离次优末端效应器姿势 (NBP) 的高级任务规划代理，以及生成最佳运动轨迹的低级目标条件扩散策略。分解的策略表示使 HDP 能够处理长期任务规划，同时生成细粒度的低级别行动。为了在满足机器人运动学约束的同时生成上下文感知的运动轨迹，我们提出了一种新颖的运动学感知目标条件控制代理，机器人运动学扩散器（RK-Diffuser）。具体来说，RK-Diffuser 学习生成末端执行器姿态和关节位置轨迹，并通过可微分运动学将精确但运动学未知的末端执行器姿态扩散器提炼为运动学感知但不太准确的关节位置扩散器。
根据经验，我们表明 HDP 在模拟和现实世界中都比最先进的方法取得了显着更高的成功率。
Key words: RLBench, 
2. BiGym: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark --- [Paper 2024.07](https://arxiv.org/pdf/2407.07788) --- [ Project Web ](https://chernyadev.github.io/bigym/) --- [Code](https://github.com/chernyadev/bigym)
	>我们推出 BiGym，这是一种用于移动双手演示驱动的机器人操作的新基准和学习环境。 BiGym 在家庭环境中提供 40 种不同的任务，从简单的目标达成到复杂的厨房清洁。为了准确地捕捉现实世界的表现，我们为每项任务提供了人类收集的演示，反映了现实世界机器人轨迹中发现的多种模式。 BiGym 支持各种观察，包括本体感受数据和视觉输入（例如 RGB）以及来自 3 个摄像机视图的深度。为了验证 BiGym 的可用性，我们在环境中对最先进的模仿学习算法和演示驱动的强化学习算法进行了彻底的基准测试，并讨论了未来的机会。
	Key words: Manipulation Benchmark, long-horizon task, MuJoCo, Unitree_h1 , Demostration
4. RePLan: Robotic Replanning with Perception and Language Models --- [Paper](https://arxiv.org/abs/2401.04157) --- [ Webset ](https://replan-lm.github.io/replan.github.io/)
	>Motivation:  利用视觉语言模型进行机器人重新规划，可以为 long -horizon 任务提供重新规划能力。
	大型语言模型（LLM）的进步已经证明了它们在促进高级推理、逻辑推理和机器人规划方面的潜力。最近，LLMs 还能为低级机器人行动生成奖励函数，有效地连接了高级规划与低级机器人控制之间的界面。然而，挑战依然存在，即使制定了语法正确的计划，由于计划不完善或意外环境问题，机器人仍可能无法实现预期目标。
	Solution: 利用视觉语言模型的能力，作者提出了一个名为 "利用感知和语言模型进行机器人重新规划"（RePLan）的新框架，该框架能够为 long-horizon 任务提供在线重新规划能力。该框架利用 VLM 对世界状态的理解所提供的物理基础，在初始计划无法实现预期目标时调整机器人的行动。作者开发了一个推理与控制（RC）基准，其中包含八个long-horizon任务来测试该方法。RePLan 能够让机器人成功适应不可预见的障碍，同时完成开放式的 long-horizon 目标，而 baseline 模型则无法做到这一点，并且可以很容易地应用到真实机器人中。
5. Octo: An Open-Source Generalist Robot Policy --- [ICRA 2024](https://arxiv.org/abs/2405.12213) --- [Project web ](https://octo-models.github.io/) --- [Code](https://github.com/octo-models/octo)
	>Motivation: 开发具有泛化性的机器人操作通用策略。
	在各种机器人数据集上预先训练的大型策略有可能改变机器人的学习方式：这种通用型机器人策略无需从头开始训练新策略，只需少量域内数据即可进行微调，但却具有广泛的通用性。然而，为了广泛适用于各种机器人学习场景、环境和任务，这些策略需要处理不同的传感器和行动空间，适应各种常用的机器人平台，并能根据新领域随时高效地进行微调。
	Solution: 作者引入了 Octo，一个基于 Transformer 的大型策略，在 Open X-Embodiment 数据集（迄今为止最大的机器人操作数据集）的 800k 轨迹上进行了训练。它可以通过语言命令或目标图像进行指令，并且可以在标准消费级 GPU 上在几个小时内通过新的感官输入和动作空间有效地微调机器人设置。在 9 个机器人平台的实验中，证明了 Octo 作为一种多功能策略初始化，可以有效地微调到新的观察和行动空间。作者还对 Octo 模型的设计决策（从架构到训练数据）进行了详细的消融，以指导未来构建通用机器人模型的研究

6. SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning --- [ICRA 2024](https://arxiv.org/abs/2401.16013) --- [Project web](https://serl-robot.github.io/) --- [Code](https://rail-berkeley.github.io/serl/)
	>Motivation : 机器人 RL 的广泛应用以及机器人 RL 方法相对来说难以使用。开发一个RL 学习库。
	Solution: 作者开发了一个精心实施的库，其中包含一个高效的非政策深度 RL 方法样本，以及计算奖励和重置环境的方法、一个广泛采用的机器人的高质量控制器和一些具有挑战性的示例任务。提供了这个库作为社区资源，介绍了它的设计选择，并展示了实验结果。也许令人惊讶的是，平均每个策略只需 25 到 50 分钟的训练就能获得 PCB 板组装、电缆布线和物体搬迁的策略，比文献中类似任务的最新结果要好。这些策略实现了完美或近乎完美的成功率，即使在扰动下也具有极强的鲁棒性，并表现出突发性的恢复和修正行为。
	Key words: 真实环境， Franka 机械臂，Mujoco simulation gym environment, Reinforcement Learning

7. SCENEREPLICA: Benchmarking Real-World Robot Manipulation by Creating Replicable Scenes --- [ICRA 2024](https://arxiv.org/abs/2306.15620) --- [Code](https://github.com/IRVLUTD/SceneReplica)
	>提出了一个新的可重复基准来评估现实世界中的机器人操作，特别关注拾取和放置。我们的基准测试使用 YCB 对象（机器人社区中常用的数据集）来确保我们的结果与其他研究具有可比性。此外，该基准的设计目的是可以在现实世界中轻松复制，以便研究人员和从业者可以使用它。我们还提供了基于模型和无模型 6D 机器人抓取基准的实验结果和分析，其中评估了代表性算法的对象感知、抓取规划和运动规划。我们相信，我们的基准测试将成为推动机器人操纵领域发展的宝贵工具。通过提供标准化的评估框架，研究人员可以更轻松地比较不同的技术和算法，从而在开发机器人操纵方法方面取得更快的进展。
8.  ASGrasp: Generalizable Transparent Object Reconstruction and Grasping from RGB-D Active Stereo Camera --- [ICRA 20224](https://arxiv.org/abs/2405.05648) --- [Project Web](https://pku-epic.github.io/ASGrasp/) --- [Code](https://github.com/jun7-shi/ASGrasp)
	>Motivation: 探讨了如何抓取透明和镜面物体的问题。这个问题非常重要，但由于深度摄像头无法恢复物体的精确几何形状，因此在机器人领域仍未得到解决。
	Solution: 作者首次提出了 ASGrasp，一个使用 RGB-D 主动立体摄像机的 6-DoF 抓取检测网络。ASGrasp 利用基于学习的双层立体网络实现透明物体重建，从而在杂乱的环境中抓取与材料无关的物体。现有的基于 RGB-D 的抓取检测方法在很大程度上依赖于深度还原网络和深度摄像头生成的深度图的质量，与之相比，该系统能够直接利用原始红外和 RGB 图像来重建透明物体的几何形状。以 GraspNet-1Billion 为基础，通过域随机化创建了一个广泛的合成数据集。实验证明，ASGrasp 通过从模拟到现实的无缝传输，在模拟和现实中都能实现超过 90% 的通用透明物体抓取成功率。该方法明显优于 SOTA 网络，甚至超过了完美可见点云输入所设定的性能上限。
9. Towards Diverse Behaviors: A Benchmark for Imitation Learning with Human Demonstrations --- [Paper](https://arxiv.org/abs/2402.14606)
	>利用人类数据进行模仿学习在教授机器人多种技能方面取得了显着的成功。然而，人类行为固有的多样性导致多模态数据分布的出现，从而对现有的模仿学习算法提出了巨大的挑战。量化模型有效捕获和复制这种多样性的能力仍然是一个悬而未决的问题。在这项工作中，我们介绍了模拟基准环境和相应的具有多种人类模仿学习演示的数据集（D3IL），其明确设计用于评估模型学习多模态行为的能力。我们的环境被设计为涉及需要解决的多个子任务，考虑操纵多个对象，这增加了行为的多样性，并且只能通过依赖闭环感官反馈的策略来解决。其他可用的数据集至少缺少这些具有挑战性的特性之一。为了解决多样性量化的挑战，我们引入了易于处理的指标，这些指标为模型获取和重现多样化行为的能力提供了有价值的见解。这些指标提供了评估模仿学习算法的鲁棒性和多功能性的实用方法。此外，我们对所提出的任务套件的最先进方法进行了彻底的评估。该评估可作为评估他们学习不同行为的能力的基准。我们的研究结果揭示了这些方法在解决捕获和概括多模式人类行为的复杂问题方面的有效性，为未来模仿学习算法的设计提供了有价值的参考

10. Generalize by Touching: Tactile Ensemble Skill Transfer for Robotic Furniture Assembly --- [Paper](https://arxiv.org/abs/2404.17684)
	>由于任务周期长和不可概括的操作计划，家具组装仍然是机器人操作中未解决的问题。本文提出了触觉集成技能转移 (TEST) 框架，这是一种开创性的离线强化学习 (RL) 方法，它将触觉反馈纳入控制环路。 TEST 的核心设计是学习用于高层规划的技能转换模型，以及一套自适应的技能内目标达成策略。这种设计旨在以更通用的方式解决机器人家具组装问题，促进这项长期任务的技能无缝链接。我们首先从一组启发式策略和轨迹中进行样本演示，这些策略和轨迹由一组随机的子技能部分组成，从而能够获取丰富的机器人轨迹，捕获技能阶段、机器人状态、视觉指示器，以及最重要的触觉信号。利用这些轨迹，我们的离线强化学习方法可以识别技能终止条件并协调技能转换。我们的评估强调了 TEST 对分销家具组件的熟练程度、对看不见的家具配置的适应性以及对视觉干扰的鲁棒性。消融研究进一步强调了两个算法组件的关键作用：技能转换模型和触觉集成策略。结果表明，在分布内和泛化设置中，TEST 可以实现 90% 的成功率，并且比启发式策略的效率高出 4 倍以上，这表明了一种用于接触丰富的操作的可扩展的技能转移方法


11. THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation --- [ RSS 2024](https://arxiv.org/abs/2402.08191)
	>为了实现有效的大规模、现实世界的机器人应用，我们必须评估我们的机器人策略如何适应环境条件的变化。不幸的是，大多数研究都是在与训练设置非常相似甚至相同的环境中评估机器人的性能。我们推出了 THE COLOSSEUM，这是一种新颖的模拟基准，具有 20 种不同的操作任务，可以对 14 个环境扰动轴的模型进行系统评估。这些扰动包括物体、桌面和背景的颜色、纹理和大小的变化；我们还改变照明、干扰物、物理属性扰动和相机姿势。使用 THE COLOSSEUM，我们比较了 5 种最先进的操纵模型，结果表明，在这些扰动因素的影响下，它们的成功率下降了 30-50%。当同时应用多个扰动时，成功率降低 ≥75%。我们发现，改变干扰物体的数量、目标物体的颜色或照明条件是最能降低模型性能的扰动。为了验证我们结果的生态有效性，我们表明我们的模拟结果与现实世界实验中的类似扰动相关（R 2 = 0.614）。我们开源代码供其他人使用 THE COLOSSEUM，还发布代码来 3D 打印用于复制现实世界扰动的对象。最终，我们希望 COLOSSEUM 能够成为确定建模决策的基准，从而系统地提高操纵的泛化能力

12. Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning --- [RSS 2024](https://arxiv.org/abs/2405.18196) --- [Project web](https://vv19.github.io/render-and-diffuse/)
	>在机器人学习领域，高维观察（例如 RGB 图像）和低级机器人动作（两个本质上非常不同的空间）之间的复杂映射构成了复杂的学习问题，尤其是在数据量有限的情况下。在这项工作中，我们引入了渲染和漫反射 (R&D) 方法，该方法使用机器人 3D 模型的虚拟渲染来统一图像空间内的低级机器人动作和 RGB 观察。使用这种联合观察动作表示，它使用学习的扩散过程来计算低级机器人动作，该过程迭代地更新机器人的虚拟渲染。这种空间统一简化了学习问题，并引入了对于样本效率和空间泛化至关重要的归纳偏差。我们在模拟中全面评估了研发的几种变体，并展示了它们在现实世界中六项日常任务中的适用性。我们的结果表明，R&D 表现出强大的空间泛化能力，并且比更常见的图像到动作方法具有更高的样本效率。
13. LHManip: A Dataset for Long-Horizon Language-Grounded Manipulation Tasks in Cluttered Tabletop Environments --- [RSS 2024](https://arxiv.org/abs/2312.12036) --- [Code](https://github.com/fedeceola/LHManip)
	>长期以来，指导机器人在家中完成日常任务一直是机器人技术面临的挑战。虽然最近在语言条件模仿学习和离线强化学习方面取得了进展，在广泛的任务中表现出了令人印象深刻的性能，但它们通常仅限于短视距任务，不能反映家用机器人预期完成的任务。虽然现有架构具有学习这些预期行为的潜力，但由于缺乏用于真实机器人系统的必要长视距、多步骤数据集，因此面临着巨大的挑战。为此，我们提出了长视距操纵（LHManip）数据集，该数据集由 200 个事件组成，通过真实机器人远程操作演示了 20 种不同的操纵任务。这些任务包含多个子任务，包括在高度杂乱的环境中抓取、推动、堆叠和投掷物体。每项任务都配有自然语言指令和用于点云或 NeRF 重建的多摄像头视点。该数据集共包含 176,278 对观察-动作，是开放 X-Embodiment 数据集的一部分

14. Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks --- [ICLR 2024](https://arxiv.org/abs/2405.01534) --- [ Project web](https://mihdalal.github.io/planseqlearn/) --- [Code](https://github.com/mihdalal/planseqlearn)
	>Motivation: 是否可以将 LLMs 中的互联网尺度知识用于高级策略，指导强化学习（RL）策略，从而在线高效地解决机器人控制任务，而无需预先确定一套技能呢？
	大型语言模型（LLM）已被证明能够执行 long-horizon 机器人任务的高级规划，但现有的方法需要访问预定义的技能库（如picking、placing、pulling、pushing、navigating）。然而，LLM 规划并不涉及如何设计或学习这些行为，尤其是在 long-horizon 环境中，这仍然具有挑战性。此外，对于许多感兴趣的任务，机器人需要能够以细粒度的方式调整其行为，这就要求代理能够修改底层控制动作。
	>\
	>Solution: 作者提出了 Plan-Seq-Learn (PSL)，一种模块化方法，利用运动规划来弥合抽象语言与学习到的底层控制之间的差距，从而从头开始解决 long-horizon 机器人任务。PSL 在多达 10 个阶段的超过 25 个具有挑战性的机器人任务中取得了最先进的成果。PSL 解决了跨越四个基准的原始视觉输入的long-horizon任务，成功率超过 85%，优于基于语言的方法、经典方法和端到端方法。
	Key words: Long Horizon, LLM,

15. JUICER: Data-Efficient Imitation Learning for Robotic Assembly --- [Paper](https://arxiv.org/abs/2404.03729) --- [Project Web](https://imitation-juicer.github.io/) --- [Code](https://github.com/ankile/imitation-juicer)
	>虽然从演示中学习对于获取视觉运动策略非常有效，但对于需要精确、长视距操作的任务来说，没有大量演示数据集的高性能模仿仍然具有挑战性。本文提出了一种利用少量人类演示预算提高模仿学习性能的方法。我们将这一方法应用于装配任务中，这些任务需要在较长的时间跨度和多个任务阶段内精确地抓取、调整和插入多个部件。我们的管道结合了富有表现力的策略架构以及用于数据集扩展和基于模拟的数据增强的各种技术。这些技术有助于扩大数据集支持，并在需要高精度的瓶颈区域附近对模型进行局部纠正监督。我们在四项家具装配任务的仿真中演示了我们的管道，使机械手能够在近 2500 个时间步骤内直接从 RGB 图像中装配多达五个部件，表现优于模仿和数据增强基线。
### 2023
1. Open X-Embodiment: Robotic Learning Datasets and RT-X Models --- [paper](https://arxiv.org/abs/2310.08864) --- [Project Web ](https://robotics-transformer-x.github.io/) --- [Code](https://github.com/google-deepmind/open_x_embodiment)
	>Motivation: 能否训练出通用的 X-机器人策略，使其能有效地适应新的机器人、任务和环境？
	在不同数据集上训练的大型高容量模型在高效处理下游应用方面取得了显著的成功。在从 NLP 到计算机视觉等领域，这导致了预训练模型的整合，通用预训练骨干成为许多应用的起点。机器人技术能否实现这种整合？传统的机器人学习方法是为每种应用、每个机器人甚至每个环境训练一个单独的模型。
	Solution: 作者提供了标准化数据格式和模型的数据集，以便在机器人操纵的背景下探索这种可能性，同时还提供了有效 X 机器人策略的实验结果。汇集了 21 家机构合作收集的 22 种不同机器人的数据集，展示了 527 种技能（160266 项任务）。研究表明，在这些数据基础上训练出来的大容量模型（称之为 RT-X）可以利用来自其他平台的经验，实现正迁移并提高多个机器人的能力。

2. LoHoRavens: A Long-Horizon Language-Conditioned Benchmark for Robotic Tabletop Manipulation --- [Paper](https://arxiv.org/abs/2310.12020) --- [Project web](https://cisnlp.github.io/lohoravens-webpage/) --- [Code](https://github.com/Shengqiang-Zhang/LoHo-Ravens)
	>智能代理和大型语言模型（LLM）的融合为智能指令跟踪带来了重大进步。尤其是 LLMs 强大的推理能力，使得机器人可以在没有昂贵的注释演示的情况下执行长视距任务。然而，用于测试各种场景下语言条件机器人的长视距推理能力的公开基准仍然缺失。为了填补这一空白，这项工作将重点放在桌面操作任务上，并发布了一个模拟基准--textit{LoHoRavens}，它涵盖了颜色、大小、空间、算术和参考等多个长视距推理方面。此外，在使用 LLM 的长视距操纵任务中还存在一个关键的模态桥接问题：如何将机器人执行过程中的观察反馈纳入 LLM 的闭环规划，然而之前的工作对此研究较少。作者研究了两种弥合模态差距的方法：字幕生成和可学习界面，分别用于将显式和隐式观察反馈纳入 LLM。实验表明，这两种方法在解决某些任务时都比较吃力，这表明长视距操作任务对于当前流行的模型来说仍然具有挑战性。作者希望所提出的公共基准和基线能帮助社区开发出更好的模型，用于长视距桌面操纵任务。

3. FurnitureBench: Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation --- [RSS 2023](https://arxiv.org/abs/2305.12821) --- [Code](https://clvrai.com/furniture-bench/)
	>Motivation：强化学习（RL）、模仿学习（IL）和任务与运动规划（TAMP）在各种机器人操纵任务中都表现出了令人印象深刻的性能。然而，这些方法仅限于学习当前真实世界操纵基准中的简单行为，如推或拾放。为了让自主机器人能够做出更复杂、更长远的行为，作者提出将重点放在现实世界的家具组装上，这是一项复杂、long-horizon 机器人操纵任务，需要解决目前机器人操纵面临的许多难题。
	Solution: 他们提出的 FurnitureBench 是一个可重现的真实世界家具组装benchmark，旨在提供一个低门槛且易于重现的平台，以便世界各地的研究人员能够可靠地测试他们的算法，并将其与之前的工作进行比较。为了便于使用，作者提供了 200 多个小时的预收集数据（5000 多次演示）、3D 可打印家具模型、机器人环境设置指南和系统化任务初始化。此外，还提供了 FurnitureBench 的快速逼真模拟器 FurnitureSim。对离线 RL 算法和 IL 算法在装配任务上的性能进行了基准测试，并证明需要改进这些算法才能解决现实世界中的任务，这为未来的研究提供了大量机会。
Key words: Long-horizon task, isaac sim, RL，IL, benchmark

4. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion --- [RSS 2023](https://arxiv.org/abs/2303.04137) --- [Project web ](https://diffusion-policy.cs.columbia.edu/) --- [Code](https://github.com/real-stanford/diffusion_policy)
	>本文介绍了扩散策略，这是一种通过将机器人的视觉运动策略表示为条件去噪扩散过程来生成机器人行为的新方法。我们对来自 4 个不同机器人操作基准的 12 个不同任务的扩散策略进行了基准测试，发现它始终优于现有最先进的机器人学习方法，平均提高了 46.9%。扩散策略学习动作分布得分函数的梯度，并在推理过程中通过一系列随机朗之万动力学步骤对该梯度场进行迭代优化。我们发现扩散公式在用于机器人策略时具有强大的优势，包括优雅地处理多模态动作分布、适用于高维动作空间以及表现出令人印象深刻的训练稳定性。为了充分释放扩散模型在物理机器人视觉运动策略学习中的潜力，本文提出了一系列关键技术贡献，包括结合后退地平线控制、视觉调节和时间序列扩散变压器。我们希望这项工作将有助于激发新一代政策学习技术，这些技术能够利用扩散模型强大的生成建模功能。
	Key words: Imitation Learning ,

5. Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation --- [Paper 2023.06](https://arxiv.org/abs/2306.17817) --- [Project web](https://act3d.github.io/) --- [Code](https://github.com/zhouxian/act3d-chained-diffuser)
	>3D 感知表示非常适合机器人操作，因为它们可以轻松编码遮挡并简化空间推理。许多操纵任务在末端执行器姿态预测中需要高空间精度，这通常需要高分辨率的 3D 特征网格，而处理起来的计算成本很高。因此，大多数操纵策略直接在 2D 中运行，而忽略了 3D 归纳偏差。在本文中，我们介绍了 Act3D，这是一种操作策略转换器，它使用 3D 特征场来表示机器人的工作空间，其自适应分辨率取决于手头的任务。该模型使用感测深度将 2D 预训练特征提升为 3D，并利用它们来计算采样 3D 点的特征。它以从粗到细的方式对 3D 点网格进行采样，使用相对位置注意力对其进行特征化，并选择下一轮点采样的聚焦位置。通过这种方式，它可以有效地计算高空间分辨率的 3D 动作图。 Act3D 在 RL-Bench（一个既定的操纵基准）中树立了新的最先进水平，在 74 个 RLBench 任务上，它比之前的 SOTA 2D 多视图策略实现了 10% 的绝对改进，并且在计算量减少 3 倍的情况下实现了 22% 的绝对改进优于之前的 SOTA 3D 政策。我们量化了相对空间注意力、大规模视觉语言预训练的 2D 主干以及烧蚀实验中从粗到细注意力的权重绑定的重要性。
	Key words: Imitation learning,


### 2022
1. Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation --- [CoRL 2022](https://arxiv.org/abs/2209.05451)  --- [Project website](https://peract.github.io/) --- [Code](https://github.com/peract/peract)
	>Transformer 凭借其可扩展大型数据集的能力，彻底改变了视觉和自然语言处理。但在机器人操作中，数据既有限又昂贵。通过正确的问题表述，我们还能从 Transformers 中受益吗？
	>\
我们使用 PerAct 来研究这个问题，PerAct 是一种用于多任务 6-DoF 操作的语言条件行为克隆代理。 PerAct 使用 Perceiver Transformer 对语言目标和 RGB-D 体素观察进行编码，并通过“检测下一个最佳体素动作”来输出离散动作。与操作 2D 图像的框架不同，体素化观察和动作空间为有效学习 6-DoF 策略提供了强大的结构先验。
	>\
通过这个公式，我们通过每个任务的几个演示来训练一个多任务 Transformer，用于 18 个 RLBench（有 249 个变体）和 7 个现实世界任务（有 18 个变体）。我们的结果表明，对于各种桌面任务，PerAct 的性能显着优于非结构化图像到动作代理（34 倍）和 3D ConvNet 基线（2.8 倍）。
	Keywords: RLBench, Imitation learning
