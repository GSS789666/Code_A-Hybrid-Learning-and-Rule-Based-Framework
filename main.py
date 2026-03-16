import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from typing import Dict, List, Tuple
import pandas as pd
import re

# 设置SCI学术风格绘图
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# ============================
# 数据集路径配置
# ============================
# 请根据你的实际文件路径修改这里
PROBLEM_DIR = "problem_instances_small"  # 问题实例文件夹路径
SOLUTION_DIR = "solutions_small"  # 解决方案文件夹路径


class EarlyStopping:
    """早停类"""

    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # 修复：使用 np.inf 而不是 np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss


class MRTADataset(Dataset):
    """MRTA数据集类"""

    def __init__(self, problem_dir: str, solution_dir: str, max_instances: int = 1000):
        self.problem_dir = Path(problem_dir)
        self.solution_dir = Path(solution_dir)
        self.instances = []

        # 检查路径是否存在
        if not self.problem_dir.exists():
            raise FileNotFoundError(f"问题实例文件夹不存在: {self.problem_dir}")
        if not self.solution_dir.exists():
            raise FileNotFoundError(f"解决方案文件夹不存在: {self.solution_dir}")

        # 加载数据 - 根据实际文件命名规则匹配
        problem_files = sorted(self.problem_dir.glob("*.json"))[:max_instances]

        print(f"找到 {len(problem_files)} 个问题文件")

        for prob_file in problem_files:
            # 根据问题文件名找到对应的解决方案文件
            sol_file = self._find_solution_file(prob_file)

            if not sol_file or not sol_file.exists():
                print(f"警告: 找不到对应的解决方案文件: {prob_file.name}")
                continue

            try:
                with open(prob_file, 'r') as f:
                    problem = json.load(f)
                with open(sol_file, 'r') as f:
                    solution = json.load(f)

                self.instances.append((problem, solution))
            except Exception as e:
                print(f"加载文件时出错 {prob_file}: {e}")
                continue

    def _find_solution_file(self, problem_file: Path) -> Path:
        """根据问题文件名找到对应的解决方案文件"""
        filename = problem_file.name

        # 尝试不同的文件名匹配模式
        patterns = [
            # 模式1: problem_instance_1p_000000.json -> optimal_schedule_1p_000000.json
            filename.replace("problem_instance", "optimal_schedule"),
            # 模式2: instance_000000.json -> instance_000000.json (相同文件名)
            filename,
            # 模式3: 其他可能的命名模式...
        ]

        for pattern in patterns:
            sol_file = self.solution_dir / pattern
            if sol_file.exists():
                return sol_file

        return None

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        problem, solution = self.instances[idx]

        # 特征工程
        features = self._extract_features(problem)

        # 标签提取 - 任务分配矩阵 (tasks x robots)
        labels = self._extract_labels(problem, solution)

        return torch.FloatTensor(features), torch.FloatTensor(labels)

    def _extract_features(self, problem: Dict) -> np.ndarray:
        """提取问题特征"""
        features = []

        # 1. 机器人能力特征
        Q = np.array(problem['Q'])
        features.extend(Q.flatten())

        # 2. 任务需求特征 (排除虚拟任务)
        R = np.array(problem['R'])[1:-1]  # 排除第一个和最后一个虚拟任务
        features.extend(R.flatten())

        # 3. 执行时间特征
        T_e = np.array(problem['T_e'])[1:-1]
        features.extend(T_e)

        # 4. 位置特征
        locations = np.array(problem['task_locations'])[1:-1]
        features.extend(locations.flatten())

        # 5. 优先约束特征 (one-hot编码)
        n_tasks = len(R)
        prec_matrix = np.zeros((n_tasks, n_tasks))
        for constraint in problem['precedence_constraints']:
            i, j = constraint
            if 1 <= i <= n_tasks and 1 <= j <= n_tasks:
                prec_matrix[i - 1, j - 1] = 1
        features.extend(prec_matrix.flatten())

        return np.array(features)

    def _extract_labels(self, problem: Dict, solution: Dict) -> np.ndarray:
        """提取分配标签"""
        n_tasks = len(problem['R']) - 2  # 实际任务数
        n_robots = len(problem['Q'])

        # 创建分配矩阵
        assignment_matrix = np.zeros((n_tasks, n_robots))

        for robot_id, schedule in solution['robot_schedules'].items():
            robot_id = int(robot_id)
            for task_info in schedule:
                task_id = task_info['task'] - 1  # 转换为0-based索引
                if 0 <= task_id < n_tasks:
                    assignment_matrix[task_id, robot_id] = 1

        return assignment_matrix.flatten()


class MRTAConstraintLoss(nn.Module):
    """带约束的损失函数"""

    def __init__(self, alpha=0.1, beta=0.1):
        super().__init__()
        self.alpha = alpha  # 能力约束权重
        self.beta = beta  # 时间约束权重
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets, problems):
        # 基础二元交叉熵损失
        base_loss = self.bce_loss(predictions, targets)

        # 能力约束违反惩罚
        capability_violation = self._compute_capability_violation(predictions, problems)

        # 时间约束违反惩罚
        time_violation = self._compute_time_violation(predictions, problems)

        total_loss = base_loss + self.alpha * capability_violation + self.beta * time_violation

        return total_loss, base_loss, capability_violation, time_violation

    def _compute_capability_violation(self, predictions, problems):
        """计算能力约束违反程度"""
        batch_size = predictions.shape[0]
        n_tasks = 8  # 实际任务数
        n_robots = 3

        total_violation = 0

        for i in range(batch_size):
            # 重塑预测为任务x机器人矩阵
            pred_matrix = predictions[i].view(n_tasks, n_robots)

            # 获取问题实例
            problem = problems[i]
            R = torch.tensor(problem['R'][1:-1], dtype=torch.float32)  # 任务需求
            Q = torch.tensor(problem['Q'], dtype=torch.float32)  # 机器人能力

            # 检查每个任务的能力满足情况
            for task_idx in range(n_tasks):
                task_requirements = R[task_idx]
                assigned_robots = pred_matrix[task_idx]

                if assigned_robots.sum() > 0:  # 如果有机器人分配
                    # 计算分配机器人的总能力
                    total_capability = torch.zeros(3)
                    for robot_idx in range(n_robots):
                        if assigned_robots[robot_idx] > 0.5:  # 使用0.5作为阈值
                            total_capability += Q[robot_idx]

                    # 计算能力不足的部分
                    shortage = torch.clamp(task_requirements - total_capability, min=0)
                    total_violation += shortage.sum()

        return total_violation / batch_size if batch_size > 0 else torch.tensor(0.0)

    def _compute_time_violation(self, predictions, problems):
        """简化版时间约束违反计算"""
        return torch.tensor(0.0)


class RuleBasedPostProcessor:
    """基于规则的后处理器"""

    def __init__(self):
        pass

    def process(self, predictions: torch.Tensor, problems: List[Dict]) -> torch.Tensor:
        """对神经网络输出进行后处理"""
        batch_size = predictions.shape[0]
        n_tasks = 8
        n_robots = 3

        processed_predictions = predictions.clone()

        for i in range(batch_size):
            pred_matrix = predictions[i].view(n_tasks, n_robots)
            problem = problems[i]

            # 应用能力匹配规则
            pred_matrix = self._apply_capability_rules(pred_matrix, problem)

            # 应用任务分配平衡规则
            pred_matrix = self._apply_balancing_rules(pred_matrix)

            processed_predictions[i] = pred_matrix.flatten()

        return processed_predictions

    def _apply_capability_rules(self, pred_matrix: torch.Tensor, problem: Dict) -> torch.Tensor:
        """应用能力匹配规则"""
        R = torch.tensor(problem['R'][1:-1], dtype=torch.float32)
        Q = torch.tensor(problem['Q'], dtype=torch.float32)
        n_tasks, n_robots = pred_matrix.shape

        for task_idx in range(n_tasks):
            task_req = R[task_idx]
            current_assignment = pred_matrix[task_idx]

            # 检查当前分配是否满足能力需求
            total_capability = torch.zeros(3)
            for robot_idx in range(n_robots):
                if current_assignment[robot_idx] > 0.5:
                    total_capability += Q[robot_idx]

            # 如果不满足需求，调整分配
            if (total_capability < task_req).any():
                # 找到能够提供缺失能力的机器人
                for robot_idx in range(n_robots):
                    if current_assignment[robot_idx] <= 0.5:  # 当前未分配
                        robot_capability = Q[robot_idx]
                        # 如果这个机器人能提供缺失的能力
                        missing_capabilities = task_req - total_capability
                        if (robot_capability * missing_capabilities > 0).any():
                            pred_matrix[task_idx, robot_idx] = 1.0
                            total_capability += robot_capability

            # 移除多余分配（可选）
            # 这里可以添加逻辑来移除不必要的机器人分配

        return pred_matrix

    def _apply_balancing_rules(self, pred_matrix: torch.Tensor) -> torch.Tensor:
        """应用负载均衡规则"""
        return pred_matrix  # 简化实现


class MRTANetwork(nn.Module):
    """MRTA预测网络"""

    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int] = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MRTATrainer:
    """MRTA模型训练器"""

    def __init__(self, model, use_constraint_loss=False, use_post_processing=False):
        self.model = model
        self.use_constraint_loss = use_constraint_loss
        self.use_post_processing = use_post_processing

        if use_constraint_loss:
            self.criterion = MRTAConstraintLoss(alpha=0.1, beta=0.1)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.8)

        if use_post_processing:
            self.post_processor = RuleBasedPostProcessor()

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # 添加早停功能
        self.early_stopping = EarlyStopping(patience=25, verbose=True)
        self.best_model_state = None

    def train_epoch(self, dataloader, problems):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()

            outputs = self.model(features)

            if self.use_constraint_loss:
                loss, base_loss, cap_violation, time_violation = self.criterion(
                    outputs, targets, [problems[i] for i in range(features.shape[0])]
                )
            else:
                loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 计算准确率
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.numel()

        accuracy = correct / total if total > 0 else 0
        return total_loss / len(dataloader), accuracy

    def validate(self, dataloader, problems):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, targets in dataloader:
                outputs = self.model(features)

                # 应用后处理（如果启用）
                if self.use_post_processing:
                    outputs = self.post_processor.process(outputs,
                                                          [problems[i] for i in range(features.shape[0])])

                if self.use_constraint_loss:
                    loss, _, _, _ = self.criterion(
                        outputs, targets, [problems[i] for i in range(features.shape[0])]
                    )
                else:
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == targets).sum().item()
                total += targets.numel()

        accuracy = correct / total if total > 0 else 0
        return total_loss / len(dataloader), accuracy

    def train(self, train_loader, val_loader, train_problems, val_problems, epochs=100):
        print("开始训练...")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, train_problems)
            val_loss, val_acc = self.validate(val_loader, val_problems)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.scheduler.step()

            # 早停检查
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"早停在第 {epoch} 轮")
                break

            # 保存最佳模型
            if val_loss == self.early_stopping.val_loss_min:
                self.best_model_state = self.model.state_dict().copy()

            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # 获取最终训练准确率
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0

        return final_train_acc, val_acc


def evaluate_feasibility(predictions, problems, solutions):
    """评估解的可行性"""
    n_instances = len(problems)
    feasibility_scores = []
    conflict_counts = []
    replanning_counts = []

    for i in range(n_instances):
        pred_matrix = predictions[i].view(8, 3)  # 8个任务，3个机器人
        problem = problems[i]
        solution = solutions[i]

        # 可行性检查
        feasibility_score = check_solution_feasibility(pred_matrix, problem, solution)
        feasibility_scores.append(feasibility_score)

        # 冲突计数
        conflict_count = count_conflicts(pred_matrix, problem)
        conflict_counts.append(conflict_count)

        # 重规划次数估计
        replanning_count = estimate_replanning(pred_matrix, problem)
        replanning_counts.append(replanning_count)

    return {
        'feasibility_rate': np.mean(feasibility_scores) if feasibility_scores else 0,
        'avg_conflicts': np.mean(conflict_counts) if conflict_counts else 0,
        'avg_replanning': np.mean(replanning_counts) if replanning_counts else 0
    }


def check_solution_feasibility(pred_matrix, problem, solution):
    """检查解的可行性"""
    # 简化版可行性检查
    R = torch.tensor(problem['R'][1:-1], dtype=torch.float32)
    Q = torch.tensor(problem['Q'], dtype=torch.float32)

    feasible = 1

    # 检查能力匹配
    for task_idx in range(8):
        task_req = R[task_idx]
        assigned_robots = pred_matrix[task_idx]

        total_capability = torch.zeros(3)
        for robot_idx in range(3):
            if assigned_robots[robot_idx] > 0.5:
                total_capability += Q[robot_idx]

        if (total_capability < task_req).any():
            feasible = 0
            break

    return feasible


def count_conflicts(pred_matrix, problem):
    """计算冲突数量"""
    # 简化版冲突计数
    conflicts = 0

    # 检查机器人超载（同一时间分配过多任务）
    robot_loads = pred_matrix.sum(dim=0)
    conflicts += (robot_loads > 3).sum().item()  # 假设每个机器人最多同时处理3个任务

    return conflicts


def estimate_replanning(pred_matrix, problem):
    """估计重规划次数"""
    # 基于冲突数量和可行性进行估计
    feasibility = check_solution_feasibility(pred_matrix, problem, None)
    conflicts = count_conflicts(pred_matrix, problem)

    return max(0, conflicts + (1 - feasibility) * 2)


def run_ablation_study(problem_dir, solution_dir):
    """运行消融实验"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    print("加载数据...")
    try:
        dataset = MRTADataset(problem_dir, solution_dir, max_instances=500)
        print(f"成功加载 {len(dataset)} 个实例")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return {}

    if len(dataset) == 0:
        print("没有找到有效的数据实例")
        return {}

    # 分割数据集 - 训练、验证、测试
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取问题实例用于约束处理
    train_problems = [dataset.instances[i][0] for i in train_dataset.indices]
    val_problems = [dataset.instances[i][0] for i in val_dataset.indices]
    test_problems = [dataset.instances[i][0] for i in test_dataset.indices]
    test_solutions = [dataset.instances[i][1] for i in test_dataset.indices]

    # 模型配置
    input_size = len(dataset[0][0])
    output_size = len(dataset[0][1])

    print(f"输入特征维度: {input_size}, 输出维度: {output_size}")
    print(f"数据集分割: 训练 {train_size}, 验证 {val_size}, 测试 {test_size}")

    # 四种实验配置
    configurations = [
        {"name": "Baseline", "constraint_loss": False, "post_processing": False},
        {"name": "Constraint_Loss", "constraint_loss": True, "post_processing": False},
        {"name": "Post_Processing", "constraint_loss": False, "post_processing": True},
        {"name": "Combined", "constraint_loss": True, "post_processing": True}
    ]

    results = {}

    for config in configurations:
        print(f"\n训练配置: {config['name']}")

        # 创建模型和训练器
        model = MRTANetwork(input_size, output_size)
        trainer = MRTATrainer(
            model,
            use_constraint_loss=config['constraint_loss'],
            use_post_processing=config['post_processing']
        )

        # 训练模型
        start_time = time.time()
        final_train_acc, final_val_acc = trainer.train(train_loader, val_loader, train_problems, val_problems,
                                                       epochs=100)
        training_time = time.time() - start_time

        # 在测试集上评估模型
        test_loss, test_acc = trainer.validate(test_loader, test_problems)

        # 获取测试集预测结果进行可行性分析
        test_predictions = []
        with torch.no_grad():
            for features, _ in test_loader:
                outputs = model(features)
                if config['post_processing']:
                    outputs = trainer.post_processor.process(
                        outputs, [test_problems[i] for i in range(features.shape[0])]
                    )
                test_predictions.extend(torch.sigmoid(outputs) > 0.5)

        feasibility_results = evaluate_feasibility(test_predictions, test_problems, test_solutions)

        # 保存结果
        results[config['name']] = {
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'final_test_accuracy': test_acc,
            'final_val_loss': test_loss,
            'training_time': training_time,
            'feasibility_rate': feasibility_results['feasibility_rate'],
            'avg_conflicts': feasibility_results['avg_conflicts'],
            'avg_replanning': feasibility_results['avg_replanning'],
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_accuracies': trainer.val_accuracies,
            'stopped_epoch': len(trainer.train_losses)  # 记录实际训练的epoch数
        }

        print(f"配置 {config['name']} 结果:")
        print(f"  训练准确率: {final_train_acc:.4f}, 验证准确率: {final_val_acc:.4f}, 测试准确率: {test_acc:.4f}")
        print(f"  可行性率: {feasibility_results['feasibility_rate']:.4f}")
        print(f"  训练时间: {training_time:.2f}s, 训练轮数: {len(trainer.train_losses)}")

    return results


def visualize_results(results):
    """可视化结果"""
    if not results:
        print("没有结果可可视化")
        return pd.DataFrame()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 准备数据
    models = list(results.keys())

    # 1. 准确率和可行性率对比
    train_acc_data = [results[model]['final_train_accuracy'] for model in models]
    val_acc_data = [results[model]['final_val_accuracy'] for model in models]
    test_acc_data = [results[model]['final_test_accuracy'] for model in models]
    feasibility_data = [results[model]['feasibility_rate'] for model in models]

    x = np.arange(len(models))
    width = 0.2

    axes[0, 0].bar(x - 1.5 * width, train_acc_data, width, label='Training Accuracy', alpha=0.8, color='blue')
    axes[0, 0].bar(x - 0.5 * width, val_acc_data, width, label='Validation Accuracy', alpha=0.8, color='green')
    axes[0, 0].bar(x + 0.5 * width, test_acc_data, width, label='Test Accuracy', alpha=0.8, color='red')
    axes[0, 0].bar(x + 1.5 * width, feasibility_data, width, label='Feasibility Rate', alpha=0.8, color='orange')
    axes[0, 0].set_xlabel('Model Configuration')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Accuracy Metrics and Feasibility Rate')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 冲突和重规划次数
    conflict_data = [results[model]['avg_conflicts'] for model in models]
    replanning_data = [results[model]['avg_replanning'] for model in models]

    axes[0, 1].bar(x - width / 2, conflict_data, width, label='Conflicts', alpha=0.8, color='red')
    axes[0, 1].bar(x + width / 2, replanning_data, width, label='Replanning', alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Model Configuration')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Conflicts and Replanning Counts')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 训练时间
    time_data = [results[model]['training_time'] for model in models]
    axes[0, 2].bar(models, time_data, alpha=0.8, color='green')
    axes[0, 2].set_xlabel('Model Configuration')
    axes[0, 2].set_ylabel('Time (seconds)')
    axes[0, 2].set_title('Training Time Comparison')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 训练曲线 - 损失
    for model in models:
        axes[1, 0].plot(results[model]['train_losses'], label=f'{model} Train')
        axes[1, 0].plot(results[model]['val_losses'], label=f'{model} Val', linestyle='--')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training and Validation Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 训练曲线 - 准确率
    for model in models:
        axes[1, 1].plot(results[model]['train_accuracies'], label=f'{model} Train')
        axes[1, 1].plot(results[model]['val_accuracies'], label=f'{model} Val', linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Training and Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 综合性能雷达图
    metrics_radar = ['final_test_accuracy', 'feasibility_rate', 'avg_conflicts', 'avg_replanning']
    radar_data = {}
    for model in models:
        model_data = [
            results[model]['final_test_accuracy'],
            results[model]['feasibility_rate'],
            1 - min(results[model]['avg_conflicts'] / 10, 1),  # 归一化并反向
            1 - min(results[model]['avg_replanning'] / 10, 1)  # 归一化并反向
        ]
        radar_data[model] = model_data

    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    for model, data in radar_data.items():
        data += data[:1]  # 闭合数据
        axes[1, 2].plot(angles, data, 'o-', linewidth=2, label=model)
        axes[1, 2].fill(angles, data, alpha=0.1)

    axes[1, 2].set_xticks(angles[:-1])
    axes[1, 2].set_xticklabels(['Test Accuracy', 'Feasibility', 'Conflict\nReduction', 'Replanning\nReduction'])
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Comprehensive Performance Radar')
    axes[1, 2].legend(bbox_to_anchor=(1.1, 1.05))
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('ablation_study_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 创建结果表格
    results_df = pd.DataFrame({
        'Model': models,
        'Training Accuracy': [results[m]['final_train_accuracy'] for m in models],
        'Validation Accuracy': [results[m]['final_val_accuracy'] for m in models],
        'Test Accuracy': [results[m]['final_test_accuracy'] for m in models],
        'Feasibility Rate': [results[m]['feasibility_rate'] for m in models],
        'Average Conflicts': [results[m]['avg_conflicts'] for m in models],
        'Average Replanning': [results[m]['avg_replanning'] for m in models],
        'Training Time (s)': [results[m]['training_time'] for m in models],
        'Training Epochs': [results[m]['stopped_epoch'] for m in models]
    })

    print("\n" + "=" * 80)
    print("消融实验结果汇总")
    print("=" * 80)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    # 运行消融实验
    print("开始MRTA消融实验...")

    # 使用配置的路径
    results = run_ablation_study(PROBLEM_DIR, SOLUTION_DIR)

    if results:
        # 可视化结果
        results_df = visualize_results(results)

        # 保存详细结果
        with open('ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\n实验完成！结果已保存到 'ablation_results.json' 和 'ablation_study_results.pdf'")
    else:
        print("实验失败，请检查数据集路径和文件格式")