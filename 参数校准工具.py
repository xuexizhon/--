import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ======================
# 1. 技术风险指数计算类
# ======================
class TechnicalRiskIndex:
    """技术风险指数计算器"""

    def __init__(self, n_devices=10, random_seed=42):
        """
        初始化技术风险指数计算器

        Parameters:
        -----------
        n_devices : int, 设备数量
        random_seed : int, 随机种子
        """
        np.random.seed(random_seed)
        self.n_devices = n_devices

        # 生成模拟数据
        self.devices = [f"设备{i + 1:02d}" for i in range(n_devices)]
        self.alpha = np.random.uniform(0.01, 0.2, n_devices)  # 权重系数
        self.device_scores = np.random.uniform(0.1, 1.5, n_devices)  # 设备风险得分

        # 归一化权重
        self.alpha = self.alpha / self.alpha.sum()

        # 安全阈值
        self.safety_threshold = 0.62

    def calculate_R_tech(self):
        """计算技术风险指数"""
        R_tech = np.sum(self.alpha * self.device_scores)
        return R_tech

    def check_safety(self):
        """检查是否安全"""
        R_tech = self.calculate_R_tech()
        is_safe = R_tech <= self.safety_threshold
        return is_safe, R_tech

    def get_device_risk_contributions(self):
        """获取各设备风险贡献度"""
        contributions = self.alpha * self.device_scores
        total = contributions.sum()
        percentages = (contributions / total * 100) if total > 0 else np.zeros_like(contributions)

        df = pd.DataFrame({
            '设备名称': self.devices,
            '权重系数α': self.alpha.round(4),
            '设备风险得分': self.device_scores.round(4),
            '风险贡献值': contributions.round(4),
            '贡献百分比%': percentages.round(2)
        }).sort_values('风险贡献值', ascending=False)

        return df

    def generate_report(self):
        """生成技术风险报告"""
        is_safe, R_tech = self.check_safety()

        print("=" * 60)
        print("技术风险指数分析报告")
        print("=" * 60)

        print(f"\n📊 计算公式：")
        print(f"  R_tech = Σ(α_i × Device_i)")

        print(f"\n🔢 计算参数：")
        print(f"  设备数量：{self.n_devices}")
        print(f"  权重系数：α = [{', '.join([f'{a:.3f}' for a in self.alpha[:3]])}, ...]")

        print(f"\n📈 计算结果：")
        print(f"  技术风险指数：R_tech = {R_tech:.4f}")
        print(f"  安全阈值：{self.safety_threshold}")
        print(f"  安全状态：{'✅ 安全' if is_safe else '❌ 不安全'}")

        print(f"\n📋 风险评估：")
        if is_safe:
            print(f"  - 技术风险控制在安全范围内")
            print(f"  - 可正常部署系统")
        else:
            print(f"  - 技术风险超出安全阈值 {(R_tech - self.safety_threshold):.4f}")
            print(f"  - 建议优化高风险设备配置")

        return is_safe, R_tech


# ======================
# 2. 伦理容忍阈值计算类
# ======================
class EthicalToleranceThreshold:
    """伦理容忍阈值计算器"""

    def __init__(self, n_regions=10, random_seed=42):
        """
        初始化伦理容忍阈值计算器

        Parameters:
        -----------
        n_regions : int, 区域数量
        random_seed : int, 随机种子
        """
        np.random.seed(random_seed)
        self.n_regions = n_regions

        # 生成模拟数据
        self.regions = [f"区域{i + 1:02d}" for i in range(n_regions)]

        # 城乡差异系数 (0-1之间，1表示差异最大)
        self.urban_rural_gap = np.random.uniform(0.1, 1.0, n_regions)

        # 基础信任度
        self.base_trust = 0.68
        self.gap_coefficient = 0.05

    def calculate_Trust_min(self, urban_rural_gap=None):
        """计算伦理容忍阈值"""
        if urban_rural_gap is None:
            urban_rural_gap = self.urban_rural_gap

        Trust_min = self.base_trust - self.gap_coefficient * urban_rural_gap
        return Trust_min

    def get_region_analysis(self):
        """获取各区域分析"""
        trust_min = self.calculate_Trust_min()

        df = pd.DataFrame({
            '区域名称': self.regions,
            '城乡差异系数': self.urban_rural_gap.round(4),
            '伦理容忍阈值': trust_min.round(4),
            '风险等级': pd.cut(trust_min,
                               bins=[0, 0.5, 0.6, 0.68],
                               labels=['高风险', '中风险', '低风险'])
        }).sort_values('伦理容忍阈值')

        return df

    def calculate_sensitivity(self, gap_range=np.arange(0, 1.1, 0.1)):
        """敏感性分析：城乡差异对伦理容忍阈值的影响"""
        sensitivities = []
        for gap in gap_range:
            trust = self.base_trust - self.gap_coefficient * gap
            sensitivities.append((gap, trust))

        return pd.DataFrame(sensitivities, columns=['城乡差异系数', '伦理容忍阈值'])

    def generate_report(self):
        """生成伦理容忍阈值报告"""
        trust_min_values = self.calculate_Trust_min()
        min_trust = trust_min_values.min()
        max_trust = trust_min_values.max()
        avg_trust = trust_min_values.mean()

        print("=" * 60)
        print("伦理容忍阈值分析报告")
        print("=" * 60)

        print(f"\n📊 计算公式：")
        print(f"  Trust_min = 0.68 - 0.05 × UrbanRuralGap")

        print(f"\n🔢 计算参数：")
        print(f"  区域数量：{self.n_regions}")
        print(f"  基础信任度：0.68")
        print(f"  城乡差异系数：0.05")

        print(f"\n📈 计算结果：")
        print(f"  伦理容忍阈值范围：[{min_trust:.4f}, {max_trust:.4f}]")
        print(f"  平均伦理容忍阈值：{avg_trust:.4f}")

        print(f"\n📋 风险评估：")
        low_risk = (trust_min_values > 0.6).sum()
        medium_risk = ((trust_min_values >= 0.5) & (trust_min_values <= 0.6)).sum()
        high_risk = (trust_min_values < 0.5).sum()

        print(f"  - 低风险区域：{low_risk}个（阈值>0.6）")
        print(f"  - 中风险区域：{medium_risk}个（0.5≤阈值≤0.6）")
        print(f"  - 高风险区域：{high_risk}个（阈值<0.5）")

        print(f"\n💡 建议措施：")
        if high_risk > 0:
            print(f"  - {high_risk}个高风险区域需加强伦理审查")
            print(f"  - 建议启动动态脱敏协议")

        return trust_min_values


# ======================
# 3. 小学段模块风险检测类
# ======================
class ElementarySchoolRiskDetector:
    """小学段模块风险检测器"""

    def __init__(self):
        """初始化检测器"""
        self.module_name = "小学段行为分析模块"
        self.actual_weight = 0.75
        self.safety_limit = 0.60

    def calculate_exceedance(self):
        """计算超标幅度"""
        exceedance = self.actual_weight - self.safety_limit
        exceedance_percentage = (exceedance / self.safety_limit) * 100
        return exceedance, exceedance_percentage

    def generate_report(self):
        """生成风险检测报告"""
        exceedance, exceedance_percentage = self.calculate_exceedance()

        print("=" * 60)
        print("小学段模块风险检测报告")
        print("=" * 60)

        print(f"\n📊 检测模块：{self.module_name}")
        print(f"\n📈 检测结果：")
        print(f"  实际权重：{self.actual_weight:.2f}")
        print(f"  安全上限：{self.safety_limit:.2f}")
        print(f"  超标幅度：{exceedance:.2f}（{exceedance_percentage:.1f}%）")

        print(f"\n⚠️ 风险评估：")
        if exceedance_percentage > 20:
            print(f"  ❌ 严重超出安全阈值（超过20%）")
            print(f"  ⚠️ 存在高隐私风险")
        elif exceedance_percentage > 10:
            print(f"  ⚠️ 超出安全阈值（10%-20%）")
            print(f"  ⚠️ 存在中等隐私风险")
        else:
            print(f"  ✅ 在安全范围内")

        print(f"\n🚨 紧急处置建议：")
        print(f"  1. 🔴 立即启用动态脱敏协议")
        print(f"  2. 🟡 启动三级应急响应机制")
        print(f"  3. 🟠 组织伦理审查委员会紧急会议")
        print(f"  4. 🔴 暂停相关数据采集24小时")

        return exceedance_percentage


# ======================
# 4. 可视化模块
# ======================
class VisualizationTools:
    """可视化工具"""

    @staticmethod
    def plot_technical_risk_breakdown(risk_calculator):
        """绘制技术风险分解图"""
        df = risk_calculator.get_device_risk_contributions()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左侧：风险贡献条形图
        colors = ['#ff6b6b' if x > df['风险贡献值'].mean() else '#4ecdc4' for x in df['风险贡献值']]
        ax1.barh(df['设备名称'][:10], df['风险贡献值'][:10], color=colors)
        ax1.set_xlabel('风险贡献值', fontsize=12)
        ax1.set_title('各设备风险贡献度排名（Top 10）', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # 右侧：风险分布饼图
        top5 = df.head(5)
        labels = top5['设备名称'].tolist()
        sizes = top5['贡献百分比%'].tolist()
        colors = ['#ff6b6b', '#ffa726', '#ffca28', '#4ecdc4', '#45b7d1']

        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('前5大风险设备分布', fontsize=14, fontweight='bold')

        plt.suptitle('技术风险指数分解分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_ethical_threshold_analysis(ethic_calculator):
        """绘制伦理容忍阈值分析图"""
        df = ethic_calculator.get_region_analysis()
        sensitivity_df = ethic_calculator.calculate_sensitivity()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左侧：各区域伦理容忍阈值
        colors = {'高风险': '#ff6b6b', '中风险': '#ffca28', '低风险': '#4ecdc4'}
        bar_colors = [colors[risk] for risk in df['风险等级']]

        bars = ax1.barh(df['区域名称'], df['伦理容忍阈值'], color=bar_colors)
        ax1.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='高风险阈值')
        ax1.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='中风险阈值')
        ax1.set_xlabel('伦理容忍阈值', fontsize=12)
        ax1.set_title('各区域伦理容忍阈值分布', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)

        # 右侧：敏感性分析
        ax2.plot(sensitivity_df['城乡差异系数'], sensitivity_df['伦理容忍阈值'],
                 'o-', linewidth=2, markersize=8, color='#7e57c2')
        ax2.fill_between(sensitivity_df['城乡差异系数'], sensitivity_df['伦理容忍阈值'],
                         0.5, where=(sensitivity_df['伦理容忍阈值'] < 0.5),
                         color='#ff6b6b', alpha=0.3, label='高风险区')
        ax2.fill_between(sensitivity_df['城乡差异系数'], sensitivity_df['伦理容忍阈值'],
                         0.6, where=(sensitivity_df['伦理容忍阈值'] >= 0.5) &
                                    (sensitivity_df['伦理容忍阈值'] < 0.6),
                         color='#ffca28', alpha=0.3, label='中风险区')
        ax2.set_xlabel('城乡差异系数', fontsize=12)
        ax2.set_ylabel('伦理容忍阈值', fontsize=12)
        ax2.set_title('城乡差异对伦理容忍阈值的影响', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.suptitle('伦理容忍阈值分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_school_module_risk(detector):
        """绘制小学段模块风险图"""
        exceedance, exceedance_percentage = detector.calculate_exceedance()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左侧：权重对比
        categories = ['安全上限', '实际权重']
        values = [detector.safety_limit, detector.actual_weight]
        colors = ['#4CAF50', '#FF5722']

        bars = ax1.bar(categories, values, color=colors, width=0.6)
        ax1.set_ylabel('权重值', fontsize=12)
        ax1.set_title('小学段行为分析模块权重对比', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(values) * 1.2)

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{value:.2f}', ha='center', va='bottom', fontsize=12)

        # 添加超标标注
        ax1.annotate(f'超标 {exceedance:.2f}',
                     xy=(1, detector.actual_weight),
                     xytext=(1.3, detector.actual_weight),
                     arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
                     fontsize=12, color='red', fontweight='bold')

        # 右侧：风险等级图
        risk_levels = ['低风险', '中风险', '高风险']
        thresholds = [0.6, 0.7, 0.75]
        colors = ['#4CAF50', '#FFC107', '#F44336']

        for i, (level, threshold, color) in enumerate(zip(risk_levels, thresholds, colors)):
            ax2.barh(level, threshold, color=color, alpha=0.6, height=0.4)
            ax2.text(threshold, i, f' {threshold}', va='center', fontsize=11)

        # 标记实际位置
        ax2.axvline(x=detector.actual_weight, color='black', linestyle='--', linewidth=2)
        ax2.text(detector.actual_weight, 2.5, f' 实际值: {detector.actual_weight}',
                 va='center', fontsize=12, fontweight='bold', color='black')

        ax2.set_xlabel('权重值', fontsize=12)
        ax2.set_title('风险等级划分', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 0.8)

        plt.suptitle('小学段模块高风险预警分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# ======================
# 5. 主程序：完整报告生成
# ======================
def generate_comprehensive_report():
    """生成综合报告"""
    print("=" * 80)
    print("学校智能识别隐私治理系统校准报告".center(80))
    print("=" * 80)
    print(f"生成时间：{pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    print(f"版本：1.0 | 分析工具：Python")
    print("-" * 80)

    # 1. 技术风险指数分析
    print("\n" + "🔧 第一部分：技术风险评估".center(80, '-'))
    tech_risk = TechnicalRiskIndex(n_devices=15)
    is_safe_tech, R_tech = tech_risk.generate_report()

    # 2. 伦理容忍阈值分析
    print("\n" + "⚖️ 第二部分：伦理容忍阈值分析".center(80, '-'))
    ethic_risk = EthicalToleranceThreshold(n_regions=12)
    trust_values = ethic_risk.generate_report()

    # 3. 小学段模块风险检测
    print("\n" + "📚 第三部分：特定模块风险检测".center(80, '-'))
    school_detector = ElementarySchoolRiskDetector()
    exceedance_percentage = school_detector.generate_report()

    # 4. 综合结论
    print("\n" + "📋 第四部分：综合校准结论".center(80, '-'))

    # 构建结论数据框
    conclusion_data = {
        '评估维度': ['技术风险指数', '伦理容忍阈值（平均）', '小学段模块权重'],
        '实际值': [R_tech, trust_values.mean(), school_detector.actual_weight],
        '安全阈值': [tech_risk.safety_threshold, 0.6, school_detector.safety_limit],
        '状态': [
            '✅ 安全' if is_safe_tech else '❌ 不安全',
            '✅ 安全' if trust_values.mean() > 0.6 else '❌ 不安全',
            '❌ 高风险' if exceedance_percentage > 0 else '✅ 安全'
        ]
    }

    df_conclusion = pd.DataFrame(conclusion_data)
    print("\n综合评估结果：")
    print(df_conclusion.to_string(index=False))

    print("\n📊 总体风险评级：")
    unsafe_count = sum([1 for status in conclusion_data['状态'] if '❌' in status])

    if unsafe_count == 0:
        print("  ✅ 绿色（低风险）：所有指标均在安全范围内")
        overall_risk = "低风险"
    elif unsafe_count == 1:
        print("  🟡 黄色（中风险）：1个指标超出安全范围")
        print("  ⚠️ 建议：重点关注高风险模块，采取相应措施")
        overall_risk = "中风险"
    else:
        print("  🔴 红色（高风险）：2个或更多指标超出安全范围")
        print("  🚨 紧急建议：立即启动应急预案，全面审查系统配置")
        overall_risk = "高风险"

    print("\n" + "💡 校准建议总结".center(80, '-'))
    print("1. 技术维度：定期监测设备风险，优化高风险设备配置")
    print("2. 制度维度：建立动态阈值调整机制，适应不同场景")
    print("3. 管理维度：加强城乡差异区域的伦理审查")
    print("4. 应急措施：对高风险模块立即启用动态脱敏协议")

    # 5. 生成可视化
    print("\n" + "📈 可视化分析".center(80, '-'))
    print("正在生成可视化图表...")

    viz = VisualizationTools()

    # 技术风险分解图
    viz.plot_technical_risk_breakdown(tech_risk)

    # 伦理容忍阈值分析图
    viz.plot_ethical_threshold_analysis(ethic_risk)

    # 小学段模块风险图
    viz.plot_school_module_risk(school_detector)

    return {
        '技术风险指数': R_tech,
        '伦理容忍阈值': trust_values.mean(),
        '小学段模块超标幅度': exceedance_percentage,
        '总体风险评级': overall_risk,
        '结论数据框': df_conclusion
    }


# ======================
# 6. 额外功能：批量计算和导出
# ======================
def batch_analysis_and_export(n_simulations=100):
    """批量模拟分析并导出结果"""
    print("正在进行批量模拟分析...")

    results = {
        '技术风险指数': [],
        '技术风险状态': [],
        '伦理容忍阈值平均': [],
        '小学段模块超标%': [],
        '总体风险评级': []
    }

    for i in range(n_simulations):
        # 每次使用不同的随机种子
        tech_risk = TechnicalRiskIndex(n_devices=15, random_seed=i)
        ethic_risk = EthicalToleranceThreshold(n_regions=12, random_seed=i)
        school_detector = ElementarySchoolRiskDetector()

        # 计算结果
        is_safe_tech, R_tech = tech_risk.check_safety()
        trust_values = ethic_risk.calculate_Trust_min()
        exceedance_percentage = school_detector.calculate_exceedance()[1]

        # 确定总体风险
        unsafe_count = sum([
            0 if is_safe_tech else 1,
            1 if trust_values.mean() <= 0.6 else 0,
            1 if exceedance_percentage > 0 else 0
        ])

        if unsafe_count == 0:
            overall_risk = "低风险"
        elif unsafe_count == 1:
            overall_risk = "中风险"
        else:
            overall_risk = "高风险"

        # 存储结果
        results['技术风险指数'].append(R_tech)
        results['技术风险状态'].append('安全' if is_safe_tech else '不安全')
        results['伦理容忍阈值平均'].append(trust_values.mean())
        results['小学段模块超标%'].append(exceedance_percentage)
        results['总体风险评级'].append(overall_risk)

    # 创建数据框
    df_results = pd.DataFrame(results)

    # 统计信息
    print("\n批量模拟分析结果（100次模拟）：")
    print("=" * 60)
    print(f"技术风险安全比例：{(df_results['技术风险状态'] == '安全').sum() / n_simulations * 100:.1f}%")
    print(f"伦理容忍阈值<0.6比例：{(df_results['伦理容忍阈值平均'] <= 0.6).sum() / n_simulations * 100:.1f}%")
    print(f"小学段模块超标比例：{(df_results['小学段模块超标%'] > 0).sum() / n_simulations * 100:.1f}%")

    risk_counts = df_results['总体风险评级'].value_counts()
    print("\n总体风险分布：")
    for risk, count in risk_counts.items():
        print(f"  {risk}: {count}次 ({count / n_simulations * 100:.1f}%)")

    # 导出到Excel
    output_file = "系统校准分析结果.xlsx"

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 详细结果表
        df_results.to_excel(writer, sheet_name='详细结果', index=False)

        # 汇总统计表
        summary_stats = {
            '指标': ['技术风险指数', '伦理容忍阈值', '小学段模块超标%'],
            '平均值': [
                df_results['技术风险指数'].mean(),
                df_results['伦理容忍阈值平均'].mean(),
                df_results['小学段模块超标%'].mean()
            ],
            '标准差': [
                df_results['技术风险指数'].std(),
                df_results['伦理容忍阈值平均'].std(),
                df_results['小学段模块超标%'].std()
            ],
            '最小值': [
                df_results['技术风险指数'].min(),
                df_results['伦理容忍阈值平均'].min(),
                df_results['小学段模块超标%'].min()
            ],
            '最大值': [
                df_results['技术风险指数'].max(),
                df_results['伦理容忍阈值平均'].max(),
                df_results['小学段模块超标%'].max()
            ]
        }

        df_summary = pd.DataFrame(summary_stats)
        df_summary.to_excel(writer, sheet_name='统计摘要', index=False)

        # 风险分布表
        df_risk_dist = pd.DataFrame({
            '风险等级': risk_counts.index,
            '出现次数': risk_counts.values,
            '占比%': (risk_counts.values / n_simulations * 100).round(1)
        })
        df_risk_dist.to_excel(writer, sheet_name='风险分布', index=False)

    print(f"\n✅ 分析结果已导出到：{output_file}")

    return df_results


# ======================
# 7. 运行示例
# ======================
if __name__ == "__main__":
    print("学校智能识别隐私治理系统 - 公式计算与风险评估工具")
    print("版本：1.0 | 作者：张越")
    print("-" * 60)

    while True:
        print("\n请选择操作：")
        print("1. 生成单次校准报告")
        print("2. 批量模拟分析（100次）")
        print("3. 单独计算技术风险指数")
        print("4. 单独计算伦理容忍阈值")
        print("5. 退出程序")

        choice = input("\n请输入选项 (1-5): ").strip()

        if choice == '1':
            # 生成综合报告
            print("\n正在生成综合校准报告...")
            results = generate_comprehensive_report()

            # 询问是否保存
            save_choice = input("\n是否保存报告到文件？(y/n): ").lower()
            if save_choice == 'y':
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                filename = f"校准报告_{timestamp}.txt"

                import sys

                original_stdout = sys.stdout
                with open(filename, 'w', encoding='utf-8') as f:
                    sys.stdout = f
                    generate_comprehensive_report()
                    sys.stdout = original_stdout

                print(f"✅ 报告已保存到：{filename}")

        elif choice == '2':
            # 批量分析
            print("\n开始批量模拟分析...")
            batch_results = batch_analysis_and_export(n_simulations=100)

        elif choice == '3':
            # 单独计算技术风险
            print("\n单独计算技术风险指数")
            n_devices = int(input("请输入设备数量 (默认15): ") or 15)
            tech_risk = TechnicalRiskIndex(n_devices=n_devices)
            is_safe, R_tech = tech_risk.generate_report()

            # 显示设备贡献度
            show_details = input("\n是否显示设备风险贡献详情？(y/n): ").lower()
            if show_details == 'y':
                df_details = tech_risk.get_device_risk_contributions()
                print("\n设备风险贡献详情：")
                print(df_details.to_string(index=False))

        elif choice == '4':
            # 单独计算伦理容忍阈值
            print("\n单独计算伦理容忍阈值")
            n_regions = int(input("请输入区域数量 (默认10): ") or 10)
            ethic_risk = EthicalToleranceThreshold(n_regions=n_regions)
            trust_values = ethic_risk.generate_report()

            # 显示区域详情
            show_details = input("\n是否显示区域详情？(y/n): ").lower()
            if show_details == 'y':
                df_regions = ethic_risk.get_region_analysis()
                print("\n各区域伦理容忍阈值详情：")
                print(df_regions.to_string(index=False))

        elif choice == '5':
            print("\n感谢使用，程序退出！")
            break

        else:
            print("❌ 无效选项，请重新选择")