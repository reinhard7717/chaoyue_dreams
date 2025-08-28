# 文件: services/chip_score_calculator.py
import pandas as pd

def calculate_chip_health_score(row: pd.Series) -> float:
    """
    【V1.1 健壮性修复版】根据一行包含所有基础和衍生指标的数据，计算最终的筹码健康分。
    这个函数被设计为在所有指标（包括斜率）都计算完毕后，通过 df.apply() 调用。
    
    Args:
        row (pd.Series): DataFrame的一行，必须包含以下列:
                         'concentration_90pct', 'concentration_90pct_slope_5d',
                         'winner_profit_margin', 'price_to_peak_ratio'.
    Returns:
        float: 0-100之间的筹码健康分。
    """
    score = 50.0  # 基础分

    # 1. 集中度越高越好 (最多加30分)
    #    - concentration_90pct: 值越小越集中
    conc_90 = row.get('concentration_90pct')
    if pd.notna(conc_90):
        # 当集中度从 0.3 (30%) 开始下降时，线性加分。小于等于0时加满30分。
        # 将 decimal.Decimal 类型的 conc_90 转换为 float 类型，以避免与 float 类型的 0.3 运算时发生 TypeError
        score += max(0, (0.3 - float(conc_90))) * 100
    
    # 2. 集中度趋势在收敛，加分 (固定加10分)
    #    - concentration_90pct_slope_5d: 负值表示趋于集中
    conc_slope = row.get('concentration_90pct_slope_5d')
    if pd.notna(conc_slope):
        # 将 conc_slope 转换为 float 类型，以确保与 0 的比较和后续运算的类型一致性
        if float(conc_slope) < 0:
            score += 10
            
    # 3. 获利盘安全垫越高越好 (最多加25分)
    #    - winner_profit_margin: 获利盘的平均利润率(%)
    profit_margin = row.get('winner_profit_margin')
    if pd.notna(profit_margin):
        # 每1%的安全垫加1分，最多加25分
        # 将 profit_margin 转换为 float 类型，以确保与整数/浮点数运算的类型安全
        score += min(max(0, float(profit_margin)), 25)
        
    # 4. 股价显著高于主筹码峰，加分 (固定加15分)
    #    - price_to_peak_ratio: 股价/筹码峰成本比
    price_ratio = row.get('price_to_peak_ratio')
    if pd.notna(price_ratio):
        # 将 price_ratio 转换为 float 类型，以确保与浮点数 1.05 比较的类型安全
        if float(price_ratio) > 1.05:
            score += 15
            
    # 确保分数在0-100之间
    return max(0, min(100, round(score, 2)))
