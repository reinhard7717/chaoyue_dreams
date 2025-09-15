# 文件: strategies/trend_following/intelligence/fund_flow_intelligence.py
# 资金流情报模块
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.trend_following.utils import get_params_block, get_param_value

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _normalize_score(self, series: pd.Series, window: int, target_index: pd.Index, ascending: bool = True) -> pd.Series:
        """
        【V12.1 健壮性修复版】计算一个系列在滚动窗口内的归一化得分 (0-1)。
        - 核心修复: 移除对 self.strategy.df.index 的依赖，改为接收一个 target_index 参数，
                      确保在创建默认Series时使用正确的索引，避免因状态不同步导致返回空Series。
        """
        if series is None or series.isnull().all():
            return pd.Series(0.5, index=target_index)

        return series.rolling(
            window=window, 
            min_periods=int(window * 0.2)
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5).astype(np.float32)

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V19.0 终极信号版】资金流情报分析总指挥
        - 核心重构: 遵循终极信号范式，本模块不再返回一堆零散的原子信号。
                      现在只调用唯一的终极信号引擎 `diagnose_ultimate_fund_flow_signals`，
                      并将其产出的16个S+/S/A/B级信号作为本模块的最终输出。
        - 收益: 架构与其他情报模块完全统一，极大提升了信号质量和架构清晰度。
        """
        # print("      -> [资金流情报分析总指挥 V19.0 终极信号版] 启动...")
        
        p = get_params_block(self.strategy, 'fund_flow_params')
        is_enabled = get_param_value(p.get('enabled') if p else None, False)
        if not is_enabled:
            return {}
            
        # 直接调用终极信号引擎，并将其结果作为本模块的唯一输出
        ultimate_ff_states = self.diagnose_ultimate_fund_flow_signals(df)

        # print(f"      -> [资金流情报分析总指挥 V19.0] 分析完毕，共生成 {len(ultimate_ff_states)} 个终极资金流信号。")
        return ultimate_ff_states

    def diagnose_ultimate_fund_flow_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 逻辑修复版】终极资金流信号诊断模块
        - 核心修复 (本次修改):
          - [BUG修复] 修复了在计算看跌信号组件时，因拼写错误（将 `overall_bearish_health` 错写为 `bearish_bearish_health`）而导致的 `NameError`。
        - 核心升级 (V1.1逻辑保留):
          - [性能优化] 重构了核心计算逻辑，采用“批量预处理 + NumPy原生计算”范式，避免了在循环中生成大量中间Series，大幅降低了内存峰值和计算开销。
        - 核心范式 (V1.0逻辑保留):
          - 1. 六大支柱: 将资金流情报提炼为聚合流、结构流、冲突流、量能流、核心流、强度流六大支柱。
          - 2. 深度交叉验证: 对每一支柱，在每一时间周期上进行“静态 x 动态(斜率) x 加速”三维交叉验证。
          - 3. 多维共识融合: 将六大支柱在同一周期的“健康分”进行几何平均，形成“全面共识健康度”。
          - 4. 终极信号合成: 基于“全面共识健康度”，构建标准的S+/S/A/B四级共振与反转信号。
        - 收益:
          - 修复了导致程序崩溃的严重BUG。
          - 保持了V1.1版本带来的显著性能提升。
        """
        # print("        -> [终极资金流信号诊断模块 V1.2 逻辑修复版] 启动...")
        states = {}
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        pillars = {
            'fund_flow': 'net_flow_consensus',
            'structure': 'main_force_net_flow_consensus',
            'conflict': 'flow_divergence_mf_vs_retail',
            'cmf': 'CMF_21',
            'xl_order': 'net_xl_amount_consensus',
            'intensity': 'main_force_flow_intensity_ratio'
        }
        pillar_types = {
            'uses_sum': ['fund_flow', 'structure', 'xl_order'],
            'uses_daily': ['conflict', 'intensity', 'cmf']
        }
        required_cols = set()
        for p in periods:
            for pillar_key, pillar_prefix in pillars.items():
                if p > 1 and pillar_key in pillar_types['uses_sum']:
                    static_base = f"{pillar_prefix}_sum_{p}d"
                else:
                    static_base = pillar_prefix
                slope_base = f"SLOPE_{p}_{static_base}"
                accel_base = f"ACCEL_{p}_{pillar_prefix}"
                required_cols.add(f"{static_base}_D")
                required_cols.add(f"{slope_base}_D")
                required_cols.add(f"{accel_base}_D")
        missing_cols = list(required_cols - set(df.columns))
        if missing_cols:
            print(f"          -> [严重警告] 终极资金流引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states
        # --- 2. 核心要素数值化 (批量预处理) ---
        all_cols_to_normalize = set()
        for p in periods:
            for pillar_key, pillar_prefix in pillars.items():
                if p > 1 and pillar_key in pillar_types['uses_sum']:
                    static_col = f"{pillar_prefix}_sum_{p}d_D"
                else:
                    static_col = f"{pillar_prefix}_D"
                static_base_for_slope = static_col.replace('_D', '')
                slope_col = f"SLOPE_{p}_{static_base_for_slope}_D"
                accel_col = f"ACCEL_{p}_{pillar_prefix}_D"
                all_cols_to_normalize.add(static_col)
                all_cols_to_normalize.add(slope_col)
                all_cols_to_normalize.add(accel_col)
        normalized_scores = {
            col_name: self._normalize_score(df[col_name], norm_window, df.index)
            for col_name in all_cols_to_normalize
        }
        # --- 3. 融合生成“全面共识健康度” (NumPy原生计算) ---
        overall_bullish_health = {}
        for p in periods:
            pillar_health_arrays = []
            for pillar_key, pillar_prefix in pillars.items():
                if p > 1 and pillar_key in pillar_types['uses_sum']:
                    static_col = f"{pillar_prefix}_sum_{p}d_D"
                else:
                    static_col = f"{pillar_prefix}_D"
                static_base_for_slope = static_col.replace('_D', '')
                slope_col = f"SLOPE_{p}_{static_base_for_slope}_D"
                accel_col = f"ACCEL_{p}_{pillar_prefix}_D"
                static_score_arr = normalized_scores[static_col].values
                slope_score_arr = normalized_scores[slope_col].values
                accel_score_arr = normalized_scores[accel_col].values
                pillar_health_arr = static_score_arr * slope_score_arr * accel_score_arr
                pillar_health_arrays.append(pillar_health_arr)
            stacked_health_arrays = np.stack(pillar_health_arrays, axis=0)
            num_pillars = len(pillars)
            overall_health_arr = np.prod(stacked_health_arrays, axis=0)**(1/num_pillars)
            overall_bullish_health[p] = pd.Series(overall_health_arr, index=df.index, dtype=np.float32)
        overall_bearish_health = {p: 1.0 - overall_bullish_health[p] for p in periods}
        # --- 4. 定义信号组件 ---
        bullish_short_force = (overall_bullish_health[1] * overall_bullish_health[5])**0.5
        bullish_medium_trend = (overall_bullish_health[13] * overall_bullish_health[21])**0.5
        bullish_long_inertia = overall_bullish_health[55]
        bearish_short_force = (overall_bearish_health[1] * overall_bearish_health[5])**0.5
        bearish_medium_trend = (overall_bearish_health[13] * overall_bearish_health[21])**0.5
        bearish_long_inertia = overall_bearish_health[55]
        # --- 5. 共振信号合成 ---
        states['SCORE_FF_BULLISH_RESONANCE_B'] = overall_bullish_health[5].astype(np.float32)
        states['SCORE_FF_BULLISH_RESONANCE_A'] = (overall_bullish_health[5] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FF_BULLISH_RESONANCE_S'] = (bullish_short_force * bullish_medium_trend).astype(np.float32)
        states['SCORE_FF_BULLISH_RESONANCE_S_PLUS'] = (states['SCORE_FF_BULLISH_RESONANCE_S'] * bullish_long_inertia).astype(np.float32)
        states['SCORE_FF_BEARISH_RESONANCE_B'] = overall_bearish_health[5].astype(np.float32)
        states['SCORE_FF_BEARISH_RESONANCE_A'] = (overall_bearish_health[5] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FF_BEARISH_RESONANCE_S'] = (bearish_short_force * bearish_medium_trend).astype(np.float32)
        states['SCORE_FF_BEARISH_RESONANCE_S_PLUS'] = (states['SCORE_FF_BEARISH_RESONANCE_S'] * bearish_long_inertia).astype(np.float32)
        # --- 6. 反转信号合成 ---
        states['SCORE_FF_BOTTOM_REVERSAL_B'] = (overall_bullish_health[1] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FF_BOTTOM_REVERSAL_A'] = (overall_bullish_health[5] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FF_BOTTOM_REVERSAL_S'] = (bullish_short_force * bearish_long_inertia).astype(np.float32)
        states['SCORE_FF_BOTTOM_REVERSAL_S_PLUS'] = (bullish_short_force * bullish_medium_trend * bearish_long_inertia).astype(np.float32)
        states['SCORE_FF_TOP_REVERSAL_B'] = (overall_bearish_health[1] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FF_TOP_REVERSAL_A'] = (overall_bearish_health[5] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FF_TOP_REVERSAL_S'] = (bearish_short_force * bullish_long_inertia).astype(np.float32)
        states['SCORE_FF_TOP_REVERSAL_S_PLUS'] = (bearish_short_force * bearish_medium_trend * bullish_long_inertia).astype(np.float32)
        # print(f"        -> [终极资金流信号诊断模块 V1.2] 分析完毕，生成 {len(states)} 个终极信号。")
        return states

    def _calculate_normalized_score(self, series: pd.Series, window: int, target_index: pd.Index, ascending: bool = True) -> pd.Series:
        """
        【V12.1 健壮性修复版】计算一个系列在滚动窗口内的归一化得分 (0-1)。
        - 核心修复: 移除对 self.strategy.df.index 的依赖，改为接收一个 target_index 参数，
                      确保在创建默认Series时使用正确的索引，避免因状态不同步导致返回空Series。
        """
        # 使用传入的 target_index 代替 self.strategy.df.index
        if series is None or series.isnull().all():
            return pd.Series(0.5, index=target_index)

        return series.rolling(
            window=window, 
            min_periods=int(window * 0.2)
        ).rank(
            pct=True, 
            ascending=ascending
        ).fillna(0.5)

    def _diagnose_fund_flow_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V12.1 - 动态共振与反转引擎】
        - 核心: 基于多时间维度(5, 13, 21, 55日)的静态、斜率、加速度指标，进行深度交叉验证。
        - 产出: 生成4类核心动态信号得分：上升共振、下跌共振、底部反转、顶部反转。
        - 优化: 融合更多加速度指标，提升信号置信度。
        """
        print("            -> [资金流动态引擎 V12.1] 启动交叉验证...")
        available_cols = set(df.columns)
        periods = [5, 13, 21, 55]
        norm_window = 120
        # --- 步骤 1: 预计算所有周期的共识指标得分 ---
        metrics = {}
        for p in periods:
            metrics[f'static_{p}'] = df.get(f"net_flow_consensus_sum_{p}d_D")
            metrics[f'slope_{p}'] = df.get(f"SLOPE_{p}_net_flow_consensus_sum_{p}d_D")
            if p in [5, 13, 21]: # 仅计算模型中存在的加速度周期
                 metrics[f'accel_{p}'] = df.get(f"ACCEL_{p}_net_flow_consensus_D")
        # --- 步骤 2: 生成上升共振信号 (Upward Resonance) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s5 = self._calculate_normalized_score(metrics['static_5'], norm_window, df.index)
        sl5 = self._calculate_normalized_score(metrics['slope_5'], norm_window, df.index)
        a5 = self._calculate_normalized_score(metrics['accel_5'], norm_window, df.index)
        a13 = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index) # 获取并归一化13日加速度
        a21 = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index) # 获取并归一化21日加速度
        s21 = self._calculate_normalized_score(metrics['static_21'], norm_window, df.index)
        sl21 = self._calculate_normalized_score(metrics['slope_21'], norm_window, df.index)
        s55 = self._calculate_normalized_score(metrics['static_55'], norm_window, df.index)
        sl55 = self._calculate_normalized_score(metrics['slope_55'], norm_window, df.index)

        df['FF_SCORE_RESONANCE_UP_LOW'] = s5 * sl5 * a5
        df['FF_SCORE_RESONANCE_UP_MID'] = df['FF_SCORE_RESONANCE_UP_LOW'] * s21 * sl21 * a13 * a21 # 中置信度信号融合13日和21日加速度，增强信号强度
        df['FF_SCORE_RESONANCE_UP_HIGH'] = df['FF_SCORE_RESONANCE_UP_MID'] * s55 * sl55
        print("               - 上升共振信号已生成 (低/中/高置信度)")
        # --- 步骤 3: 生成下跌共振信号 (Downward Resonance) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s5_neg = self._calculate_normalized_score(metrics['static_5'], norm_window, df.index, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics['slope_5'], norm_window, df.index, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics['accel_5'], norm_window, df.index, ascending=False)
        a13_neg = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index, ascending=False) # 获取并归一化13日加速度(负向)
        a21_neg = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index, ascending=False) # 获取并归一化21日加速度(负向)
        s21_neg = self._calculate_normalized_score(metrics['static_21'], norm_window, df.index, ascending=False)
        sl21_neg = self._calculate_normalized_score(metrics['slope_21'], norm_window, df.index, ascending=False)
        s55_neg = self._calculate_normalized_score(metrics['static_55'], norm_window, df.index, ascending=False)
        sl55_neg = self._calculate_normalized_score(metrics['slope_55'], norm_window, df.index, ascending=False)

        df['FF_SCORE_RESONANCE_DOWN_LOW'] = s5_neg * sl5_neg * a5_neg
        df['FF_SCORE_RESONANCE_DOWN_MID'] = df['FF_SCORE_RESONANCE_DOWN_LOW'] * s21_neg * sl21_neg * a13_neg * a21_neg # 中置信度信号融合13日和21日加速度，增强信号强度
        df['FF_SCORE_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_RESONANCE_DOWN_MID'] * s55_neg * sl55_neg
        print("               - 下跌共振信号已生成 (低/中/高置信度)")
        # --- 步骤 4: 生成底部反转信号 (Bottom Reversal) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        long_term_bottoming = self._calculate_normalized_score(metrics['slope_55'], norm_window, df.index, ascending=False)
        mid_term_stabilizing = self._calculate_normalized_score(metrics['slope_21'], norm_window, df.index)
        short_term_reversing = self._calculate_normalized_score(metrics['slope_5'], norm_window, df.index)
        short_term_accelerating = self._calculate_normalized_score(metrics['accel_5'], norm_window, df.index)
        mid_term_accelerating = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index) # 获取13日加速度作为额外确认

        df['FF_SCORE_REVERSAL_BOTTOM_MID'] = mid_term_stabilizing * short_term_reversing
        df['FF_SCORE_REVERSAL_BOTTOM_HIGH'] = long_term_bottoming * mid_term_stabilizing * short_term_reversing * short_term_accelerating * mid_term_accelerating # 高置信度信号增加13日加速度作为共振确认，提升S+级别置信度
        print("               - 底部反转信号已生成 (中/高置信度)")
        # --- 步骤 5: 生成顶部反转信号 (Top Reversal) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        long_term_topping = self._calculate_normalized_score(metrics['slope_55'], norm_window, df.index)
        mid_term_stalling = self._calculate_normalized_score(metrics['slope_21'], norm_window, df.index, ascending=False)
        short_term_diverging = self._calculate_normalized_score(metrics['slope_5'], norm_window, df.index, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics['accel_5'], norm_window, df.index, ascending=False)
        mid_term_decelerating = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index, ascending=False) # 获取13日加速度(负向)作为额外确认

        df['FF_SCORE_REVERSAL_TOP_MID'] = mid_term_stalling * short_term_diverging
        df['FF_SCORE_REVERSAL_TOP_HIGH'] = long_term_topping * mid_term_stalling * short_term_diverging * short_term_decelerating * mid_term_decelerating # 高置信度信号增加13日加速度作为共振确认，提升S+级别置信度
        print("               - 顶部反转信号已生成 (中/高置信度)")
        return df

    def _diagnose_capital_structure_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V16.1 - 主力资金共振与反转引擎】
        - 核心重构: 逻辑与聚合流引擎对齐，使用更可靠的指标组合。
            - 静态背景: 使用【累计】主力净流入 (`_sum_`)，代表主力资金的存量和仓位。
            - 趋势方向: 使用【累计】主力净流入的斜率，代表主力建仓/减仓的趋势。
            - 动能变化: 使用【每日】主力净流入的加速度，代表主力动作的瞬时变化率。
        - 新增信号: 增加了主力资金行为的顶部/底部反转信号。
        - 优化: 融合更多加速度指标，提升信号置信度。
        """
        print("            -> [资金结构引擎 V16.1] 启动交叉验证...")
        periods = [5, 13, 21, 55]
        norm_window = 120
        # --- 步骤 1: 获取预计算好的主力资金共识指标 ---
        # 修复了复制粘贴错误，将指标从超大单(xl)修正为主力(main_force)
        metrics = {}
        for p in periods:
            metrics[f'static_{p}'] = df.get(f"main_force_net_flow_consensus_sum_{p}d_D")
            metrics[f'slope_{p}'] = df.get(f"SLOPE_{p}_main_force_net_flow_consensus_sum_{p}d_D")
            if p in [5, 13, 21]: # 仅计算模型中存在的加速度周期
                 metrics[f'accel_{p}'] = df.get(f"ACCEL_{p}_main_force_net_flow_consensus_D")
        # --- 步骤 2: 生成主力资金上升共振信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s5 = self._calculate_normalized_score(metrics.get('static_5'), norm_window, df.index)
        sl5 = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        a5 = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index)
        a13 = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index) # 获取并归一化13日加速度
        a21 = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index) # 获取并归一化21日加速度
        s21 = self._calculate_normalized_score(metrics.get('static_21'), norm_window, df.index)
        sl21 = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index)
        s55 = self._calculate_normalized_score(metrics.get('static_55'), norm_window, df.index)
        sl55 = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index)

        df['FF_SCORE_STRUCTURE_RESONANCE_UP_LOW'] = s5 * sl5 * a5
        df['FF_SCORE_STRUCTURE_RESONANCE_UP_MID'] = df['FF_SCORE_STRUCTURE_RESONANCE_UP_LOW'] * s21 * sl21 * a13 * a21 # 中置信度信号融合13日和21日加速度
        df['FF_SCORE_STRUCTURE_RESONANCE_UP_HIGH'] = df['FF_SCORE_STRUCTURE_RESONANCE_UP_MID'] * s55 * sl55
        print("               - [结构]上升共振信号已生成 (低/中/高置信度)")
        # --- 步骤 3: 生成主力资金下跌共振信号 (主力撤退) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s5_neg = self._calculate_normalized_score(metrics.get('static_5'), norm_window, df.index, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False)
        a13_neg = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index, ascending=False) # 获取并归一化13日加速度(负向)
        a21_neg = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index, ascending=False) # 获取并归一化21日加速度(负向)
        s21_neg = self._calculate_normalized_score(metrics.get('static_21'), norm_window, df.index, ascending=False)
        sl21_neg = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index, ascending=False)
        s55_neg = self._calculate_normalized_score(metrics.get('static_55'), norm_window, df.index, ascending=False)
        sl55_neg = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index, ascending=False)

        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_LOW'] = s5_neg * sl5_neg * a5_neg
        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_MID'] = df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_LOW'] * s21_neg * sl21_neg * a13_neg * a21_neg # 中置信度信号融合13日和21日加速度
        df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_STRUCTURE_RESONANCE_DOWN_MID'] * s55_neg * sl55_neg
        print("               - [结构]下跌共振(主力撤退)信号已生成 (低/中/高置信度)")
        # --- 步骤 4: 生成主力资金底部反转信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        long_term_selling = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index, ascending=False)
        mid_term_stabilizing = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index)
        short_term_reversing = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        short_term_accelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index)
        mid_term_accelerating = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index) # 获取13日加速度作为额外确认

        df['FF_SCORE_STRUCTURE_REVERSAL_BOTTOM_MID'] = mid_term_stabilizing * short_term_reversing
        df['FF_SCORE_STRUCTURE_REVERSAL_BOTTOM_HIGH'] = long_term_selling * mid_term_stabilizing * short_term_reversing * short_term_accelerating * mid_term_accelerating # 高置信度信号增加13日加速度作为共振确认
        print("               - [结构]底部反转信号已生成 (中/高置信度)")
        # --- 步骤 5: 生成主力资金顶部反转信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        long_term_buying = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index)
        mid_term_stalling = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index, ascending=False)
        short_term_diverging = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False)
        mid_term_decelerating = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index, ascending=False) # 获取13日加速度(负向)作为额外确认

        df['FF_SCORE_STRUCTURE_REVERSAL_TOP_MID'] = mid_term_stalling * short_term_diverging
        df['FF_SCORE_STRUCTURE_REVERSAL_TOP_HIGH'] = long_term_buying * mid_term_stalling * short_term_diverging * short_term_decelerating * mid_term_decelerating # 高置信度信号增加13日加速度作为共振确认
        print("               - [结构]顶部反转信号已生成 (中/高置信度)")
        return df

    def _diagnose_capital_conflict_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V19.1 - 主力散户分歧·多维交叉验证引擎】
        - 核心升级: 全面利用数据层提供的分歧度静态、斜率、加速度指标，构建共振与反转信号。
        - 信号体系:
            - 基础分歧: 当前主力与散户的对立状态。
            - 分歧共振: 确认主力吸筹或派发趋势的持续性与强度。
            - 分歧反转: 预警主力行为发生关键性逆转的顶部或底部。
        - 数据依赖: 依赖 `flow_divergence_mf_vs_retail_D` 及其 `SLOPE` 和 `accel` 衍生列。
        """
        print("            -> [资金冲突引擎 V19.1] 启动多维交叉验证...")
        norm_window = 120
        periods = [5, 13, 21] # 定义用于共振的核心周期
        # --- 步骤 1: 获取分歧度的静态、斜率、加速度全套指标 ---
        metrics = {
            'static': df.get('flow_divergence_mf_vs_retail_D'),
            'slope_5': df.get('SLOPE_5_flow_divergence_mf_vs_retail_D'),
            'slope_13': df.get('SLOPE_13_flow_divergence_mf_vs_retail_D'),
            'slope_21': df.get('SLOPE_21_flow_divergence_mf_vs_retail_D'),
            'accel_5': df.get('accel_5d_flow_divergence_mf_vs_retail_D'),
            'accel_13': df.get('accel_13d_flow_divergence_mf_vs_retail_D'),
            'accel_21': df.get('accel_21d_flow_divergence_mf_vs_retail_D'),
        }
        # --- 步骤 2: [保留并优化] 计算基础分歧信号 (当前状态) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        df['FF_SCORE_CONFLICT_MF_BUYS_RETAIL_SELLS'] = self._calculate_normalized_score(metrics['static'], norm_window, df.index)
        df['FF_SCORE_CONFLICT_MF_SELLS_RETAIL_BUYS'] = self._calculate_normalized_score(metrics['static'], norm_window, df.index, ascending=False)

        print("               - [冲突]基础分歧信号已生成 (当前状态)")
        # --- 步骤 3: 生成分歧共振信号 (趋势确认) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        static_pos = df['FF_SCORE_CONFLICT_MF_BUYS_RETAIL_SELLS'] # 复用已计算的静态分歧度得分
        static_neg = df['FF_SCORE_CONFLICT_MF_SELLS_RETAIL_BUYS'] # 复用已计算的静态分歧度得分(负向)
        sl5_pos = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        sl13_pos = self._calculate_normalized_score(metrics.get('slope_13'), norm_window, df.index)
        sl21_pos = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        sl13_neg = self._calculate_normalized_score(metrics.get('slope_13'), norm_window, df.index, ascending=False)
        sl21_neg = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index, ascending=False)
        a5_pos = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index) # 获取并归一化多周期加速度
        a13_pos = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index) # 获取并归一化多周期加速度
        a21_pos = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index) # 获取并归一化多周期加速度
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False) # 获取并归一化多周期加速度(负向)
        a13_neg = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index, ascending=False) # 获取并归一化多周期加速度(负向)
        a21_neg = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index, ascending=False) # 获取并归一化多周期加速度(负向)

        # 重构分歧共振信号，融合静态、斜率、加速度进行多维交叉验证
        df['FF_SCORE_CONFLICT_RESONANCE_UP_LOW'] = static_pos * sl5_pos * a5_pos
        df['FF_SCORE_CONFLICT_RESONANCE_UP_MID'] = df['FF_SCORE_CONFLICT_RESONANCE_UP_LOW'] * sl13_pos * a13_pos
        df['FF_SCORE_CONFLICT_RESONANCE_UP_HIGH'] = df['FF_SCORE_CONFLICT_RESONANCE_UP_MID'] * sl21_pos * a21_pos
        df['FF_SCORE_CONFLICT_RESONANCE_DOWN_LOW'] = static_neg * sl5_neg * a5_neg
        df['FF_SCORE_CONFLICT_RESONANCE_DOWN_MID'] = df['FF_SCORE_CONFLICT_RESONANCE_DOWN_LOW'] * sl13_neg * a13_neg
        df['FF_SCORE_CONFLICT_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_CONFLICT_RESONANCE_DOWN_MID'] * sl21_neg * a21_neg
        print("               - [冲突]分歧共振信号已生成 (上升/下跌趋势确认)")
        # --- 步骤 4: 生成分歧反转信号 (顶部/底部预警) ---
        static_high_score = df['FF_SCORE_CONFLICT_MF_BUYS_RETAIL_SELLS'] # 复用基础分歧信号
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        slope_reversing_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        accel_reversing_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False)

        df['FF_SCORE_CONFLICT_REVERSAL_TOP_HIGH'] = static_high_score * slope_reversing_neg * accel_reversing_neg
        static_low_score = df['FF_SCORE_CONFLICT_MF_SELLS_RETAIL_BUYS'] # 复用基础分歧信号
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        slope_reversing_pos = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        accel_reversing_pos = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index)

        df['FF_SCORE_CONFLICT_REVERSAL_BOTTOM_HIGH'] = static_low_score * slope_reversing_pos * accel_reversing_pos
        print("               - [冲突]分歧反转信号已生成 (高置信度顶部/底部预警)")
        
        return df

    def _diagnose_cmf_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V15.1 - CMF动态诊断引擎】
        - 核心: 基于Chaikin Money Flow (CMF)指标的静态、斜率、加速度进行交叉验证。
        - 视角: 从成交量加权的角度衡量买卖压力及其动态变化。
        - 产出: 生成买卖压力的共振与反转信号。
        - 优化: 增强高置信度共振信号的验证条件。
        """
        print("            -> [CMF动态引擎 V1.1] 启动交叉验证...")
        norm_window = 120
        # --- 步骤 1: 获取CMF全套指标 ---
        metrics = {
            'static': df.get('CMF_21_D'),
            'slope_5': df.get('SLOPE_5_CMF_21_D'),
            'slope_21': df.get('SLOPE_21_CMF_21_D'),
            'accel_5': df.get('ACCEL_5_CMF_21_D'),
            'accel_21': df.get('ACCEL_21_CMF_21_D'),
        }
        # --- 步骤 2: 生成买压共振信号 (Upward Resonance) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s_pos = self._calculate_normalized_score(metrics['static'], norm_window, df.index)
        sl5_pos = self._calculate_normalized_score(metrics['slope_5'], norm_window, df.index)
        sl21_pos = self._calculate_normalized_score(metrics['slope_21'], norm_window, df.index)
        a5_pos = self._calculate_normalized_score(metrics['accel_5'], norm_window, df.index)
        a21_pos = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index) # 获取并归一化21日加速度

        df['FF_SCORE_CMF_RESONANCE_UP_LOW'] = s_pos * sl5_pos * a5_pos
        df['FF_SCORE_CMF_RESONANCE_UP_HIGH'] = df['FF_SCORE_CMF_RESONANCE_UP_LOW'] * sl21_pos * a21_pos # 高置信度信号融合21日斜率与21日加速度，形成更强的中长周期共振
        print("               - [CMF]买压共振信号已生成 (低/高置信度)")
        # --- 步骤 3: 生成卖压共振信号 (Downward Resonance) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s_neg = self._calculate_normalized_score(metrics['static'], norm_window, df.index, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics['slope_5'], norm_window, df.index, ascending=False)
        sl21_neg = self._calculate_normalized_score(metrics['slope_21'], norm_window, df.index, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics['accel_5'], norm_window, df.index, ascending=False)
        a21_neg = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index, ascending=False) # 获取并归一化21日加速度(负向)

        df['FF_SCORE_CMF_RESONANCE_DOWN_LOW'] = s_neg * sl5_neg * a5_neg
        df['FF_SCORE_CMF_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_CMF_RESONANCE_DOWN_LOW'] * sl21_neg * a21_neg # 高置信度信号融合21日斜率与21日加速度，形成更强的中长周期共振
        print("               - [CMF]卖压共振信号已生成 (低/高置信度)")
        # --- 步骤 4: 生成底部反转信号 (Bottom Reversal) ---
        df['FF_SCORE_CMF_REVERSAL_BOTTOM_HIGH'] = s_neg * sl5_pos * a5_pos
        print("               - [CMF]底部反转信号已生成")
        # --- 步骤 5: 生成顶部反转信号 (Top Reversal) ---
        df['FF_SCORE_CMF_REVERSAL_TOP_HIGH'] = s_pos * sl5_neg * a5_neg
        print("               - [CMF]顶部反转信号已生成")
        return df

    def _diagnose_xl_order_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V16.1 - 超大单动态诊断引擎】
        - 核心: 聚焦于net_xl_amount (超大单净额)，分析市场中最核心力量的动态。
        - 逻辑: 与主力资金引擎对齐，采用“累计值(静态)+累计值斜率(趋势)+每日值加速度(动能)”的黄金组合。
        - 产出: 生成关于“聪明钱”的共振与反转信号。
        - 数据假设: 依赖数据层提供 net_xl_amount 的 sum, SLOPE, ACCEL 衍生列。
        - 优化: 增加中置信度信号，强化反转信号逻辑。
        """
        print("            -> [超大单动态引擎 V1.1] 启动交叉验证...")
        periods = [5, 21, 55] # 使用简化的周期组合
        norm_window = 120
        # --- 步骤 1: 获取超大单资金指标 ---
        metrics = {}
        for p in periods:
            # 在所有列名中添加 "_consensus" 后缀以匹配军械库清单
            metrics[f'static_{p}'] = df.get(f"net_xl_amount_consensus_sum_{p}d_D")
            metrics[f'slope_{p}'] = df.get(f"SLOPE_{p}_net_xl_amount_consensus_sum_{p}d_D")
            if p in [5, 21]:
                 # 在所有列名中添加 "_consensus" 后缀以匹配军械库清单
                 metrics[f'accel_{p}'] = df.get(f"ACCEL_{p}_net_xl_amount_consensus_D")
        # --- 步骤 2: 生成超大单吸筹共振信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s5 = self._calculate_normalized_score(metrics.get('static_5'), norm_window, df.index)
        sl5 = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        a5 = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index)
        s21 = self._calculate_normalized_score(metrics.get('static_21'), norm_window, df.index) # 获取并归一化21日周期的静态指标
        sl21 = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index) # 获取并归一化21日周期的斜率指标
        a21 = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index) # 获取并归一化21日周期的加速度指标
        s55 = self._calculate_normalized_score(metrics.get('static_55'), norm_window, df.index)
        sl55 = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index)

        df['FF_SCORE_XL_RESONANCE_UP_LOW'] = s5 * sl5 * a5
        df['FF_SCORE_XL_RESONANCE_UP_MID'] = df['FF_SCORE_XL_RESONANCE_UP_LOW'] * s21 * sl21 * a21 # 创建中置信度共振信号，融合21日周期指标
        df['FF_SCORE_XL_RESONANCE_UP_HIGH'] = df['FF_SCORE_XL_RESONANCE_UP_MID'] * s55 * sl55 # 高置信度信号基于中置信度信号进行构建
        print("               - [超大单]吸筹共振信号已生成 (低/中/高置信度)") # 更新输出信息
        # --- 步骤 3: 生成超大单派发共振信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s5_neg = self._calculate_normalized_score(metrics.get('static_5'), norm_window, df.index, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False)
        s21_neg = self._calculate_normalized_score(metrics.get('static_21'), norm_window, df.index, ascending=False) # 获取并归一化21日周期的静态指标(负向)
        sl21_neg = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index, ascending=False) # 获取并归一化21日周期的斜率指标(负向)
        a21_neg = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index, ascending=False) # 获取并归一化21日周期的加速度指标(负向)
        s55_neg = self._calculate_normalized_score(metrics.get('static_55'), norm_window, df.index, ascending=False)
        sl55_neg = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index, ascending=False)

        df['FF_SCORE_XL_RESONANCE_DOWN_LOW'] = s5_neg * sl5_neg * a5_neg
        df['FF_SCORE_XL_RESONANCE_DOWN_MID'] = df['FF_SCORE_XL_RESONANCE_DOWN_LOW'] * s21_neg * sl21_neg * a21_neg # 创建中置信度共振信号，融合21日周期指标(负向)
        df['FF_SCORE_XL_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_XL_RESONANCE_DOWN_MID'] * s55_neg * sl55_neg # 高置信度信号基于中置信度信号进行构建
        print("               - [超大单]派发共振信号已生成 (低/中/高置信度)") # 更新输出信息
        # --- 步骤 4: 生成超大单底部反转信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        long_term_selling = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index, ascending=False)
        mid_term_stabilizing = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index) # 获取并归一化21日斜率作为中期企稳信号
        short_term_reversing = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        short_term_accelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index)

        df['FF_SCORE_XL_REVERSAL_BOTTOM_HIGH'] = long_term_selling * mid_term_stabilizing * short_term_reversing * short_term_accelerating # 高置信度反转信号增加中期企稳(sl21)验证，逻辑更严谨
        print("               - [超大单]底部反转信号已生成")
        # --- 步骤 5: 生成超大单顶部反转信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        long_term_buying = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index)
        mid_term_stalling = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index, ascending=False) # 获取并归一化21日斜率(负向)作为中期失速信号
        short_term_diverging = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False)

        df['FF_SCORE_XL_REVERSAL_TOP_HIGH'] = long_term_buying * mid_term_stalling * short_term_diverging * short_term_decelerating # 高置信度反转信号增加中期失速(sl21)验证，逻辑更严谨
        print("               - [超大单]顶部反转信号已生成")
        return df

    def _diagnose_retail_flow_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V17.1 - 散户动态诊断引擎】
        - 核心: 独立分析散户资金流 (retail_net_flow_consensus) 的动态，捕捉市场情绪。
        - 视角: 将散户行为作为市场情绪的“温度计”和潜在的“逆向指标”。
        - 产出: 生成描述散户“买入狂热”和“恐慌杀跌”的共振与反转信号。
        - 数据假设: 依赖数据层提供 retail_net_flow_consensus 的全套衍生列。
        - 优化: 增加中置信度信号，强化反转信号逻辑。
        """
        print("            -> [散户动态引擎 V1.1] 启动交叉验证...")
        periods = [5, 21, 55]
        norm_window = 120
        # --- 步骤 1: 获取散户资金指标 ---
        metrics = {}
        for p in periods:
            metrics[f'static_{p}'] = df.get(f"retail_net_flow_consensus_sum_{p}d_D")
            metrics[f'slope_{p}'] = df.get(f"SLOPE_{p}_retail_net_flow_consensus_sum_{p}d_D")
            if p in [5, 21]:
                 metrics[f'accel_{p}'] = df.get(f"ACCEL_{p}_retail_net_flow_consensus_D")
        # --- 步骤 2: 生成散户买入狂热共振信号 (Buying Frenzy) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s5 = self._calculate_normalized_score(metrics.get('static_5'), norm_window, df.index)
        sl5 = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        a5 = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index)
        s21 = self._calculate_normalized_score(metrics.get('static_21'), norm_window, df.index) # 获取并归一化21日周期的静态指标
        sl21 = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index) # 获取并归一化21日周期的斜率指标
        a21 = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index) # 获取并归一化21日周期的加速度指标
        s55 = self._calculate_normalized_score(metrics.get('static_55'), norm_window, df.index)
        sl55 = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index)

        df['FF_SCORE_RETAIL_RESONANCE_FRENZY_LOW'] = s5 * sl5 * a5
        df['FF_SCORE_RETAIL_RESONANCE_FRENZY_MID'] = df['FF_SCORE_RETAIL_RESONANCE_FRENZY_LOW'] * s21 * sl21 * a21 # 创建中置信度狂热信号
        df['FF_SCORE_RETAIL_RESONANCE_FRENZY_HIGH'] = df['FF_SCORE_RETAIL_RESONANCE_FRENZY_MID'] * s55 * sl55 # 高置信度信号基于中置信度构建
        print("               - [散户]买入狂热共振信号已生成 (低/中/高置信度)") # 更新输出信息
        # --- 步骤 3: 生成散户恐慌杀跌共振信号 (Capitulation) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s5_neg = self._calculate_normalized_score(metrics.get('static_5'), norm_window, df.index, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False)
        s21_neg = self._calculate_normalized_score(metrics.get('static_21'), norm_window, df.index, ascending=False) # 获取并归一化21日周期的静态指标(负向)
        sl21_neg = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index, ascending=False) # 获取并归一化21日周期的斜率指标(负向)
        a21_neg = self._calculate_normalized_score(metrics.get('accel_21'), norm_window, df.index, ascending=False) # 获取并归一化21日周期的加速度指标(负向)
        s55_neg = self._calculate_normalized_score(metrics.get('static_55'), norm_window, df.index, ascending=False)
        sl55_neg = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index, ascending=False)

        df['FF_SCORE_RETAIL_RESONANCE_CAPITULATION_LOW'] = s5_neg * sl5_neg * a5_neg
        df['FF_SCORE_RETAIL_RESONANCE_CAPITULATION_MID'] = df['FF_SCORE_RETAIL_RESONANCE_CAPITULATION_LOW'] * s21_neg * sl21_neg * a21_neg # 创建中置信度杀跌信号
        df['FF_SCORE_RETAIL_RESONANCE_CAPITULATION_HIGH'] = df['FF_SCORE_RETAIL_RESONANCE_CAPITULATION_MID'] * s55_neg * sl55_neg # 高置信度信号基于中置信度构建
        print("               - [散户]恐慌杀跌共振信号已生成 (低/中/高置信度)") # 更新输出信息
        # --- 步骤 4: 生成散户抄底反转信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        long_term_selling = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index, ascending=False)
        mid_term_stabilizing = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index) # 获取并归一化21日斜率作为中期企稳信号
        short_term_reversing = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        short_term_accelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index)

        df['FF_SCORE_RETAIL_REVERSAL_BOTTOM_FISHING'] = long_term_selling * mid_term_stabilizing * short_term_reversing * short_term_accelerating # 强化反转信号逻辑
        print("               - [散户]抄底反转信号已生成")
        # --- 步骤 5: 生成散户顶部派发反转信号 ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        long_term_buying = self._calculate_normalized_score(metrics.get('slope_55'), norm_window, df.index)
        mid_term_stalling = self._calculate_normalized_score(metrics.get('slope_21'), norm_window, df.index, ascending=False) # 获取并归一化21日斜率(负向)作为中期失速信号
        short_term_diverging = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        short_term_decelerating = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False)

        df['FF_SCORE_RETAIL_REVERSAL_TOP_SELLING'] = long_term_buying * mid_term_stalling * short_term_diverging * short_term_decelerating # 强化反转信号逻辑
        print("               - [散户]顶部派发反转信号已生成")

        return df

    def _diagnose_flow_intensity_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【升级 V18.1 - 资金流强度动态诊断引擎】
        - 核心: 分析主力资金流强度比率(主动买盘/总主动盘)，量化交易意愿和信念。
        - 视角: 从资金流的“质量”而非“数量”出发，判断主力控盘的决心。
        - 产出: 生成“信念买入”和“信念卖出”的共振与反转信号。
        - 数据假设: 依赖数据层提供 main_force_flow_intensity_ratio_D 的全套衍生列。
        - 优化: 增强高置信度共振信号的验证条件。
        """
        print("            -> [资金流强度引擎 V1.1] 启动交叉验证...")
        periods = [5, 13, 21]
        norm_window = 120
        # --- 步骤 1: 获取资金流强度指标 ---
        metrics = {
            'static': df.get('main_force_flow_intensity_ratio_D'),
            'slope_5': df.get('SLOPE_5_main_force_flow_intensity_ratio_D'),
            'slope_13': df.get('SLOPE_13_main_force_flow_intensity_ratio_D'),
            'accel_5': df.get('ACCEL_5_main_force_flow_intensity_ratio_D'),
            'accel_13': df.get('ACCEL_13_main_force_flow_intensity_ratio_D'), # 获取13日加速度指标
        }
        # --- 步骤 2: 生成信念买入共振信号 (Conviction Buying) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s_pos = self._calculate_normalized_score(metrics.get('static'), norm_window, df.index)
        sl5_pos = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index)
        sl13_pos = self._calculate_normalized_score(metrics.get('slope_13'), norm_window, df.index)
        a5_pos = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index)
        a13_pos = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index) # 获取并归一化13日加速度

        df['FF_SCORE_INTENSITY_RESONANCE_UP_LOW'] = s_pos * sl5_pos * a5_pos
        df['FF_SCORE_INTENSITY_RESONANCE_UP_HIGH'] = df['FF_SCORE_INTENSITY_RESONANCE_UP_LOW'] * sl13_pos * a13_pos # 高置信度信号融合13日斜率与13日加速度
        print("               - [强度]信念买入共振信号已生成")
        # --- 步骤 3: 生成信念卖出共振信号 (Conviction Selling) ---
        # 在所有 _calculate_normalized_score 调用中传入 df.index
        s_neg = self._calculate_normalized_score(metrics.get('static'), norm_window, df.index, ascending=False)
        sl5_neg = self._calculate_normalized_score(metrics.get('slope_5'), norm_window, df.index, ascending=False)
        sl13_neg = self._calculate_normalized_score(metrics.get('slope_13'), norm_window, df.index, ascending=False)
        a5_neg = self._calculate_normalized_score(metrics.get('accel_5'), norm_window, df.index, ascending=False)
        a13_neg = self._calculate_normalized_score(metrics.get('accel_13'), norm_window, df.index, ascending=False) # 获取并归一化13日加速度(负向)

        df['FF_SCORE_INTENSITY_RESONANCE_DOWN_LOW'] = s_neg * sl5_neg * a5_neg
        df['FF_SCORE_INTENSITY_RESONANCE_DOWN_HIGH'] = df['FF_SCORE_INTENSITY_RESONANCE_DOWN_LOW'] * sl13_neg * a13_neg # 高置信度信号融合13日斜率与13日加速度(负向)
        print("               - [强度]信念卖出共振信号已生成")
        # --- 步骤 4: 生成买入意愿拐点信号 (Bottom Reversal) ---
        df['FF_SCORE_INTENSITY_REVERSAL_BOTTOM_HIGH'] = s_neg * sl5_pos * a5_pos
        print("               - [强度]买入意愿拐点信号已生成")
        # --- 步骤 5: 生成卖出意愿拐点信号 (Top Reversal) ---
        df['FF_SCORE_INTENSITY_REVERSAL_TOP_HIGH'] = s_pos * sl5_neg * a5_neg
        print("               - [强度]卖出意愿拐点信号已生成")
        return df











