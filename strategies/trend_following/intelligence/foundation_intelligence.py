# 文件: strategies/trend_following/intelligence/foundation_intelligence.py
# 基础情报模块 (波动率, 震荡指标)
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state

class FoundationIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化基础情报模块。
        :param strategy_instance: 策略主实例的引用，用于访问df和atomic_states。
        """
        self.strategy = strategy_instance

    def _normalize_score(self, series: pd.Series, window: int = 120, ascending: bool = True) -> pd.Series:
        """
        辅助函数：将一个Series进行滚动窗口排名归一化，生成0-1分。
        :param series: 原始数据Series。
        :param window: 归一化滚动窗口。
        :param ascending: 归一化方向，True表示值越大分数越高。
        :return: 归一化后的0-1分数Series。
        """
        min_periods = max(1, window // 5)
        rank = series.rolling(window=window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        score = rank if ascending else 1 - rank
        return score.astype(np.float32)

    def run_foundation_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V3.0 终极信号版】基础情报分析总指挥
        - 核心重构: 遵循终极信号范式，本模块不再返回一堆零散的原子信号。
                      现在只调用唯一的终极信号引擎 `diagnose_ultimate_foundation_signals`，
                      并将其产出的16个S+/S/A/B级信号作为本模块的最终输出。
        - 收益: 架构与其他情报模块完全统一，极大提升了信号质量和架构清晰度。
        """
        print("      -> [基础情报分析总指挥 V3.0 终极信号版] 启动...") # 修改: 更新版本号和描述
        
        # 直接调用终极信号引擎，并将其结果作为本模块的唯一输出
        ultimate_foundation_states = self.diagnose_ultimate_foundation_signals(self.strategy.df_indicators)

        print(f"      -> [基础情报分析总指挥 V3.0] 分析完毕，共生成 {len(ultimate_foundation_states)} 个终极基础层信号。") # 修改: 更新打印信息
        return ultimate_foundation_states

    def diagnose_ultimate_foundation_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】终极基础层信号诊断模块
        - 核心范式:
          - 1. 四大支柱: 将基础层情报提炼为EMA(趋势)、RSI(动能)、MACD(强度)、CMF(量能)四大支柱。
          - 2. 深度交叉验证: 对每一支柱，在每一时间周期上进行“静态 x 动态(斜率) x 加速”三维交叉验证，形成“已验证的周期健康分”。
          - 3. 多维共识融合: 将四大支柱在同一周期的“健康分”进行几何平均，形成代表基础层“全面共识”的“完美健康度”。
          - 4. 终极信号合成: 基于“全面共识健康度”，构建标准的S+/S/A/B四级共振与反转信号。
        - 收益: 产出经过多指标、多周期、多维度三重交叉验证的、最高质量的基础层信号。
        """
        print("        -> [终极基础层信号诊断模块 V1.0] 启动...")
        states = {}
        p_conf = get_params_block(self.strategy, 'foundation_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
            
        # --- 1. 军备检查 (Arsenal Check) ---
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        
        required_cols = set()
        for p in periods:
            required_cols.update([
                f'EMA_{p}_D' if p > 1 else 'close_D', f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D', f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D',
                'RSI_13_D', f'SLOPE_{p}_RSI_13_D', f'ACCEL_{p}_RSI_13_D',
                'MACDh_13_34_8_D', f'SLOPE_{p}_MACDh_13_34_8_D', f'ACCEL_{p}_MACDh_13_34_8_D',
                'CMF_21_D', f'SLOPE_{p}_CMF_21_D', f'ACCEL_{p}_CMF_21_D'
            ])
        
        missing_cols = list(required_cols - set(df.columns))
        if missing_cols:
            print(f"          -> [严重警告] 终极基础层引擎缺少关键数据: {sorted(missing_cols)}，模块已跳过！")
            return states

        # --- 2. 计算四大支柱在各周期的“完美健康度” ---
        pillar_health = {'EMA': {}, 'RSI': {}, 'MACD': {}, 'CMF': {}}
        
        for p in periods:
            # 2.1 EMA健康度
            ema_static_score = self._normalize_score(df[f'EMA_{p}_D' if p > 1 else 'close_D'] - df[f'EMA_{max(periods)}_D'])
            ema_slope_score = self._normalize_score(df[f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D'])
            ema_accel_score = self._normalize_score(df[f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D'])
            pillar_health['EMA'][p] = ema_static_score * ema_slope_score * ema_accel_score

            # 2.2 RSI健康度
            rsi_static_score = self._normalize_score(df['RSI_13_D'])
            rsi_slope_score = self._normalize_score(df[f'SLOPE_{p}_RSI_13_D'])
            rsi_accel_score = self._normalize_score(df[f'ACCEL_{p}_RSI_13_D'])
            pillar_health['RSI'][p] = rsi_static_score * rsi_slope_score * rsi_accel_score

            # 2.3 MACD健康度
            macd_static_score = self._normalize_score(df['MACDh_13_34_8_D'])
            macd_slope_score = self._normalize_score(df[f'SLOPE_{p}_MACDh_13_34_8_D'])
            macd_accel_score = self._normalize_score(df[f'ACCEL_{p}_MACDh_13_34_8_D'])
            pillar_health['MACD'][p] = macd_static_score * macd_slope_score * macd_accel_score

            # 2.4 CMF健康度
            cmf_static_score = self._normalize_score(df['CMF_21_D'])
            cmf_slope_score = self._normalize_score(df[f'SLOPE_{p}_CMF_21_D'])
            cmf_accel_score = self._normalize_score(df[f'ACCEL_{p}_CMF_21_D'])
            pillar_health['CMF'][p] = cmf_static_score * cmf_slope_score * cmf_accel_score

        # --- 3. 融合生成“全面共识健康度” ---
        overall_bullish_health = {}
        for p in periods:
            overall_bullish_health[p] = (pillar_health['EMA'][p] * pillar_health['RSI'][p] * pillar_health['MACD'][p] * pillar_health['CMF'][p])**0.25
        
        overall_bearish_health = {p: 1.0 - overall_bullish_health[p] for p in periods}

        # --- 4. 定义信号组件 ---
        bullish_short_force = (overall_bullish_health[1] * overall_bullish_health[5])**0.5
        bullish_medium_trend = (overall_bullish_health[13] * overall_bullish_health[21])**0.5
        bullish_long_inertia = overall_bullish_health[55]
        
        bearish_short_force = (overall_bearish_health[1] * overall_bearish_health[5])**0.5
        bearish_medium_trend = (overall_bearish_health[13] * overall_bearish_health[21])**0.5
        bearish_long_inertia = overall_bearish_health[55]

        # --- 5. 共振信号合成 ---
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_B'] = overall_bullish_health[5].astype(np.float32)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_A'] = (overall_bullish_health[5] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_S'] = (bullish_short_force * bullish_medium_trend).astype(np.float32)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_S_PLUS'] = (states['SCORE_FOUNDATION_BULLISH_RESONANCE_S'] * bullish_long_inertia).astype(np.float32)
        
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_B'] = overall_bearish_health[5].astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_A'] = (overall_bearish_health[5] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_S'] = (bearish_short_force * bearish_medium_trend).astype(np.float32)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_S_PLUS'] = (states['SCORE_FOUNDATION_BEARISH_RESONANCE_S'] * bearish_long_inertia).astype(np.float32)

        # --- 6. 反转信号合成 ---
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_B'] = (overall_bullish_health[1] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_A'] = (overall_bullish_health[5] * overall_bearish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S'] = (bullish_short_force * bearish_long_inertia).astype(np.float32)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S_PLUS'] = (bullish_short_force * bearish_medium_trend * bearish_long_inertia).astype(np.float32)
        
        states['SCORE_FOUNDATION_TOP_REVERSAL_B'] = (overall_bearish_health[1] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_TOP_REVERSAL_A'] = (overall_bearish_health[5] * overall_bullish_health[21]).astype(np.float32)
        states['SCORE_FOUNDATION_TOP_REVERSAL_S'] = (bearish_short_force * bullish_long_inertia).astype(np.float32)
        states['SCORE_FOUNDATION_TOP_REVERSAL_S_PLUS'] = (bearish_short_force * bearish_medium_trend * bearish_long_inertia).astype(np.float32)
        
        print(f"        -> [终极基础层信号诊断模块 V1.0] 分析完毕，生成 {len(states)} 个终极信号。")
        return states

    def diagnose_ema_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 终极交叉验证版】EMA均线协同诊断模块
        - 核心重构 (本次修改):
          - [交叉验证] 引入“静态(排列) x 动态(斜率) x 加速”三维交叉验证，生成B/A/S三级置信度信号。
          - [1日维度] 将1日周期的斜率和加速度纳入计算，捕捉最即时的边际变化。
        - 核心逻辑:
          - 共振: B级(短期斜率共振) -> A级(B级+静态排列确认) -> S级(A级+短期加速确认)。
          - 反转: B级(短期加速拐点) -> A级(B级+长期趋势环境确认) -> S级(A级+短期斜率确认)。
        - 收益: 信号逻辑更严谨，能更精确地识别趋势的形成、持续与转折。
        """
        states = {}
        p = get_params_block(self.strategy, 'multi_dim_ma_params')
        if not get_param_value(p.get('enabled'), True): return {}
        
        # --- 1. 军备检查 ---
        periods = get_param_value(p.get('ma_periods'), [1, 5, 13, 21, 55])
        required_cols = [f'EMA_{p}_D' if p > 1 else 'close_D' for p in periods] # 1日EMA即为收盘价
        required_cols.extend([f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D' for p in periods])
        required_cols.extend([f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D' for p in periods])
        
        if not all(c in df.columns for c in required_cols):
            print(f"          -> [警告] EMA协同模块缺少依赖，模块已跳过。")
            return states
        # --- 2. 核心要素数值化 ---
        # 2.1 静态排列分 (多头排列程度)
        alignment_scores = []
        for i in range(len(periods) - 1):
            # 适配1日周期使用close_D
            short_col = f'EMA_{periods[i]}_D' if periods[i] > 1 else 'close_D'
            long_col = f'EMA_{periods[i+1]}_D'
            alignment_scores.append((df[short_col] > df[long_col]).astype(float))
        score_static_bullish = pd.Series(np.mean(alignment_scores, axis=0), index=df.index)
        # 2.2 动态斜率分
        slope_scores = {p: self._normalize_score(df[f'SLOPE_{p}_EMA_{p}_D' if p > 1 else 'SLOPE_1_close_D']) for p in periods}
        # 2.3 动态加速分
        accel_scores = {p: self._normalize_score(df[f'ACCEL_{p}_EMA_{p}_D' if p > 1 else 'ACCEL_1_close_D']) for p in periods}
        # --- 3. 上升/下跌共振信号 ---
        # 严格遵循 B/A/S 逻辑
        # B级: 短中期斜率共振
        bullish_momentum_b = (slope_scores[1] * slope_scores[5] * slope_scores[13]).astype(np.float32)
        states['SCORE_EMA_BULLISH_RESONANCE_B'] = bullish_momentum_b
        # A级: B级信号 + 静态排列确认
        bullish_momentum_a = (bullish_momentum_b * score_static_bullish).astype(np.float32)
        states['SCORE_EMA_BULLISH_RESONANCE_A'] = bullish_momentum_a
        # S级: A级信号 + 加速确认
        bullish_accel_s = (accel_scores[1] * accel_scores[5]).astype(np.float32)
        states['SCORE_EMA_BULLISH_RESONANCE_S'] = (bullish_momentum_a * bullish_accel_s).astype(np.float32)
        # 对称的下跌共振逻辑
        bearish_momentum_b = ((1 - slope_scores[1]) * (1 - slope_scores[5]) * (1 - slope_scores[13])).astype(np.float32)
        states['SCORE_EMA_BEARISH_RESONANCE_B'] = bearish_momentum_b
        bearish_momentum_a = (bearish_momentum_b * (1 - score_static_bullish)).astype(np.float32)
        states['SCORE_EMA_BEARISH_RESONANCE_A'] = bearish_momentum_a
        bearish_accel_s = ((1 - accel_scores[1]) * (1 - accel_scores[5])).astype(np.float32)
        states['SCORE_EMA_BEARISH_RESONANCE_S'] = (bearish_momentum_a * bearish_accel_s).astype(np.float32)
        # --- 4. 顶部/底部反转信号 ---
        # 严格遵循 B/A/S 逻辑
        # B级: 短期加速拐点
        bottom_trigger_b = accel_scores[1].astype(np.float32)
        states['SCORE_EMA_BOTTOM_REVERSAL_B'] = bottom_trigger_b
        # A级: B级信号 + 长期趋势环境确认
        long_term_down_env = (1 - slope_scores[55])
        bottom_trigger_a = (bottom_trigger_b * long_term_down_env).astype(np.float32)
        states['SCORE_EMA_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        # S级: A级信号 + 短期斜率确认
        short_term_mom_up = slope_scores[5]
        states['SCORE_EMA_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * short_term_mom_up).astype(np.float32)

        # 对称的顶部反转逻辑
        top_trigger_b = (1 - accel_scores[1]).astype(np.float32)
        states['SCORE_EMA_TOP_REVERSAL_B'] = top_trigger_b
        long_term_up_env = slope_scores[55]
        top_trigger_a = (top_trigger_b * long_term_up_env).astype(np.float32)
        states['SCORE_EMA_TOP_REVERSAL_A'] = top_trigger_a
        short_term_mom_down = (1 - slope_scores[5])
        states['SCORE_EMA_TOP_REVERSAL_S'] = (top_trigger_a * short_term_mom_down).astype(np.float32)
        
        return states

    def diagnose_foundation_synergy(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 共识引擎版】基础层最终协同诊断模块
        - 核心重构 (本次修改):
          - [逻辑升级] 废除V1.0的“或”门逻辑(np.maximum)，升级为基于加权平均的“共识”逻辑。
          - [信号升级] 生成全新的、代表多领域共识的S级信号，如`SCORE_FOUNDATION_BULLISH_RESONANCE_S`。
          - [S+信号] 新增S+级别信号，将S级共识与关键外部条件（如波动率）交叉验证，产生最高置信度信号。
        - 核心逻辑:
          - 对来自各子模块（EMA, RSI, MACD等）的同类型S级信号进行加权平均。
          - 一个信号的最终分数，取决于有多少个不同维度的指标在同时为它“投票”，以及每个“投票”的权重。
        - 收益: 极大提升了最终输出信号的置信度，能有效识别出得到市场多方面验证的交易机会。
        """
        # 此方法为全新的顶层融合模块
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        def get_weighted_synergy_score(signal_weights: Dict[str, float]) -> pd.Series:
            """辅助函数，用于计算加权协同分数"""
            total_score = pd.Series(0.0, index=df.index)
            total_weight = 0.0
            for signal, weight in signal_weights.items():
                score_series = atomic.get(signal, default_score)
                total_score += score_series * weight
                total_weight += weight
            return total_score / total_weight if total_weight > 0 else default_score

        # --- 1. 上升共振协同 (Bullish Resonance Synergy) ---
        bullish_resonance_sources = {
            'SCORE_EMA_BULLISH_RESONANCE_S': 0.4,
            'SCORE_RSI_BULLISH_RESONANCE_S': 0.3,
            'SCORE_MACD_BULLISH_RESONANCE_S': 0.2,
            'SCORE_CMF_BULLISH_RESONANCE_S': 0.1,
        }
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_S'] = get_weighted_synergy_score(bullish_resonance_sources)

        # --- 2. 下跌共振协同 (Bearish Resonance Synergy) ---
        bearish_resonance_sources = {
            'SCORE_EMA_BEARISH_RESONANCE_S': 0.4,
            'SCORE_RSI_BEARISH_RESONANCE_S': 0.3,
            'SCORE_MACD_BEARISH_RESONANCE_S': 0.2,
            'SCORE_CMF_BEARISH_RESONANCE_S': 0.1,
        }
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_S'] = get_weighted_synergy_score(bearish_resonance_sources)

        # --- 3. 底部反转协同 (Bottom Reversal Synergy) ---
        bottom_reversal_sources = {
            'SCORE_EMA_BOTTOM_REVERSAL_S': 0.35,
            'SCORE_RSI_BOTTOM_REVERSAL_S': 0.35,
            'SCORE_MACD_BOTTOM_REVERSAL_S': 0.2,
            'SCORE_VOL_TIPPING_POINT_BOTTOM_OPP': 0.1,
        }
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S'] = get_weighted_synergy_score(bottom_reversal_sources)

        # --- 4. 顶部反转协同 (Top Reversal Synergy) ---
        top_reversal_sources = {
            'SCORE_EMA_TOP_REVERSAL_S': 0.35,
            'SCORE_RSI_TOP_REVERSAL_S': 0.35,
            'SCORE_MACD_TOP_REVERSAL_S': 0.2,
            'SCORE_VOL_TIPPING_POINT_TOP_RISK': 0.1,
        }
        states['SCORE_FOUNDATION_TOP_REVERSAL_S'] = get_weighted_synergy_score(top_reversal_sources)

        # --- 5. S+ 信号生成 (Synergy-Plus)，代表最高置信度的机会 ---
        # S+ 信号定义: S级共识信号与关键外部条件（如波动率、市场状态）的乘积。
        
        # 5.1 上升共振 S+
        # 定义：S级上升共振 + S级突破潜力（波动率压缩+趋势环境），捕捉“共振突破”的黄金时刻。
        bullish_resonance_s = states['SCORE_FOUNDATION_BULLISH_RESONANCE_S']
        breakout_potential_s = atomic.get('SCORE_VOL_BREAKOUT_POTENTIAL_S', default_score)
        states['SCORE_FOUNDATION_BULLISH_RESONANCE_S_PLUS'] = (bullish_resonance_s * breakout_potential_s).astype(np.float32)

        # 5.2 下跌共振 S+
        # 定义：S级下跌共振 + S级破位风险（波动率扩张+非趋势环境），捕捉“共振破位”的危险信号。
        bearish_resonance_s = states['SCORE_FOUNDATION_BEARISH_RESONANCE_S']
        breakdown_risk_s = atomic.get('SCORE_VOL_BREAKDOWN_RISK_S', default_score)
        states['SCORE_FOUNDATION_BEARISH_RESONANCE_S_PLUS'] = (bearish_resonance_s * breakdown_risk_s).astype(np.float32)

        # 5.3 底部反转 S+
        # 定义：S级底部反转共识 + 波动率扩张引爆点，捕捉“反转启动+波动率放大”的强力信号。
        bottom_reversal_s = states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S']
        vol_ignition_opp = atomic.get('SCORE_VOL_TIPPING_POINT_BOTTOM_OPP', default_score)
        states['SCORE_FOUNDATION_BOTTOM_REVERSAL_S_PLUS'] = (bottom_reversal_s * vol_ignition_opp).astype(np.float32)

        # 5.4 顶部反转 S+
        # 定义：S级顶部反转共识 + 波动率衰竭风险点，捕捉“反转确认+波动率收缩”的高危信号。
        top_reversal_s = states['SCORE_FOUNDATION_TOP_REVERSAL_S']
        vol_exhaustion_risk = atomic.get('SCORE_VOL_TIPPING_POINT_TOP_RISK', default_score)
        states['SCORE_FOUNDATION_TOP_REVERSAL_S_PLUS'] = (top_reversal_s * vol_exhaustion_risk).astype(np.float32)

        return states

    def diagnose_oscillator_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 终极交叉验证版】震荡与动能统一情报中心
        - 核心重构 (本次修改):
          - [交叉验证] 引入“静态(RSI区间) x 动态(斜率) x 加速”三维交叉验证。
          - [1日维度] 将1日周期的斜率和加速度纳入计算，捕捉最即时的边际变化。
        - 核心逻辑:
          - 共振: B级(短期斜率) -> A级(B级+静态区间确认) -> S级(A级+短期加速确认)。
          - 反转: B级(短期加速拐点) -> A级(B级+超卖/超买环境确认) -> S级(A级+短期斜率确认)。
        - 收益: 极大提升了RSI信号的可靠性，能有效过滤在“钝化区”的假信号。
        """
        states = {}
        p = get_params_block(self.strategy, 'oscillator_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 1. 军备检查 ---
        required_cols = [
            'RSI_13_D', 'SLOPE_1_RSI_13_D', 'SLOPE_5_RSI_13_D', 'ACCEL_1_RSI_13_D', 'ACCEL_5_RSI_13_D',
            'RSI_13_W', 'SLOPE_5_RSI_13_W',
            'MACD_HIST_ZSCORE_D', 'BIAS_55_D', 'high_D', 'low_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 震荡与动能情报中心缺少必需列: {missing}，模块已跳过。")
            # 增加对数据层缺失的提示
            print(f"          -> [数据需求] 请确保数据工程层已为 RSI_13_D 计算了 1日和5日的斜率与加速度。")
            return states

        # --- 2. 基础静态诊断 (逻辑不变) ---
        rsi_col = 'RSI_13_D'
        overbought_threshold = get_param_value(p.get('rsi_overbought_start'), 70)
        score_overbought = (df[rsi_col] - overbought_threshold) / (100 - overbought_threshold)
        states['SCORE_RSI_OVERBOUGHT_EXTENT'] = score_overbought.clip(0, 1).astype(np.float32)
        oversold_threshold = get_param_value(p.get('rsi_oversold_start'), 30)
        score_oversold = (oversold_threshold - df[rsi_col]) / (oversold_threshold - 0)
        states['SCORE_RSI_OVERSOLD_EXTENT'] = score_oversold.clip(0, 1).astype(np.float32)
        # ... (BIAS 和 MACD Divergence 逻辑保持不变) ...
        p_bias = p.get('bias_dynamic_threshold', {})
        window = get_param_value(p_bias.get('window'), 120)
        quantile = get_param_value(p_bias.get('quantile'), 0.1)
        dynamic_oversold_threshold = df['BIAS_55_D'].rolling(window=window).quantile(quantile)
        negative_deviation = (dynamic_oversold_threshold - df['BIAS_55_D']).clip(lower=0)
        states['SCORE_BIAS_OVERSOLD_EXTENT'] = self._normalize_score(negative_deviation, window=window)
        dynamic_overbought_threshold = df['BIAS_55_D'].rolling(window=window).quantile(1 - quantile)
        positive_deviation = (df['BIAS_55_D'] - dynamic_overbought_threshold).clip(lower=0)
        states['SCORE_BIAS_OVERBOUGHT_EXTENT'] = self._normalize_score(positive_deviation, window=window)
        price_overshoot = (df['high_D'] - df['high_D'].rolling(10).max().shift(1)).clip(lower=0)
        macd_undershoot = (df['MACD_HIST_ZSCORE_D'].rolling(10).max().shift(1) - df['MACD_HIST_ZSCORE_D']).clip(lower=0)
        states['SCORE_MACD_BEARISH_DIVERGENCE_RISK'] = self._normalize_score(price_overshoot) * self._normalize_score(macd_undershoot)
        price_undershoot = (df['low_D'].rolling(10).min().shift(1) - df['low_D']).clip(lower=0)
        macd_overshoot = (df['MACD_HIST_ZSCORE_D'] - df['MACD_HIST_ZSCORE_D'].rolling(10).min().shift(1)).clip(lower=0)
        states['SCORE_MACD_BULLISH_DIVERGENCE_OPP'] = self._normalize_score(price_undershoot) * self._normalize_score(macd_overshoot)

        # --- 3. RSI协同诊断单元 ---
        # 全面重构RSI信号生成逻辑
        # 3.1 核心要素数值化
        score_rsi_d_static_bull = self._normalize_score(df['RSI_13_D'].clip(50, 100))
        score_rsi_d_static_bear = self._normalize_score(df['RSI_13_D'].clip(0, 50), ascending=False)
        score_rsi_d_mom_short = self._normalize_score(df['SLOPE_1_RSI_13_D'])
        score_rsi_d_mom_mid = self._normalize_score(df['SLOPE_5_RSI_13_D'])
        score_rsi_d_accel_short = self._normalize_score(df['ACCEL_1_RSI_13_D'])
        score_rsi_w_mom = self._normalize_score(df['SLOPE_5_RSI_13_W'])

        # 3.2 上升共振 (Bullish Resonance)
        bullish_momentum_b = score_rsi_d_mom_mid
        states['SCORE_RSI_BULLISH_RESONANCE_B'] = bullish_momentum_b.astype(np.float32)
        bullish_momentum_a = (bullish_momentum_b * score_rsi_d_static_bull * score_rsi_w_mom).astype(np.float32)
        states['SCORE_RSI_BULLISH_RESONANCE_A'] = bullish_momentum_a
        states['SCORE_RSI_BULLISH_RESONANCE_S'] = (bullish_momentum_a * score_rsi_d_accel_short).astype(np.float32)

        # 3.3 下跌共振 (Bearish Resonance)
        bearish_momentum_b = (1 - score_rsi_d_mom_mid)
        states['SCORE_RSI_BEARISH_RESONANCE_B'] = bearish_momentum_b.astype(np.float32)
        bearish_momentum_a = (bearish_momentum_b * score_rsi_d_static_bear * (1 - score_rsi_w_mom)).astype(np.float32)
        states['SCORE_RSI_BEARISH_RESONANCE_A'] = bearish_momentum_a
        states['SCORE_RSI_BEARISH_RESONANCE_S'] = (bearish_momentum_a * (1 - score_rsi_d_accel_short)).astype(np.float32)

        # 3.4 底部反转 (Bottom Reversal)
        bottom_trigger_b = score_rsi_d_accel_short
        states['SCORE_RSI_BOTTOM_REVERSAL_B'] = bottom_trigger_b.astype(np.float32)
        bottom_trigger_a = (bottom_trigger_b * states['SCORE_RSI_OVERSOLD_EXTENT']).astype(np.float32)
        states['SCORE_RSI_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        states['SCORE_RSI_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * score_rsi_d_mom_short).astype(np.float32)

        # 3.5 顶部反转 (Top Reversal)
        top_trigger_b = (1 - score_rsi_d_accel_short)
        states['SCORE_RSI_TOP_REVERSAL_B'] = top_trigger_b.astype(np.float32)
        top_trigger_a = (top_trigger_b * states['SCORE_RSI_OVERBOUGHT_EXTENT']).astype(np.float32)
        states['SCORE_RSI_TOP_REVERSAL_A'] = top_trigger_a
        states['SCORE_RSI_TOP_REVERSAL_S'] = (top_trigger_a * (1 - score_rsi_d_mom_short)).astype(np.float32)
        
        return states

    def diagnose_volatility_intelligence(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 交叉验证版】波动率统一情报中心
        - 核心逻辑:
          - B级 (日线压缩): 基于日线BBW的归一化评分。
          - A级 (跨周期压缩): B级信号与周线BBW评分交叉验证。
          - S级 (静态压缩): A级信号与BBW斜率为负（仍在压缩）交叉验证。
        - 收益: 能够更精确地量化波动率状态，区分“初步压缩”和“极致压缩”。
        """
        states = {}
        p = get_params_block(self.strategy, 'volatility_state_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'BBW_21_2.0_D', 'SLOPE_5_BBW_21_2.0_D', 'ACCEL_5_BBW_21_2.0_D', 
            'BBW_21_2.0_W', 'hurst_120d_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 波动率情报中心缺少必需列: {missing}，模块已跳过。")
            return states

        # --- 2. 核心要素数值化 ---
        score_squeeze_daily = self._normalize_score(df['BBW_21_2.0_D'], ascending=False)
        score_squeeze_weekly = self._normalize_score(df['BBW_21_2.0_W'], ascending=False)
        score_squeeze_momentum = self._normalize_score(df['SLOPE_5_BBW_21_2.0_D'], ascending=False)
        
        score_expansion_daily = 1 - score_squeeze_daily
        score_expansion_weekly = 1 - score_squeeze_weekly
        score_expansion_momentum = 1 - score_squeeze_momentum

        # --- 3. 生成B/A/S三级压缩信号 ---
        states['SCORE_VOL_COMPRESSION_B'] = score_squeeze_daily
        states['SCORE_VOL_COMPRESSION_A'] = (score_squeeze_daily * score_squeeze_weekly).astype(np.float32)
        states['SCORE_VOL_COMPRESSION_S'] = (states['SCORE_VOL_COMPRESSION_A'] * score_squeeze_momentum).astype(np.float32)

        # --- 4. 生成B/A/S三级扩张信号 (对称逻辑) ---
        states['SCORE_VOL_EXPANSION_B'] = score_expansion_daily
        states['SCORE_VOL_EXPANSION_A'] = (score_expansion_daily * score_expansion_weekly).astype(np.float32)
        states['SCORE_VOL_EXPANSION_S'] = (states['SCORE_VOL_EXPANSION_A'] * score_expansion_momentum).astype(np.float32)

        # --- 5. 波动率反转临界点 (逻辑优化) ---
        is_tipping_point_bottom = (df['SLOPE_5_BBW_21_2.0_D'] > 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) <= 0)
        # 使用最高置信度的S级压缩分作为环境判断
        states['SCORE_VOL_TIPPING_POINT_BOTTOM_OPP'] = (states['SCORE_VOL_COMPRESSION_S'] * is_tipping_point_bottom).astype(np.float32)
        
        is_tipping_point_top = (df['SLOPE_5_BBW_21_2.0_D'] < 0) & (df['SLOPE_5_BBW_21_2.0_D'].shift(1) >= 0)
        # 使用最高置信度的S级扩张分作为环境判断
        states['SCORE_VOL_TIPPING_POINT_TOP_RISK'] = (states['SCORE_VOL_EXPANSION_S'] * is_tipping_point_top).astype(np.float32)

        # --- 6. 市场政权与数值化评分 (逻辑不变) ---
        hurst_score = self._normalize_score(df['hurst_120d_D'])
        states['SCORE_TRENDING_REGIME'] = hurst_score
        states['SCORE_VOL_BREAKOUT_POTENTIAL_S'] = states['SCORE_VOL_COMPRESSION_S'] * hurst_score
        states['SCORE_VOL_BREAKDOWN_RISK_S'] = states['SCORE_VOL_EXPANSION_S'] * (1 - hurst_score)
        
        return states

    def diagnose_market_character_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 交叉验证版】市场特征与情绪统一情报中心
        - 核心重构 (本次修改):
          - [交叉验证] 引入“静态(情绪高/低) x 动态(斜率) x 加速”三维交叉验证。
          - [信号升级] 生成全新的B/A/S三级市场情绪共振与反转信号。
        - 收益: 对市场情绪的判断更具前瞻性和可靠性。
        """
        states = {}
        p = get_params_block(self.strategy, 'market_character_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'total_winner_rate_D', 'SLOPE_1_total_winner_rate_D', 'SLOPE_5_total_winner_rate_D', 
            'SLOPE_21_total_winner_rate_D', 'ACCEL_1_total_winner_rate_D', 'ACCEL_5_total_winner_rate_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 市场特征情报中心缺少必需列: {missing}，模块已跳过。")
            # 增加对数据层缺失的精确提示，使其更具可操作性
            print(f"          -> [数据需求] 请数据工程层为 'total_winner_rate_D' 指标补充以下缺失的衍生列: {missing}。")
            print(f"          -> [提示] 衍生列计算方法：'SLOPE_n_...' 为n日斜率, 'ACCEL_n_...' 为n日斜率的加速度。")
            return states

        # --- 2. 市场情绪协同诊断单元 ---
        # 2.1 核心要素数值化
        winner_rate = df['total_winner_rate_D']
        score_static_high = self._normalize_score(winner_rate.clip(lower=winner_rate.rolling(120).quantile(0.8)))
        score_static_low = self._normalize_score(winner_rate.clip(upper=winner_rate.rolling(120).quantile(0.2)), ascending=False)
        score_mom_short = self._normalize_score(df['SLOPE_1_total_winner_rate_D'])
        score_mom_mid = self._normalize_score(df['SLOPE_5_total_winner_rate_D'])
        score_accel_short = self._normalize_score(df['ACCEL_1_total_winner_rate_D'])

        # 2.2 上升共振 (情绪升温)
        bullish_momentum_b = (score_mom_short * score_mom_mid).astype(np.float32)
        states['SCORE_MKT_BULLISH_RESONANCE_B'] = bullish_momentum_b
        bullish_momentum_a = (bullish_momentum_b * self._normalize_score(winner_rate)).astype(np.float32) # A级用原始静态分确认
        states['SCORE_MKT_BULLISH_RESONANCE_A'] = bullish_momentum_a
        states['SCORE_MKT_BULLISH_RESONANCE_S'] = (bullish_momentum_a * score_accel_short).astype(np.float32)

        # 2.3 下跌共振 (情绪降温)
        bearish_momentum_b = ((1 - score_mom_short) * (1 - score_mom_mid)).astype(np.float32)
        states['SCORE_MKT_BEARISH_RESONANCE_B'] = bearish_momentum_b
        bearish_momentum_a = (bearish_momentum_b * (1 - self._normalize_score(winner_rate))).astype(np.float32)
        states['SCORE_MKT_BEARISH_RESONANCE_A'] = bearish_momentum_a
        states['SCORE_MKT_BEARISH_RESONANCE_S'] = (bearish_momentum_a * (1 - score_accel_short)).astype(np.float32)

        # 2.4 底部反转 (情绪冰点反转)
        bottom_trigger_b = score_accel_short
        states['SCORE_MKT_BOTTOM_REVERSAL_B'] = bottom_trigger_b.astype(np.float32)
        bottom_trigger_a = (bottom_trigger_b * score_static_low).astype(np.float32)
        states['SCORE_MKT_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        states['SCORE_MKT_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * score_mom_short).astype(np.float32)

        # 2.5 顶部反转 (情绪高点反转)
        top_trigger_b = (1 - score_accel_short)
        states['SCORE_MKT_TOP_REVERSAL_B'] = top_trigger_b.astype(np.float32)
        top_trigger_a = (top_trigger_b * score_static_high).astype(np.float32)
        states['SCORE_MKT_TOP_REVERSAL_A'] = top_trigger_a
        states['SCORE_MKT_TOP_REVERSAL_S'] = (top_trigger_a * (1 - score_mom_short)).astype(np.float32)

        # --- 3. 综合市场健康分 (逻辑优化) ---
        # 使用新的共振信号来计算健康分
        dynamic_score = states['SCORE_MKT_BULLISH_RESONANCE_A'] - states['SCORE_MKT_BEARISH_RESONANCE_A']
        normalized_dynamic_score = (dynamic_score + 1) / 2
        states['SCORE_MKT_HEALTH_S'] = (
            self._normalize_score(winner_rate) * 0.4 + normalized_dynamic_score * 0.6
        ).astype(np.float32)
        return states

    def diagnose_capital_and_range_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 交叉验证版】资金流与绝对波幅统一情报中心
        - 核心重构 (本次修改):
          - [交叉验证] 引入“静态(净流入) x 动态(斜率) x 加速”三维交叉验证。
          - [信号升级] 生成全新的B/A/S三级CMF共振与反转信号。
        - 收益: 极大提升了资金流信号的可靠性，能有效过滤噪音。
        """
        states = {}
        p_capital = get_params_block(self.strategy, 'capital_state_params')
        if not get_param_value(p_capital.get('enabled'), False): return states
        
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'CMF_21_D', 'SLOPE_1_CMF_21_D', 'SLOPE_5_CMF_21_D', 'SLOPE_21_CMF_21_D', 
            'ACCEL_1_CMF_21_D', 'ACCEL_5_CMF_21_D', 'ATR_14_D', 'SLOPE_5_ATR_14_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 资金流情报中心缺少必需列: {missing}，模块已跳过。")
            # 增加对数据层缺失的提示
            print(f"          -> [数据需求] 请确保数据工程层已为 CMF_21_D 计算了 1, 5, 21日的斜率与加速度。")
            return states

        # --- 2. CMF协同诊断单元 ---
        # 2.1 核心要素数值化
        score_cmf_static_bull = self._normalize_score(df['CMF_21_D'].clip(lower=0))
        score_cmf_static_bear = self._normalize_score(df['CMF_21_D'].clip(upper=0), ascending=False)
        score_cmf_mom_short = self._normalize_score(df['SLOPE_1_CMF_21_D'])
        score_cmf_mom_mid = self._normalize_score(df['SLOPE_5_CMF_21_D'])
        score_cmf_accel_short = self._normalize_score(df['ACCEL_1_CMF_21_D'])

        # 2.2 上升共振 (Bullish Resonance)
        bullish_momentum_b = (score_cmf_mom_short * score_cmf_mom_mid).astype(np.float32)
        states['SCORE_CMF_BULLISH_RESONANCE_B'] = bullish_momentum_b
        bullish_momentum_a = (bullish_momentum_b * score_cmf_static_bull).astype(np.float32)
        states['SCORE_CMF_BULLISH_RESONANCE_A'] = bullish_momentum_a
        states['SCORE_CMF_BULLISH_RESONANCE_S'] = (bullish_momentum_a * score_cmf_accel_short).astype(np.float32)

        # 2.3 下跌共振 (Bearish Resonance)
        bearish_momentum_b = ((1 - score_cmf_mom_short) * (1 - score_cmf_mom_mid)).astype(np.float32)
        states['SCORE_CMF_BEARISH_RESONANCE_B'] = bearish_momentum_b
        bearish_momentum_a = (bearish_momentum_b * score_cmf_static_bear).astype(np.float32)
        states['SCORE_CMF_BEARISH_RESONANCE_A'] = bearish_momentum_a
        states['SCORE_CMF_BEARISH_RESONANCE_S'] = (bearish_momentum_a * (1 - score_cmf_accel_short)).astype(np.float32)

        # 2.4 底部反转 (Bottom Reversal)
        bottom_trigger_b = score_cmf_accel_short
        states['SCORE_CMF_BOTTOM_REVERSAL_B'] = bottom_trigger_b.astype(np.float32)
        bottom_trigger_a = (bottom_trigger_b * score_cmf_static_bear).astype(np.float32)
        states['SCORE_CMF_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        states['SCORE_CMF_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * score_cmf_mom_short).astype(np.float32)

        # 2.5 顶部反转 (Top Reversal)
        top_trigger_b = (1 - score_cmf_accel_short)
        states['SCORE_CMF_TOP_REVERSAL_B'] = top_trigger_b.astype(np.float32)
        top_trigger_a = (top_trigger_b * score_cmf_static_bull).astype(np.float32)
        states['SCORE_CMF_TOP_REVERSAL_A'] = top_trigger_a
        states['SCORE_CMF_TOP_REVERSAL_S'] = (top_trigger_a * (1 - score_cmf_mom_short)).astype(np.float32)

        # --- 3. ATR 绝对波幅状态与临界点信号 (逻辑不变) ---
        atr = df['ATR_14_D']
        score_atr_compression = self._normalize_score(atr, ascending=False)
        score_atr_expansion = self._normalize_score(atr, ascending=True)
        states['SCORE_ATR_COMPRESSION_LEVEL'] = score_atr_compression
        states['SCORE_ATR_EXPANSION_LEVEL'] = score_atr_expansion
        is_tipping_point_expansion = (df['SLOPE_5_ATR_14_D'] > 0) & (df['SLOPE_5_ATR_14_D'].shift(1) <= 0)
        states['SCORE_ATR_EXPANSION_IGNITION_OPP'] = score_atr_compression * is_tipping_point_expansion.astype(np.float32)
        is_tipping_point_exhaustion = (df['SLOPE_5_ATR_14_D'] < 0) & (df['SLOPE_5_ATR_14_D'].shift(1) >= 0)
        states['SCORE_ATR_EXPANSION_EXHAUSTION_RISK'] = score_atr_expansion * is_tipping_point_exhaustion.astype(np.float32)
        return states

    def diagnose_classic_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 交叉验证版】经典指标统一情报中心
        - 核心重构 (本次修改):
          - [交叉验证] 引入MACD“静态(柱值) x 动态(斜率) x 加速”三维交叉验证。
          - [信号升级] 生成全新的B/A/S三级MACD共振与反转信号。
        - 收益: 极大提升了MACD信号的可靠性，能有效过滤假金叉/死叉。
        """
        states = {}
        p = get_params_block(self.strategy, 'classic_indicator_params')
        if not get_param_value(p.get('enabled'), True): return states
        
        # --- 1. 军备检查 (Arsenal Check) ---
        required_cols = [
            'MACDh_13_34_8_D', 'MACD_HIST_ZSCORE_D', 'SLOPE_1_MACDh_13_34_8_D', 
            'SLOPE_5_MACDh_13_34_8_D', 'ACCEL_1_MACDh_13_34_8_D',
            'volume_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D', 'close_D', 'open_D'
        ]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            print(f"          -> [警告] 经典指标情报中心缺少必需列: {missing}，模块已跳过。")
            # 增加对数据层缺失的提示
            print(f"          -> [数据需求] 请确保数据工程层已为 MACDh_13_34_8_D 计算了 1日和5日的斜率与加速度。")
            return states

        # --- 2. MACD协同诊断单元 ---
        # 2.1 核心要素数值化
        score_macdh_static_bull = self._normalize_score(df['MACDh_13_34_8_D'].clip(lower=0))
        score_macdh_static_bear = self._normalize_score(df['MACDh_13_34_8_D'].clip(upper=0), ascending=False)
        score_macdh_zscore_high = self._normalize_score(df['MACD_HIST_ZSCORE_D'].clip(lower=1.5))
        score_macdh_zscore_low = self._normalize_score(df['MACD_HIST_ZSCORE_D'].clip(upper=-1.5), ascending=False)
        score_mom_short = self._normalize_score(df['SLOPE_1_MACDh_13_34_8_D'])
        score_mom_mid = self._normalize_score(df['SLOPE_5_MACDh_13_34_8_D'])
        score_accel_short = self._normalize_score(df['ACCEL_1_MACDh_13_34_8_D'])

        # 2.2 上升共振 (多头动能)
        bullish_momentum_b = (score_mom_short * score_mom_mid).astype(np.float32)
        states['SCORE_MACD_BULLISH_RESONANCE_B'] = bullish_momentum_b
        bullish_momentum_a = (bullish_momentum_b * score_macdh_static_bull).astype(np.float32)
        states['SCORE_MACD_BULLISH_RESONANCE_A'] = bullish_momentum_a
        states['SCORE_MACD_BULLISH_RESONANCE_S'] = (bullish_momentum_a * score_accel_short).astype(np.float32)

        # 2.3 下跌共振 (空头动能)
        bearish_momentum_b = ((1 - score_mom_short) * (1 - score_mom_mid)).astype(np.float32)
        states['SCORE_MACD_BEARISH_RESONANCE_B'] = bearish_momentum_b
        bearish_momentum_a = (bearish_momentum_b * score_macdh_static_bear).astype(np.float32)
        states['SCORE_MACD_BEARISH_RESONANCE_A'] = bearish_momentum_a
        states['SCORE_MACD_BEARISH_RESONANCE_S'] = (bearish_momentum_a * (1 - score_accel_short)).astype(np.float32)

        # 2.4 底部反转 (金叉)
        bottom_trigger_b = score_accel_short
        states['SCORE_MACD_BOTTOM_REVERSAL_B'] = bottom_trigger_b.astype(np.float32)
        bottom_trigger_a = (bottom_trigger_b * score_macdh_zscore_low).astype(np.float32)
        states['SCORE_MACD_BOTTOM_REVERSAL_A'] = bottom_trigger_a
        states['SCORE_MACD_BOTTOM_REVERSAL_S'] = (bottom_trigger_a * score_mom_short).astype(np.float32)

        # 2.5 顶部反转 (死叉)
        top_trigger_b = (1 - score_accel_short)
        states['SCORE_MACD_TOP_REVERSAL_B'] = top_trigger_b.astype(np.float32)
        top_trigger_a = (top_trigger_b * score_macdh_zscore_high).astype(np.float32)
        states['SCORE_MACD_TOP_REVERSAL_A'] = top_trigger_a
        states['SCORE_MACD_TOP_REVERSAL_S'] = (top_trigger_a * (1 - score_mom_short)).astype(np.float32)

        # --- 3. 成交量动态分析 (逻辑不变) ---
        candle_body_up = (df['close_D'] - df['open_D']).clip(lower=0)
        candle_body_down = (df['open_D'] - df['close_D']).clip(lower=0)
        score_price_up_strength = self._normalize_score(candle_body_up)
        score_price_down_strength = self._normalize_score(candle_body_down)
        score_vol_slope_up = self._normalize_score(df['SLOPE_5_volume_D'].clip(lower=0))
        score_vol_accel_up = self._normalize_score(df['ACCEL_5_volume_D'].clip(lower=0))
        score_volume_igniting = score_vol_slope_up * score_vol_accel_up
        states['SCORE_VOL_PRICE_IGNITION_UP'] = score_price_up_strength * score_volume_igniting
        states['SCORE_VOL_PRICE_PANIC_DOWN_RISK'] = score_price_down_strength * score_volume_igniting
        return states










