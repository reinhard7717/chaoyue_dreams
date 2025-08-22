# 文件: strategies/weekly_context_engine.py (新文件名，或替换旧文件)
# 版本: V1.0 - 战略上下文引擎

from typing import Dict, Optional
import numpy as np
import pandas as pd
import logging
from strategies.kline_pattern_recognizer import KlinePatternRecognizer

logger = logging.getLogger(__name__)

class WeeklyContextEngine:
    """
    【V3.0】周线战略上下文引擎 - 联合情报中心版
    - 职能定位: 进化为“联合情报中心的首席战略官”，为日线策略提供
                一个包含市场状态、量价关系、关键背离和量化战略分数
                的、深刻而立体的战场沙盘。
    - 核心进化:
      1.  **市场状态机**: 融合趋势与波动率，定义四种核心市场状态。
      2.  **量价分析引擎**: 深入分析OBV和CMF，洞察资金真实意图。
      3.  **背离检测引擎**: 引入实用的价格与RSI背离检测，提供前瞻性信号。
      4.  **量化战略分数**: 将所有分析合成为一个分数，为日线策略提供
                         带有置信度的作战指导。
    """

    def __init__(self, config: dict):
        """
        【V1.1 适配统一指挥版】
        - 核心升级: 重构了配置读取逻辑，使其能够正确解析新的“统一配置文件”。
        - 新逻辑:
          - 从统一配置的顶层获取 'feature_engineering_params'。
          - 从统一配置的 'weekly_context_params' 块中获取自己的行动指令。
        """
        # 1. 从统一配置中，获取周线引擎专属的逻辑参数块
        self.params = config.get('weekly_context_params', {})
        # 2. 从统一配置的顶层，获取所有指标的定义
        self.indicator_cfg = config.get('feature_engineering_params', {}).get('indicators', {})
        # 3. 从周线专属的逻辑块中，获取剧本定义
        self.playbook_params = self.params.get('strategy_playbooks', {})
        # 初始化K线形态识别器，用于周线心理学分析
        kline_params = config.get('strategy_params', {}).get('trend_follow', {}).get('kline_pattern_params', {})
        self.pattern_recognizer = KlinePatternRecognizer(params=kline_params)

    def generate_context(self, df_weekly: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 · 战略情报生成流水线】
        """
        if df_weekly is None or df_weekly.empty:
            logger.warning("周线上下文引擎输入DataFrame为空，无法生成信号。")
            return pd.DataFrame()

        print("\n" + "="*30 + "【周线战略战场指挥官 V4.0】启动" + "="*30)
        
        context_df = df_weekly.copy()

        # --- 流水线 1/7: 定义战场关键参照物 (新) ---
        print("---【步骤1/6: 定义战场关键参照物】---")
        context_df = self._define_key_reference_levels(context_df)

        # --- 流水线 2/7: 解读周线K线心理学 (新) ---
        print("---【步骤2/6: 解读周线K线心理学】---")
        context_df = self._analyze_candlestick_psychology(context_df)

        # --- 流水线 3/7: 标定动态风险控制线 (新) ---
        print("---【步骤3/6: 标定动态风险控制线】---")
        context_df = self._calculate_dynamic_risk_levels(context_df)
        
        # --- 流水线 4/7: 诊断趋势“品格” (新) ---
        print("---【步骤4/6: 诊断趋势“品格”】---")
        context_df = self._characterize_trend_health(context_df)

        # --- 流水线 5/7: 构建市场状态机 (原) ---
        print("---【步骤5/6: 构建市场状态机】---")
        context_df = self._build_market_regime(context_df)

        # --- 流水线 6/7: 深度量价与背离分析 (原) ---
        print("---【步骤6/6: 深度量价与背离分析】---")
        context_df = self._analyze_vpa(context_df)
        context_df = self._detect_divergences(context_df)

        # --- 流水线 7/7: 合成战略分数与最终信号 (升级) ---
        print("---【最终步骤: 合成战略分数与最终信号】---")
        context_df = self._calculate_strategic_score(context_df)
        
        score = context_df['strategic_score_W']
        context_df['state_node_main_ascent_W'] = score >= 5
        context_df['state_node_ignition_W'] = score.between(2, 5, inclusive='left')
        context_df['state_node_topping_W'] = score <= -3
        
        # 筛选最终需要注入日线的信号列
        final_signal_cols = [
            # 核心分数与状态
            'strategic_score_W',
            'state_node_main_ascent_W',
            'state_node_ignition_W',
            'state_node_topping_W',
            # 新增的战略情报
            'ref_dist_from_52w_high_W', # 距离52周高点的距离(百分比)
            'ref_support_level_W',      # 关键支撑位
            'risk_volatility_stop_W',   # 波动率止损位
            'psych_reversal_bullish_W', # K线看涨反转信号
            'psych_rejection_bearish_W',# K线看跌反转信号
            'trend_health_strong_W',    # 趋势健康度强
        ]
        
        original_cols = df_weekly.columns
        output_cols = [col for col in context_df.columns if col in final_signal_cols and col not in original_cols]
        
        print(f"    - [指挥中心] 已生成 {len(output_cols)} 个最终周线战略指挥信号。")
        print("="*30 + "【周线战略战场指挥官 V4.0】执行完毕" + "="*30 + "\n")
        return context_df[output_cols]

    # --- 新增模块 ---
    def _define_key_reference_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """【新增】定义战场关键参照物"""
        # 1. 52周高点
        high_52w = df['high_W'].rolling(52, min_periods=20).max()
        df['ref_dist_from_52w_high_W'] = (df['close_W'] - high_52w) / high_52w
        
        # 2. 关键支撑位 (定义为过去26周的最低点)
        df['ref_support_level_W'] = df['low_W'].rolling(26, min_periods=10).min()
        
        print("    - [参照物] 完成。已标定52周高点与关键支撑位。")
        return df

    def _analyze_candlestick_psychology(self, df: pd.DataFrame) -> pd.DataFrame:
        """【新增】解读周线K线心理学"""
        # 复用日线策略的K线形态识别器
        df_with_patterns = self.pattern_recognizer.identify_all(df, suffix='_W')
        
        # 提取关键的心理学信号
        # 1. 看涨反转/确认信号 (如锤子线, 看涨吞没)
        bullish_patterns = ['HAMMER_W', 'INVHAMMER_W', 'BULLISH_ENGULFING_W', 'PIERCING_W', 'MORNINGSTAR_W']
        df['psych_reversal_bullish_W'] = df_with_patterns[[p for p in bullish_patterns if p in df_with_patterns.columns]].any(axis=1)
        
        # 2. 看跌反转/拒绝信号 (如射击之星, 看跌吞没)
        bearish_patterns = ['SHOOTINGSTAR_W', 'BEARISH_ENGULFING_W', 'DARKCLOUDCOVER_W', 'EVENINGSTAR_W']
        df['psych_rejection_bearish_W'] = df_with_patterns[[p for p in bearish_patterns if p in df_with_patterns.columns]].any(axis=1)

        print("    - [心理学] 完成。已解读周线K线的多空意图。")
        return df

    def _calculate_dynamic_risk_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """【新增】标定动态风险控制线"""
        atr_col = 'ATR_14_W'
        if atr_col not in df.columns:
            print(f"    - [风险线-警告] 缺少 {atr_col} 列，无法计算动态风险线。")
            df['risk_volatility_stop_W'] = np.nan
            return df
            
        # 定义风险线为：收盘价下方 2 倍 ATR
        df['risk_volatility_stop_W'] = df['close_W'] - (2 * df[atr_col])
        
        print("    - [风险线] 完成。已基于ATR计算动态风险控制线。")
        return df

    def _characterize_trend_health(self, df: pd.DataFrame) -> pd.DataFrame:
        """【新增】诊断趋势“品格”"""
        ema10_col = 'EMA_10_W'
        ema21_col = 'EMA_21_W'
        if not all(c in df.columns for c in [ema10_col, ema21_col]):
            print("    - [趋势品格-警告] 缺少EMA列，无法诊断趋势健康度。")
            df['trend_health_strong_W'] = False
            return df
            
        # 定义强健康趋势为：收盘价 > 10周线 > 21周线
        is_price_above_ema10 = df['close_W'] > df[ema10_col]
        is_ema10_above_ema21 = df[ema10_col] > df[ema21_col]
        df['trend_health_strong_W'] = is_price_above_ema10 & is_ema10_above_ema21
        
        print("    - [趋势品格] 完成。已诊断趋势的健康度。")
        return df

    def _build_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【核心进化一】构建市场状态机
        融合趋势(EMA斜率)和波动率(布林带宽度BBW)，将市场划分为四种核心状态。
        这符合A股牛市分“放量快牛”和“缩量慢牛”的特征。
        使用滚动分位数作为阈值，自适应不同股票的波动特性，避免了固定阈值的陷阱。
        """
        slope_col = 'SLOPE_5_EMA_21_W'
        bbw_col = 'BBW_20_2.0_W'
        if not all(c in df.columns for c in [slope_col, bbw_col]):
            print("    - [状态机-警告] 缺少斜率或BBW列，无法构建市场状态机。")
            return df

        # 1. 定义趋势状态
        is_uptrend = df[slope_col] > 0
        is_downtrend = df[slope_col] < 0

        # 2. 定义波动率状态 (使用滚动分位数，更具适应性)
        # 使用过去一年(52周)的数据作为参考系
        vol_high_threshold = df[bbw_col].rolling(52, min_periods=20).quantile(0.70)
        vol_low_threshold = df[bbw_col].rolling(52, min_periods=20).quantile(0.30)
        
        is_vol_expansion = df[bbw_col] > vol_high_threshold
        is_vol_contraction = df[bbw_col] < vol_low_threshold

        # 3. 组合成四种市场状态
        df['regime_bull_vol_expansion_W'] = is_uptrend & is_vol_expansion     # 牛市主升浪 (快牛)
        df['regime_bull_quiet_W'] = is_uptrend & is_vol_contraction          # 牛市静默期 (慢牛/蓄力)
        df['regime_bear_vol_expansion_W'] = is_downtrend & is_vol_expansion    # 熊市主跌浪 (杀跌)
        df['regime_bear_quiet_W'] = is_downtrend & is_vol_contraction         # 熊市静默期 (阴跌/筑底)
        
        print("    - [状态机] 完成。已将市场划分为四种核心状态。")
        return df

    def _analyze_vpa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【核心进化二】深度量价分析引擎
        利用OBV和CMF洞察资金的真实意图，判断价格行为的健康度。
        这在A股市场尤其重要，因为“聪明钱”的动向往往领先于价格。
        """
        obv_col = 'OBV_W'
        cmf_col = 'CMF_21_W'
        slope_ema_col = 'SLOPE_5_EMA_21_W'
        if not all(c in df.columns for c in [obv_col, cmf_col, slope_ema_col]):
            print("    - [VPA-警告] 缺少OBV或CMF列，无法进行深度量价分析。")
            return df

        # 1. OBV趋势与价格趋势验证
        # 使用简单差分计算OBV短期趋势， robust and simple
        slope_obv = df[obv_col].diff(5) 
        df['vpa_health_W'] = (df[slope_ema_col] > 0) & (slope_obv > 0)

        # 2. CMF资金流状态
        # CMF > 0.05 通常被认为是机构在吸筹
        # CMF < -0.05 通常被认为是机构在派发
        df['cmf_accumulation_W'] = df[cmf_col] > 0.05
        df['cmf_distribution_W'] = df[cmf_col] < -0.05
        
        print("    - [VPA] 完成。已分析OBV趋势与CMF资金流状态。")
        return df

    def _detect_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【核心进化三】关键背离检测引擎
        检测价格与RSI指标的背离，提供极具价值的前瞻性信号。
        实现方式采用滚动窗口对比，避免了寻找精确波峰/波谷的数学陷阱，更贴近实战。
        """
        rsi_col = 'RSI_12_W'
        if rsi_col not in df.columns:
            print("    - [背离检测-警告] 缺少RSI列，无法进行背离检测。")
            return df
        
        N = 26 # 使用半年(26周)作为背离的观察窗口

        # 1. 检测熊市顶背离 (价格新高附近，RSI却明显走弱)
        price_near_high = df['high_W'] >= df['high_W'].rolling(N).max() * 0.98
        rsi_not_at_high = df[rsi_col] < df[rsi_col].rolling(N).max() * 0.85
        df['risk_bearish_divergence_W'] = price_near_high & rsi_not_at_high

        # 2. 检测牛市底背离 (价格新低附近，RSI却拒绝创新低)
        price_near_low = df['low_W'] <= df['low_W'].rolling(N).min() * 1.02
        rsi_not_at_low = df[rsi_col] > df[rsi_col].rolling(N).min() * 1.15
        df['opp_bullish_divergence_W'] = price_near_low & rsi_not_at_low
        
        print("    - [背离检测] 完成。已检测价格与RSI的潜在背离。")
        return df

    def _calculate_strategic_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【核心进化四】合成量化战略分数
        将所有维度的分析结果，通过加权计分，合成为一个综合性的战略分数。
        分数权重体现了A股市场的经验：风险信号的权重通常高于机会信号。
        """
        score = pd.Series(0.0, index=df.index)

        # 1. 市场状态基础分
        score += df.get('regime_bull_vol_expansion_W', 0) * 3  # 最强的状态
        score += df.get('regime_bull_quiet_W', 0) * 2          # 次强的状态
        score -= df.get('regime_bear_vol_expansion_W', 0) * 5  # 最危险的状态，强力否决
        score -= df.get('regime_bear_quiet_W', 0) * 2          # 次危险的状态

        # 2. 量价分析加减分
        score += df.get('vpa_health_W', 0) * 2
        score += df.get('cmf_accumulation_W', 0) * 2
        score -= df.get('cmf_distribution_W', 0) * 3          # 资金派发是重要风险

        # 3. 背离信号加减分
        score += df.get('opp_bullish_divergence_W', 0) * 3     # 底背离是强反转信号
        score -= df.get('risk_bearish_divergence_W', 0) * 4    # 顶背离是极重要风险

        # 4. 战略情报加减分
        # 4.1 趋势品格加分：健康的趋势是强加分项
        score += df.get('trend_health_strong_W', 0) * 2
        
        # 4.2 K线心理学加减分：周线反转形态有很高的权重
        score += df.get('psych_reversal_bullish_W', 0) * 3
        score -= df.get('psych_rejection_bearish_W', 0) * 4
        
        # 4.3 关键位置加减分：接近52周高点是强势特征
        dist_from_high = df.get('ref_dist_from_52w_high_W', 0)
        # 当股价在52周高点下方5%以内时，认为是即将突破的强势区，加分
        score += (dist_from_high.between(-0.05, 0.05)) * 2

        df['strategic_score_W'] = score
        print("    - [战略计分] 完成。已生成综合战略分数。")
        return df

    def _calculate_all_playbooks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.5 命名规范化版】动态遍历JSON配置，并确保输出的剧本名称为大写，与日线策略对齐。
        """
        # print("\n" + "="*80)
        # print(f"---【周线战略层(V3.5 命名规范化) - 检查最新一周: {df.index[-1].date()}】---")
        # print("="*80)
        
        context_df = df.copy()
        
        playbook_map = {
            'ma20_rising_state_playbook': self._playbook_ma20_is_rising,
            'ma20_turn_up_event_playbook': self._playbook_ma20_turn_up_event,
            'early_uptrend_playbook': self._playbook_early_uptrend,
            'classic_breakout_playbook': self._playbook_classic_breakout,
            'ma_uptrend_playbook': self._playbook_check_ma_uptrend,
            'box_consolidation_breakout_playbook': self._playbook_box_consolidation_breakout,
            'oversold_rebound_bias_playbook': self._playbook_oversold_rebound_bias,
            'washout_score_playbook': self._playbook_calculate_washout_score,
            'rejection_filter_playbook': self._playbook_check_rejection_filters,
            'trix_golden_cross_playbook': self._playbook_trix_golden_cross,
            'coppock_reversal_playbook': self._playbook_coppock_reversal,
            'ace_signal_breakout_trigger_playbook': self._playbook_ace_signal_breakout_trigger,
        }

        for playbook_name, params in self.playbook_params.items():
            if playbook_name == "说明": continue

            if playbook_name in playbook_map:
                if params.get('enabled', False):
                    results = playbook_map[playbook_name](df, params)
                    
                    if isinstance(results, dict):
                        for signal_suffix, result_series in results.items():
                            # ▼▼▼【代码修改】: 规范化多信号输出的剧本名称 ▼▼▼
                            # 将 'coppock_stabilizing' 转换为 'PLAYBOOK_COPPOCK_STABILIZING_W'
                            col_name = f"playbook_{signal_suffix.upper()}_W"
                            context_df[col_name] = result_series
                            # print(f"    - [多信号输出模式] 已生成规范化列: '{col_name}'")
                    elif isinstance(results, pd.Series):
                        if 'score' in playbook_name:
                            col_name = 'washout_score_W'
                        elif 'filter' in playbook_name:
                            col_name = 'rejection_signal_W'
                        else:
                            # 将 'ma20_turn_up_event_playbook' 转换为 'PLAYBOOK_MA20_TURN_UP_EVENT_W'
                            base_name = playbook_name.replace('_playbook', '').upper()
                            col_name = f"playbook_{base_name}_W"
                        
                        # print(f"    - [单信号输出模式] 正在为规范化列 '{col_name}' 赋值...")
                        context_df[col_name] = results
                else:
                    # print(f"\n--- 剧本检查: [{params.get('说明', playbook_name)}] ---")
                    print("    - 结论: [未启用]")
            else:
                logger.warning(f"JSON中配置的剧本 '{playbook_name}' 在代码中没有找到对应的实现函数，已跳过。")

        # print("\n---【周线战略层(V3.5) - 剧本计算总结】---")
        # for col in context_df.columns:
        #     if col.startswith('playbook_'):
        #         # label = col.replace('playbook_', '').replace('_W', '').replace('_', ' ').title()
        #         print(f"【剧本-{col}】触发周数: {context_df[col].sum()}")
        # if 'washout_score_W' in context_df.columns:
        #     score = context_df['washout_score_W']
        #     print(f"【诊断-洗盘】有分数的周数: {(score > 0).sum()} (最高分: {score.max()})")
        # if 'rejection_signal_W' in context_df.columns:
        #     rejection = context_df['rejection_signal_W']
        #     print(f"【诊断-风险】有拒绝信号的周数: {(rejection < 0).sum()}")
        
        return context_df

    def _playbook_ma20_is_rising(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.1 升级剧本】: 识别指定周线均线是否处于“有效”上升状态。
        - 核心修改: 引入斜率阈值，过滤掉几乎走平的“伪上涨”状态。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', '均线处于上升状态')}] ---")
        target_ma_period = params.get('ma_period', 21)
        # 新增斜率阈值参数，要求均线每周至少上涨0.1%才算有效上涨
        slope_threshold_pct = params.get('slope_threshold_pct', 0.1) 
        # print(f"    - 配置参数: ma_period={target_ma_period}, slope_threshold_pct={slope_threshold_pct}%")
        
        ema_col = f'EMA_{target_ma_period}_W'
        close_col = 'close_W'
        if not self._check_dependencies(df, [ema_col, close_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        # 计算斜率，并进行标准化，使其不受股价绝对值影响
        slope = df[ema_col].diff(1)
        safe_ma = df[ema_col].shift(1).replace(0, 1e-9) # 防止除以零
        normalized_slope_pct = (slope / safe_ma) * 100

        # 条件1: 标准化后的斜率必须大于阈值
        condition1_is_rising_effectively = normalized_slope_pct > slope_threshold_pct
        
        # 条件2: 收盘价在均线之上
        condition2_price_confirm = df[close_col] > df[ema_col]
        
        final_signal = condition1_is_rising_effectively & condition2_price_confirm
        
        # 调试信息
        # last = df.iloc[-1]
        # c1_last = condition1_is_rising_effectively.iloc[-1]
        # c2_last = condition2_price_confirm.iloc[-1]
        
        # print(f"    - 条件1 (均线有效上涨): {'[✓]' if c1_last else '[✗]'} (周涨幅: {normalized_slope_pct.iloc[-1]:.3f}% > 阈值: {slope_threshold_pct}%)")
        # print(f"    - 条件2 (价格确认): {'[✓]' if c2_last else '[✗]'} (收盘价: {last.get(close_col, 0):.2f} vs 均线: {last.get(ema_col, 0):.2f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)

    def _playbook_ma20_turn_up_event(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.1 升级剧本】: 识别指定周线均线“加速拐头向上”的事件。
        - 核心修改: 引入二阶斜率（加速度）作为判断条件。
                     要求拐头不仅是方向改变，还必须是“加速”的，以提高信号质量。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', '均线拐头向上事件')}] ---")
        target_ma_period = params.get('ma_period', 21)
        # 新增加速度阈值参数，如果JSON中没有，则默认为0
        accel_threshold = params.get('accel_threshold', 0) 
        # print(f"    - 配置参数: ma_period={target_ma_period}, accel_threshold={accel_threshold}")
        
        ema_col = f'EMA_{target_ma_period}_W'
        close_col = 'close_W'
        if not self._check_dependencies(df, [ema_col, close_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        # 计算斜率（速度）和二阶斜率（加速度）
        slope = df[ema_col].diff(1)
        acceleration = slope.diff(1)

        # 条件1: 发生拐头 (本周斜率为正，上周为负或零)
        slope_is_positive = slope > 0
        slope_was_not_positive = slope.shift(1) <= 0
        condition1_turn_up = slope_is_positive & slope_was_not_positive
        
        # 条件2 (新增): 加速度必须为正，且大于阈值，确保是“有力”的拐头
        condition2_is_accelerating = acceleration > accel_threshold

        # 条件3: 收盘价在均线之上作为确认
        condition3_price_confirm = df[close_col] > df[ema_col]

        final_signal = condition1_turn_up & condition2_is_accelerating & condition3_price_confirm
        
        # 调试信息
        last = df.iloc[-1]
        c1_last = condition1_turn_up.iloc[-1]
        c2_last = condition2_is_accelerating.iloc[-1]
        c3_last = condition3_price_confirm.iloc[-1]
        
        # print(f"    - 条件1 (均线发生拐头): {'[✓]' if c1_last else '[✗]'} (本周斜率: {slope.iloc[-1]:.2f} > 0 AND 上周斜率: {slope.shift(1).iloc[-1]:.2f} <= 0)")
        # print(f"    - 条件2 (拐头正在加速): {'[✓]' if c2_last else '[✗]'} (加速度: {acceleration.iloc[-1]:.2f} > 阈值: {accel_threshold})")
        # print(f"    - 条件3 (价格确认): {'[✓]' if c3_last else '[✗]'} (收盘价: {last.get(close_col, 0):.2f} vs 均线: {last.get(ema_col, 0):.2f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)

    def _playbook_early_uptrend(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：捕捉周线趋势反转的早期“上拐”信号"""
        # print(f"\n--- 剧本检查: [{params.get('说明', '早期上升趋势')}] ---")
        
        # 从传递的params中读取周期，如果未定义则使用默认值10和20
        short_ma_period = params.get('short_ma', 10)
        mid_ma_period = params.get('mid_ma', 20)
        # print(f"    - 配置参数: short_ma={short_ma_period}, mid_ma={mid_ma_period}")

        # 使用读取到的周期构建列名
        short_ma_col = f'EMA_{short_ma_period}_W'
        mid_ma_col = f'EMA_{mid_ma_period}_W'

        macd_params_raw = self.indicator_cfg.get('macd', {}).get('periods', [12, 26, 9])
        p_fast, p_slow, p_signal = macd_params_raw
        macd_col = f'MACD_{p_fast}_{p_slow}_{p_signal}_W'
        macd_hist_col = f'MACDh_{p_fast}_{p_slow}_{p_signal}_W'
        required_cols = [short_ma_col, mid_ma_col, macd_col, macd_hist_col, 'close_W']
        
        if not self._check_dependencies(df, required_cols, log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        ma_slope = df[short_ma_col].diff()
        ma_is_up = ma_slope > 0
        ma_turning_up = (ma_slope > 0) & (ma_slope.shift(1) <= 0)
        price_cross_ma = (df['close_W'] > df[mid_ma_col]) & (df['close_W'].shift(1) <= df[mid_ma_col].shift(1))
        macd_cross_zero_nearby = (df[macd_hist_col] > 0) & (df[macd_hist_col].shift(1) <= 0) & (df[macd_col].abs() < df['close_W'] * 0.05)
        signal = (ma_turning_up | price_cross_ma) & macd_cross_zero_nearby
        in_early_uptrend = (df[short_ma_col] > df[mid_ma_col]) & ma_is_up
        final_signal = (signal | in_early_uptrend)
        s_last = signal.iloc[-1]
        ieu_last = in_early_uptrend.iloc[-1]
        print(f"    - 子信号1 (拐点信号): {'[✓]' if s_last else '[✗]'}")
        print(f"    - 子信号2 (趋势延续): {'[✓]' if ieu_last else '[✗]'}")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}] (逻辑: 拐点 OR 延续)")
        return final_signal.fillna(False)

    def _playbook_classic_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【剧本 V4.0 动态增强版】: 经典高点突破 (注入动态灵魂)
        - 核心升级: 增加第三个核心条件，要求突破必须发生在趋势“健康”或“加速”的背景下。
        """
        print(f"\n--- 剧本检查: [{params.get('说明', '经典高点突破')}] (V4.0 动态增强版) ---")
        lookback_weeks, volume_multiplier = params.get('lookback_weeks', 26), params.get('volume_multiplier', 1.5)
        # 依赖检查，确保我们有斜率数据
        slope_col = 'SLOPE_5_EMA_21_W' # 使用我们定义的核心斜率列
        required_cols = ['high_W', 'volume_W', 'close_W', slope_col]
        if not self._check_dependencies(df, required_cols, log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)
        
        # 条件1: 价格突破 
        period_high = df['high_W'].shift(1).rolling(window=lookback_weeks).max()
        is_price_breakout = df['close_W'] > period_high
        
        # 条件2: 放量突破 
        avg_volume = df['volume_W'].shift(1).rolling(window=lookback_weeks).mean()
        is_volume_breakout = df['volume_W'] > (avg_volume * volume_multiplier)
        
        # 条件3 (新增): 趋势动能确认
        # 要求突破发生时，周线级别的趋势速度必须是正向的
        slope_threshold = params.get('slope_threshold', 0) # 允许在JSON中配置最小斜率
        is_trend_supportive = df[slope_col] > slope_threshold
        
        final_signal = is_price_breakout & is_volume_breakout & is_trend_supportive
        last = df.iloc[-1]
        ph_last = period_high.iloc[-1]
        av_last = avg_volume.iloc[-1]
        pb_last = is_price_breakout.iloc[-1]
        vb_last = is_volume_breakout.iloc[-1]
        print(f"    - 条件1 (价格突破): {'[✓]' if pb_last else '[✗]'} (收盘价: {last.get('close_W', float('nan')):.2f} vs 前{lookback_weeks}周高点: {ph_last:.2f})")
        print(f"    - 条件2 (放量突破): {'[✓]' if vb_last else '[✗]'} (成交量: {last.get('volume_W', 0):.0f} vs 阈值: {(av_last * volume_multiplier):.0f})")
        print(f"    - 条件3 (趋势动能支持): {'[✓]' if is_trend_supportive.iloc[-1] else '[✗]'} (斜率: {df[slope_col].iloc[-1]:.4f} > 阈值: {slope_threshold})")
        print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        return final_signal.fillna(False)

    def _playbook_check_ma_uptrend(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【剧本 V4.0 动态增强版】: 均线多头排列 (注入动态灵魂)
        - 核心升级: 增加对核心均线斜率的判断，确保“多头排列”不是伪信号。
        """
        # ... (参数加载逻辑不变) ...
        short_ma, mid_ma, long_ma = params.get('short_ma', 13), params.get('mid_ma', 21), params.get('long_ma', 55)
        short_col, mid_col, long_col = f'EMA_{short_ma}_W', f'EMA_{mid_ma}_W', f'EMA_{long_ma}_W'
        
        # ▼▼▼【代码修改 V4.0】: 引入斜率作为判断依据 ▼▼▼
        mid_slope_col = f'SLOPE_5_EMA_{mid_ma}_W' # 检查中期趋势线的斜率
        required_cols = [short_col, mid_col, long_col, 'close_W', mid_slope_col]
        if not self._check_dependencies(df, required_cols, log_details=True):
            return pd.Series(False, index=df.index)
        
        tolerance_pct = params.get('tolerance_pct', 0.01)

        # 条件1: 均线排列关系不变
        ma_aligned = (df[short_col] > df[mid_col]) & (df[mid_col] > df[long_col])
        
        # 条件2 (修改): 股价在支撑均线的“容忍区”之上
        support_level_with_tolerance = df[mid_col] * (1 - tolerance_pct)
        price_above_support_zone = df['close_W'] > support_level_with_tolerance

        # 条件3 (新增): 核心趋势线必须向上运行
        slope_threshold = params.get('slope_threshold', 0)
        is_core_ma_rising = df[mid_slope_col] > slope_threshold
        
        final_signal = ma_aligned & price_above_support_zone & is_core_ma_rising
        
        last = df.iloc[-1]
        ma_last = ma_aligned.iloc[-1]
        pas_last = price_above_support_zone.iloc[-1]
        
        # print(f"    - 条件1 (均线多头): {'[✓]' if ma_last else '[✗]'} (EMA{short_ma}: {last.get(short_col, 0):.2f} > EMA{mid_ma}: {last.get(mid_col, 0):.2f} > EMA{long_ma}: {last.get(long_col, 0):.2f})")
        # print(f"    - 条件2 (股价在支撑容忍区上): {'[✓]' if pas_last else '[✗]'} (收盘价: {last.get('close_W', 0):.2f} > 支撑区下轨: {support_level_with_tolerance.iloc[-1]:.2f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)
    
    def _playbook_oversold_rebound_bias(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：利用BIAS指标捕捉周线级别的超跌反弹机会"""
        # print(f"\n--- 剧本检查: [{params.get('说明', 'BIAS超跌反弹')}] ---")
        bias_period = params.get('bias_period', 20)
        bias_col = f'BIAS_{bias_period}_W'
        if not self._check_dependencies(df, [bias_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要列 {bias_col}")
            return pd.Series(False, index=df.index)
        oversold_threshold = params.get('oversold_threshold', -15)
        rebound_trigger = params.get('rebound_trigger', -12)
        was_oversold = (df[bias_col].shift(1) < oversold_threshold)
        is_rebounding = (df[bias_col] > rebound_trigger)
        final_signal = was_oversold & is_rebounding
        last = df.iloc[-1]
        prev = df.iloc[-2]
        wo_last = was_oversold.iloc[-1]
        ir_last = is_rebounding.iloc[-1]
        # print(f"    - 条件1 (上周曾超卖): {'[✓]' if wo_last else '[✗]'} (上周BIAS: {prev.get(bias_col, 0):.2f} < 阈值: {oversold_threshold})")
        # print(f"    - 条件2 (本周正反弹): {'[✓]' if ir_last else '[✗]'} (本周BIAS: {last.get(bias_col, 0):.2f} > 阈值: {rebound_trigger})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        return final_signal.fillna(False)

    def _playbook_calculate_washout_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """诊断剧本：量化周线级别的洗盘行为"""
        print(f"\n--- 诊断检查: [{params.get('说明', '洗盘行为评分')}] ---")
        washout_score = pd.Series(0, index=df.index)
        support_level = self._get_weekly_support_level(df, params)
        if support_level is None:
            print("    - 结论: [失败] 因无法确定支撑位而跳过。")
            return washout_score
        washout_intraday = (df['low_W'] < support_level) & (df['close_W'] > support_level)
        washout_interday = (df['close_W'] > support_level) & (df['close_W'].shift(1) < support_level.shift(1))
        was_below_recently = (df['close_W'].shift(1) < support_level.shift(1)).rolling(window=params.get('drift_lookback_period', 3), min_periods=1).sum() > 0
        washout_drift = (df['close_W'] > support_level) & was_below_recently
        recent_peak = df['high_W'].shift(1).rolling(window=params.get('bull_trap_lookback_period', 8)).max()
        is_in_trap_zone = df['close_W'] < recent_peak * (1 - params.get('bull_trap_drop_threshold', 0.05))
        is_recovering_from_trap = df['close_W'] > df['close_W'].shift(1)
        washout_bull_trap = is_in_trap_zone & is_recovering_from_trap
        avg_volume = df['volume_W'].shift(1).rolling(window=params.get('volume_avg_period', 20)).mean()
        is_volume_contracted = df['volume_W'] < avg_volume * params.get('volume_contraction_threshold', 0.7)
        washout_volume_contraction = (washout_interday | washout_drift) & is_volume_contracted.shift(1).fillna(False)
        washout_score += washout_intraday.astype(int)
        washout_score += washout_interday.astype(int)
        washout_score += washout_drift.astype(int)
        washout_score += washout_bull_trap.astype(int)
        washout_score += washout_volume_contraction.astype(int)
        last_support = support_level.iloc[-1]
        print(f"    - 使用的支撑位: {last_support:.2f}")
        # print(f"    - 模式1 (日内洗盘): {'[+1分]' if washout_intraday.iloc[-1] else '[+0分]'}")
        # print(f"    - 模式2 (日间洗盘): {'[+1分]' if washout_interday.iloc[-1] else '[+0分]'}")
        # print(f"    - 模式3 (漂移收复): {'[+1分]' if washout_drift.iloc[-1] else '[+0分]'}")
        # print(f"    - 模式4 (诱多陷阱): {'[+1分]' if washout_bull_trap.iloc[-1] else '[+0分]'}")
        # print(f"    - 模式5 (缩量确认): {'[+1分]' if washout_volume_contraction.iloc[-1] else '[+0分]'}")
        print(f"    - 结论: 最新一周总得分为 [{washout_score.iloc[-1]}]")
        return washout_score.fillna(0)

    def _get_weekly_support_level(self, df: pd.DataFrame, params: dict) -> Optional[pd.Series]:
        """辅助函数: 获取周线级别的支撑位"""
        support_type = params.get('support_type', 'MA')
        support_level = pd.Series(np.nan, index=df.index)
        if support_type == 'MA':
            ma_period = params.get('support_ma_period', 21)
            ma_col = f'EMA_{ma_period}_W'
            if not self._check_dependencies(df, [ma_col], log_details=False): return None
            support_level = df[ma_col]
        elif support_type == 'BOX':
            boll_period = self.playbook_params.get('box_consolidation_breakout_playbook', {}).get('boll_period', 20)
            boll_std = self.playbook_params.get('box_consolidation_breakout_playbook', {}).get('boll_std', 2.0)
            bbw_col = f"BBW_{boll_period}_{float(boll_std)}_W"
            if not self._check_dependencies(df, [bbw_col, 'low_W'], log_details=False): return None
            quantile_level = self.playbook_params.get('box_consolidation_breakout_playbook', {}).get('bbw_quantile', 0.3)
            threshold = df[bbw_col].quantile(quantile_level)
            is_consolidating = df[bbw_col] < threshold
            if is_consolidating.any():
                box_period = self.playbook_params.get('box_consolidation_breakout_playbook', {}).get('box_period', 26)
                box_bottom = df['low_W'].rolling(window=box_period, min_periods=1).min()
                support_level = box_bottom.where(is_consolidating, np.nan)
        if support_level.isnull().all():
            return None
        return support_level.ffill()

    def _playbook_check_rejection_filters(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """诊断剧本：识别均线和箱体压力位的拒绝信号"""
        print(f"\n--- 诊断检查: [{params.get('说明', '压力位拒绝信号')}] ---")
        ma_period = params.get('ma_period', 21)
        ma_col = f'EMA_{ma_period}_W'
        ma_rejection = self._check_resistance_rejection(df, ma_col, params, "均线压力")
        box_lookback = params.get('box_lookback_period', 52)
        box_resistance_col = f'box_top_{box_lookback}W_resistance'
        df[box_resistance_col] = df['high_W'].shift(1).rolling(window=box_lookback, min_periods=int(box_lookback * 0.8)).max()
        box_rejection = self._check_resistance_rejection(df, box_resistance_col, params, "箱顶压力")
        final_signal = pd.Series(0, index=df.index)
        final_signal[ma_rejection] -= 1
        final_signal[box_rejection] -= 2
        print(f"    - 结论: 最新一周总得分为 [{final_signal.iloc[-1]}] (均线拒绝-1分, 箱顶拒绝-2分)")
        return final_signal

    def _playbook_box_consolidation_breakout(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """剧本：专业箱体突破"""
        # print(f"\n--- 剧本检查: [{params.get('说明', '专业箱体突破')}] ---")
        quantile_level = params.get('bbw_quantile', 0.3)
        boll_period = params.get('boll_period', 20)
        boll_std = self.indicator_cfg.get('boll_bands_and_width', {}).get('std_dev', 2.0)
        box_period = params.get('box_period', 26)
        volume_multiplier = params.get('volume_multiplier', 1.5)
        vol_ma_period = params.get('vol_ma_period', 5)
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_W"
        vol_ma_col = f"VOL_MA_{vol_ma_period}_W"
        required_cols = ['close_W', 'high_W', 'volume_W', bbw_col, vol_ma_col]
        if not self._check_dependencies(df, required_cols, log_details=True):
            print("    - 结论: [失败] 依赖检查失败，策略提前退出。")
            return pd.Series(False, index=df.index)
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=box_period).quantile(quantile_level)
        is_low_volatility_week = df[bbw_col] < dynamic_bbw_threshold
        consolidation_blocks = (is_low_volatility_week != is_low_volatility_week.shift()).cumsum()
        high_in_consolidation = df['high_W'].where(is_low_volatility_week)
        box_high = high_in_consolidation.groupby(consolidation_blocks).transform('max')
        volume_in_consolidation = df['volume_W'].where(is_low_volatility_week)
        box_avg_volume = volume_in_consolidation.groupby(consolidation_blocks).transform('mean')
        is_price_breakout = df['close_W'] > box_high.shift(1)
        is_volume_breakout = df['volume_W'] > (box_avg_volume.shift(1) * volume_multiplier)
        was_in_consolidation = is_low_volatility_week.shift(1).fillna(False)
        final_signal = (was_in_consolidation & is_price_breakout & is_volume_breakout)
        last_idx = -1
        prev_bbw = df[bbw_col].iloc[last_idx - 1]
        prev_bbw_thresh = dynamic_bbw_threshold.iloc[last_idx - 1]
        prev_box_high = box_high.shift(1).iloc[last_idx]
        prev_box_avg_vol = box_avg_volume.shift(1).iloc[last_idx]
        curr_close = df['close_W'].iloc[last_idx]
        curr_vol = df['volume_W'].iloc[last_idx]
        c1 = was_in_consolidation.iloc[last_idx]
        c2 = is_price_breakout.iloc[last_idx]
        c3 = is_volume_breakout.iloc[last_idx]
        # print(f"    - 条件1 (前一周处于盘整期): {'[✓]' if c1 else '[✗]'} (前周BBW: {prev_bbw:.4f} vs 动态阈值: {prev_bbw_thresh:.4f})")
        # print(f"    - 条件2 (价格突破箱顶): {'[✓]' if c2 else '[✗]'} (本周收盘: {curr_close:.2f} vs 前周箱顶: {prev_box_high:.2f})")
        # print(f"    - 条件3 (成交量突破): {'[✓]' if c3 else '[✗]'} (本周成交量: {curr_vol:.0f} vs 阈值: {(prev_box_avg_vol * volume_multiplier):.0f})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[last_idx] else '未触发'}]")
        return final_signal.fillna(False)

    def _playbook_trix_golden_cross(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.2 升级剧本】: 识别周线TRIX“强力金叉”。
        - 核心修改: 增加TRIX线自身斜率的判断，要求金叉时必须是“加速向上”的。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', 'TRIX金叉')}] ---")
        trix_cfg = self.indicator_cfg.get('trix', {})
        trix_periods = next((c.get('periods') for c in trix_cfg.get('configs', []) if 'W' in c.get('apply_on', [])), None)
        if not trix_periods or len(trix_periods) < 2:
            print("    - 结论: [失败] TRIX周期参数配置不正确。")
            return pd.Series(False, index=df.index)
        
        trix_len, signal_len = trix_periods[0], trix_periods[1]
        slope_threshold = params.get('slope_threshold', 0.01) # 从JSON获取斜率阈值
        # print(f"    - 配置参数: trix_len={trix_len}, signal_len={signal_len}, slope_threshold={slope_threshold}")

        trix_col = f'TRIX_{trix_len}_{signal_len}_W'
        trix_signal_col = f'TRIXs_{trix_len}_{signal_len}_W'
        if not self._check_dependencies(df, [trix_col, trix_signal_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要的TRIX列。")
            return pd.Series(False, index=df.index)

        # 条件1: 经典金叉
        condition1_is_golden_cross = (df[trix_col] > df[trix_signal_col]) & \
                                     (df[trix_col].shift(1) <= df[trix_signal_col].shift(1))
        
        # 条件2 (新增): TRIX线自身斜率必须大于阈值，确保是强力金叉
        trix_slope = df[trix_col].diff(1)
        condition2_is_strong_momentum = trix_slope > slope_threshold

        final_signal = condition1_is_golden_cross & condition2_is_strong_momentum
        
        last = df.iloc[-1]
        c1_last = condition1_is_golden_cross.iloc[-1]
        c2_last = condition2_is_strong_momentum.iloc[-1]
        
        # print(f"    - 条件1 (发生金叉): {'[✓]' if c1_last else '[✗]'}")
        # print(f"    - 条件2 (动能强劲): {'[✓]' if c2_last else '[✗]'} (TRIX斜率: {trix_slope.iloc[-1]:.4f} > 阈值: {slope_threshold})")
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)
    
    def _playbook_coppock_reversal(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V3.4 升级剧本】: Coppock指标形态学 - 分离左侧企稳与右侧加速信号。
        - 核心修改: 不再输出单一信号，而是返回一个包含两个独立信号的字典：
          1. coppock_stabilizing (左侧): 捕捉深水区“跌势衰竭，首次拐头”的瞬间。
          2. coppock_accelerating (右侧): 捕捉拐头后“上涨加速，动能确认”的瞬间。
        """
        # print(f"\n--- 剧本检查: [{params.get('说明', 'Coppock双信号反转')}] ---")
        coppock_cfg = self.indicator_cfg.get('coppock', {})
        coppock_periods = next((c.get('periods') for c in coppock_cfg.get('configs', []) if 'W' in c.get('apply_on', [])), None)
        if not coppock_periods or len(coppock_periods) < 3:
            print("    - 结论: [失败] Coppock周期参数配置不正确。")
            return {}
        
        p1, p2, p3 = coppock_periods[0], coppock_periods[1], coppock_periods[2]
        deep_value_threshold = params.get('deep_value_threshold', -100)
        accel_threshold = params.get('accel_threshold', 10) # 上涨加速度阈值
        # print(f"    - 配置参数: deep_value={deep_value_threshold}, accel_threshold={accel_threshold}")

        coppock_col = f'COPP_{p1}_{p2}_{p3}_W'
        if not self._check_dependencies(df, [coppock_col], log_details=True):
            print(f"    - 结论: [失败] 缺少必要的Coppock列。")
            return {}

        # --- 计算基础变量：斜率(速度)和加速度 ---
        slope = df[coppock_col].diff(1)
        acceleration = slope.diff(1)

        # --- 信号1: 左侧企稳信号 (Coppock Stabilizing) ---
        # 条件1.1: 发生拐头 (斜率由负/零转正)
        is_turning_up = (slope > 0) & (slope.shift(1) <= 0)
        # 条件1.2: 拐头必须发生在深水区
        was_in_deep_zone = df[coppock_col].shift(1) < deep_value_threshold
        signal_stabilizing = is_turning_up & was_in_deep_zone

        # --- 信号2: 右侧加速信号 (Coppock Accelerating) ---
        # 条件2.1: 必须已经处于上升趋势中 (斜率为正)
        is_rising = slope > 0
        # 条件2.2: 加速度首次超过阈值
        is_accelerating = (acceleration > accel_threshold) & (acceleration.shift(1) <= accel_threshold)
        signal_accelerating = is_rising & is_accelerating

        # --- 调试信息 ---
        last_idx = -1
        s_stab_last = signal_stabilizing.iloc[last_idx]
        s_accel_last = signal_accelerating.iloc[last_idx]
        # print(f"    - [左侧信号: 企稳]")
        # print(f"      - 条件1 (深水区拐头): {'[✓]' if s_stab_last else '[✗]'} (上周值: {df[coppock_col].shift(1).iloc[last_idx]:.2f} < {deep_value_threshold} AND 发生拐头)")
        # print(f"    - [右侧信号: 加速]")
        # print(f"      - 条件1 (上涨加速): {'[✓]' if s_accel_last else '[✗]'} (加速度: {acceleration.iloc[last_idx]:.2f} > {accel_threshold} AND 首次满足)")
        # print(f"    - 结论: 左侧信号=[{'触发' if s_stab_last else '未触发'}], 右侧信号=[{'触发' if s_accel_last else '未触发'}]")

        return {
            'coppock_stabilizing': signal_stabilizing.fillna(False),
            'coppock_accelerating': signal_accelerating.fillna(False)
        }
   
    def _playbook_ace_signal_breakout_trigger(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """【新增剧本】: 王牌突破信号，结合年度高点突破、放量和TRIX确认。"""
        # print(f"\n--- 剧本检查: [{params.get('说明', '王牌突破信号')}] ---")
        
        lookback_weeks = params.get('lookback_weeks', 52)
        volume_multiplier = params.get('volume_multiplier', 2.0)
        vol_ma_period = params.get('vol_ma_period', 5)
        trix_confirm = params.get('trix_confirm', True)
        
        vol_ma_col = f'VOL_MA_{vol_ma_period}_W'
        required_cols = ['high_W', 'close_W', 'volume_W', vol_ma_col]
        
        trix_col, trix_signal_col = None, None
        if trix_confirm:
            trix_cfg = self.indicator_cfg.get('trix', {})
            trix_periods = next((c.get('periods') for c in trix_cfg.get('configs', []) if 'W' in c.get('apply_on', [])), None)
            if trix_periods and len(trix_periods) >= 2:
                trix_len, signal_len = trix_periods[0], trix_periods[1]
                trix_col = f'TRIX_{trix_len}_{signal_len}_W'
                trix_signal_col = f'TRIXs_{trix_len}_{signal_len}_W'
                required_cols.extend([trix_col, trix_signal_col])
            else:
                print("    - 警告: TRIX确认已启用，但无法在指标配置中找到有效的周线TRIX参数。")
                trix_confirm = False

        if not self._check_dependencies(df, required_cols, log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)

        period_high = df['high_W'].shift(1).rolling(window=lookback_weeks, min_periods=int(lookback_weeks*0.8)).max()
        is_price_breakout = df['close_W'] > period_high

        is_volume_breakout = df['volume_W'] > (df[vol_ma_col] * volume_multiplier)

        is_trix_ok = pd.Series(True, index=df.index)
        if trix_confirm:
            is_trix_ok = df[trix_col] > df[trix_signal_col]

        final_signal = is_price_breakout & is_volume_breakout & is_trix_ok

        last = df.iloc[-1]
        c1 = is_price_breakout.iloc[-1]
        c2 = is_volume_breakout.iloc[-1]
        c3 = is_trix_ok.iloc[-1]
        
        # print(f"    - 条件1 (突破年线): {'[✓]' if c1 else '[✗]'} (收盘价: {last.get('close_W', 0):.2f} vs 前{lookback_weeks}周高点: {period_high.iloc[-1]:.2f})")
        # print(f"    - 条件2 (2倍放量): {'[✓]' if c2 else '[✗]'} (成交量: {last.get('volume_W', 0):.0f} vs 阈值: {(last.get(vol_ma_col, 0) * volume_multiplier):.0f})")
        # if trix_confirm:
        #     print(f"    - 条件3 (TRIX确认): {'[✓]' if c3 else '[✗]'} (TRIX: {last.get(trix_col, 0):.2f} > 信号线: {last.get(trix_signal_col, 0):.2f})")
        
        # print(f"    - 结论: 最新一周信号为 [{'触发' if final_signal.iloc[-1] else '未触发'}]")
        
        return final_signal.fillna(False)

    def _check_resistance_rejection(self, df: pd.DataFrame, resistance_col: str, params: dict, source_name: str) -> pd.Series:
        """辅助函数: 检查在给定压力列上的拒绝信号"""
        print(f"  - 检查子项: [{source_name}]")
        volume_multiplier = params.get('volume_multiplier', 1.5)
        vol_ma_period = self.indicator_cfg.get('vol_ma', {}).get('periods', [5, 20, 55])[-1]
        vol_ma_col = f'VOL_MA_{vol_ma_period}_W'
        required_cols = [resistance_col, vol_ma_col, 'open_W', 'high_W', 'close_W', 'volume_W']
        if not self._check_dependencies(df, required_cols, log_details=True):
            print(f"    - 结论: [失败] 缺少必要列")
            return pd.Series(False, index=df.index)
        is_near_resistance = df['high_W'] >= df[resistance_col]
        is_long_upper_shadow = (df['high_W'] - df[['open_W', 'close_W']].max(axis=1)) > (df['high_W'] - df['low_W']) * 0.5
        is_high_volume = df['volume_W'] > df[vol_ma_col] * volume_multiplier
        is_closing_lower = df['close_W'] < df[['open_W', 'close_W']].mean(axis=1)
        final_signal = (is_near_resistance & is_long_upper_shadow & is_high_volume & is_closing_lower)
        last = df.iloc[-1]
        c1 = is_near_resistance.iloc[-1]
        c2 = is_long_upper_shadow.iloc[-1]
        c3 = is_high_volume.iloc[-1]
        c4 = is_closing_lower.iloc[-1]
        # print(f"    - 条件1 (触及压力): {'[✓]' if c1 else '[✗]'} (最高价: {last.get('high_W', 0):.2f} vs 压力: {last.get(resistance_col, 0):.2f})")
        # print(f"    - 条件2 (长上影线): {'[✓]' if c2 else '[✗]'}")
        # print(f"    - 条件3 (放出大量): {'[✓]' if c3 else '[✗]'} (成交量: {last.get('volume_W', 0):.0f} vs 阈值: {(last.get(vol_ma_col, 0) * volume_multiplier):.0f})")
        # print(f"    - 条件4 (收盘偏低): {'[✓]' if c4 else '[✗]'}")
        # print(f"    - 小结: [{source_name}] {'触发' if final_signal.iloc[-1] else '未触发'}")
        return final_signal.fillna(False)

    def _check_dependencies(self, df: pd.DataFrame, cols: list, log_details: bool = False) -> bool:
        """检查DataFrame中是否存在所有必需的列。"""
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            if log_details:
                print(f"      - [依赖检查] 失败! 缺少以下必需列: {missing_cols}")
            if not hasattr(self, '_warned_missing_cols_weekly'):
                self._warned_missing_cols_weekly = set()
            if tuple(missing_cols) not in self._warned_missing_cols_weekly:
                 logger.warning(f"周线策略缺少必需列: {missing_cols}，相关剧本将跳过。")
                 self._warned_missing_cols_weekly.add(tuple(missing_cols))
            return False
        return True
