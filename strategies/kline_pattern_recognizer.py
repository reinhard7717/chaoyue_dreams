# 文件: strategies/kline_pattern_recognizer.py
import pandas as pd
import numpy as np
from typing import Dict

class KlinePatternRecognizer:
    """
    一个可配置的、向量化的K线形态识别器 (专业版 V2.0 - 双层识别)。
    
    功能:
    - 新增“标准级(Decent)”和“完美级(Perfect)”双层识别机制，解决“数学定义过于严苛”的问题。
    - 策略可以根据形态的完美程度，进行差异化计分。
    - 接收包含OHLCV数据的DataFrame。
    - 识别清单中所有12大类经典K线形态。
    - 返回一个增加了多个形态识别布尔列的DataFrame。
    - 所有形态的关键阈值均可通过外部参数调整。
    
    命名约定:
    - kline_s_*: 单K线形态 (Single)
    - kline_c_*: 多K线组合形态 (Combination)
    - _bullish: 看涨信号
    - _bearish: 看跌信号
    - _reversal: 反转形态
    - _continuation: 持续形态
    """

    def __init__(self, params: Dict = None):
        """
        使用传入的参数进行初始化。
        如果未提供参数，则使用内部定义的默认值。
        """
        # 合并默认参数和用户传入参数
        defaults = self._get_default_params()
        self.params = {**defaults, **(params if params is not None else {})}

    def _get_default_params(self) -> Dict:
        """
        【修改】定义所有形态识别的默认阈值，区分为“标准级”和“完美级”。
        """
        return {
            # 通用参数
            "body_ratios": {"decent": 0.5, "perfect": 0.7},  # 长实体定义
            "small_body_ratios": {"decent": 0.2, "perfect": 0.1}, # 小实体定义
            "doji_body_ratios": {"decent": 0.1, "perfect": 0.05},   # 十字星实体定义
            
            # 特定形态参数
            "hammer": {
                "lower_shadow_ratio": {"decent": 1.8, "perfect": 2.5}, # 锤子线下影线/实体
                "upper_shadow_ratio": {"decent": 1.2, "perfect": 0.8}  # 锤子线上影线/实体
            },
            "engulfing": {
                "prev_body_ratio": {"decent": 1.1, "perfect": 1.5} # 吞没形态：当前实体/前一实体
            },
            "piercing": {
                "penetration_ratio": {"decent": 0.4, "perfect": 0.6} # 刺透形态：刺入深度
            }
        }

    def identify_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【修改 V2.1 - Bug修复版】运行所有K线形态识别。
        - 修复了“三只乌鸦”形态中 prev2 未定义的严重Bug。
        - 统一并优化了“三白兵”和“三只乌鸦”的识别逻辑，使其更清晰、健壮。
        """
        # --- 0. 基础数据和衍生变量预计算 ---
        op, hi, lo, cl = df['open_D'], df['high_D'], df['low_D'], df['close_D']
        kline_range = (hi - lo).replace(0, 0.0001)
        body_size = abs(cl - op)
        body_size_safe = body_size.replace(0, 0.0001) # 用于除法的安全版本
        upper_shadow = hi - np.maximum(op, cl)
        lower_shadow = np.minimum(op, cl) - lo
        is_green = cl > op
        is_red = cl < op
        
        # 双层特征预计算
        is_long_body_decent = body_size / kline_range >= self.params['body_ratios']['decent']
        is_long_body_perfect = body_size / kline_range >= self.params['body_ratios']['perfect']
        is_small_body_decent = body_size / kline_range <= self.params['small_body_ratios']['decent']
        is_small_body_perfect = body_size / kline_range <= self.params['small_body_ratios']['perfect']
        is_doji_body_decent = body_size / kline_range <= self.params['doji_body_ratios']['decent']
        is_doji_body_perfect = body_size / kline_range <= self.params['doji_body_ratios']['perfect']
        
        #定义回溯一天(prev1)和两天(prev2)的数据字典
        prev1 = { 'op': op.shift(1), 'hi': hi.shift(1), 'lo': lo.shift(1), 'cl': cl.shift(1),
                  'body_size': body_size.shift(1), 'is_green': is_green.shift(1), 'is_red': is_red.shift(1),
                  'is_long_body_decent': is_long_body_decent.shift(1), 'is_long_body_perfect': is_long_body_perfect.shift(1) }
        
        prev2 = { 'op': op.shift(2), 'hi': hi.shift(2), 'lo': lo.shift(2), 'cl': cl.shift(2),
                  'body_size': body_size.shift(2), 'is_green': is_green.shift(2), 'is_red': is_red.shift(2),
                  'is_long_body_decent': is_long_body_decent.shift(2), 'is_long_body_perfect': is_long_body_perfect.shift(2) }
        
        # --- 1. 十字星 (Doji) ---
        df['kline_s_doji_decent'] = is_doji_body_decent
        df['kline_s_doji_perfect'] = is_doji_body_perfect

        # --- 2. 吞没形态 (Engulfing) ---
        engulf_base_bullish = prev1['is_red'] & is_green & (cl > prev1['op']) & (op < prev1['cl'])
        engulf_base_bearish = prev1['is_green'] & is_red & (cl < prev1['op']) & (op > prev1['cl'])
        df['kline_c_bullish_engulfing_decent'] = engulf_base_bullish & (body_size > prev1['body_size'] * self.params['engulfing']['prev_body_ratio']['decent'])
        df['kline_c_bullish_engulfing_perfect'] = engulf_base_bullish & (body_size > prev1['body_size'] * self.params['engulfing']['prev_body_ratio']['perfect'])
        df['kline_c_bearish_engulfing_decent'] = engulf_base_bearish & (body_size > prev1['body_size'] * self.params['engulfing']['prev_body_ratio']['decent'])
        df['kline_c_bearish_engulfing_perfect'] = engulf_base_bearish & (body_size > prev1['body_size'] * self.params['engulfing']['prev_body_ratio']['perfect'])

        # --- 3. 锤子线 (Hammer) / 上吊线 (Hanging Man) ---
        hammer_shape_decent = (lower_shadow >= body_size_safe * self.params['hammer']['lower_shadow_ratio']['decent']) & \
                              (upper_shadow <= body_size_safe * self.params['hammer']['upper_shadow_ratio']['decent'])
        hammer_shape_perfect = (lower_shadow >= body_size_safe * self.params['hammer']['lower_shadow_ratio']['perfect']) & \
                               (upper_shadow <= body_size_safe * self.params['hammer']['upper_shadow_ratio']['perfect'])
        df['kline_s_hammer_shape_decent'] = hammer_shape_decent & is_green
        df['kline_s_hammer_shape_perfect'] = hammer_shape_perfect & is_green
        df['kline_s_hanging_man_shape_decent'] = hammer_shape_decent & is_red
        df['kline_s_hanging_man_shape_perfect'] = hammer_shape_perfect & is_red

        # --- 4. 星线形态 (Morning/Evening Star) ---
        is_star_body = is_small_body_decent.shift(1)
        morning_star_gap = np.maximum(prev1['op'], prev1['cl']) < prev2['cl']
        df['kline_c_morning_star'] = prev2['is_red'] & prev2['is_long_body_decent'] & is_star_body & morning_star_gap & \
                                     is_green & is_long_body_decent & (cl > (prev2['op'] + prev2['cl']) / 2)
        evening_star_gap = np.minimum(prev1['op'], prev1['cl']) > prev2['cl']
        df['kline_c_evening_star'] = prev2['is_green'] & prev2['is_long_body_decent'] & is_star_body & evening_star_gap & \
                                     is_red & is_long_body_decent & (cl < (prev2['op'] + prev2['cl']) / 2)

        # --- 5. 刺透形态 (Piercing) / 乌云盖顶 (Dark Cloud) ---
        piercing_base = prev1['is_red'] & prev1['is_long_body_decent'] & is_green & (op < prev1['lo']) & (cl < prev1['op'])
        df['kline_c_piercing_line_decent'] = piercing_base & (cl > prev1['cl'] + prev1['body_size'] * self.params['piercing']['penetration_ratio']['decent'])
        df['kline_c_piercing_line_perfect'] = piercing_base & (cl > prev1['cl'] + prev1['body_size'] * self.params['piercing']['penetration_ratio']['perfect'])
        
        dark_cloud_base = prev1['is_green'] & prev1['is_long_body_decent'] & is_red & (op > prev1['hi']) & (cl > prev1['op'])
        df['kline_c_dark_cloud_cover_decent'] = dark_cloud_base & (cl < prev1['cl'] - prev1['body_size'] * self.params['piercing']['penetration_ratio']['decent'])
        df['kline_c_dark_cloud_cover_perfect'] = dark_cloud_base & (cl < prev1['cl'] - prev1['body_size'] * self.params['piercing']['penetration_ratio']['perfect'])

        # --- 6. 三白兵 (Three White Soldiers) ---
        is_day1_green = prev2['is_green'] & prev2['is_long_body_decent']
        is_day2_green = prev1['is_green'] & prev1['is_long_body_decent']
        is_day3_green = is_green & is_long_body_decent
        
        df['kline_c_three_white_soldiers'] = is_day1_green & is_day2_green & is_day3_green & \
                                             (prev1['cl'] > prev2['cl']) & (cl > prev1['cl']) & \
                                             (prev1['op'] < prev2['cl']) & (prev1['op'] > prev2['op']) & \
                                             (op < prev1['cl']) & (op > prev1['op'])

        # --- 7. 三只乌鸦 (Three Black Crows) ---
        is_day1_red = prev2['is_red'] & prev2['is_long_body_decent']
        is_day2_red = prev1['is_red'] & prev1['is_long_body_decent']
        is_day3_red = is_red & is_long_body_decent

        df['kline_c_three_black_crows'] = is_day1_red & is_day2_red & is_day3_red & \
                                          (prev1['cl'] < prev2['cl']) & (cl < prev1['cl']) & \
                                          (prev1['op'] > prev2['cl']) & (prev1['op'] < prev2['op']) & \
                                          (op > prev1['cl']) & (op < prev1['op'])

        # --- 7. 孕线形态 (Harami) ---
        is_body_inside = (np.maximum(op, cl) < prev1['op']) & (np.minimum(op, cl) > prev1['cl'])
        is_body_inside_rev = (np.maximum(op, cl) < prev1['cl']) & (np.minimum(op, cl) > prev1['op']) # 反向
        
        df['kline_c_bullish_harami'] = prev1['is_red'] & prev1['is_long_body_decent'] & is_green & is_small_body_decent & is_body_inside
        df['kline_c_bearish_harami'] = prev1['is_green'] & prev1['is_long_body_decent'] & is_red & is_small_body_decent & is_body_inside_rev
        df['kline_c_harami_cross'] = prev1['is_long_body_decent'] & is_doji_body_decent & (is_body_inside | is_body_inside_rev)

        # --- 8. 镊子顶 (Tweezer Top) / 镊子底 (Tweezer Bottom) ---
        # 核心是高点/低点几乎一致，前后K线颜色相反
        df['kline_c_tweezer_top'] = (abs(hi - prev1['hi']) / hi < 0.001) & prev1['is_green'] & is_red
        df['kline_c_tweezer_bottom'] = (abs(lo - prev1['lo']) / lo < 0.001) & prev1['is_red'] & is_green

        # --- 9. 光头光脚K线 (Marubozu) ---
        is_marubozu = (upper_shadow / kline_range < 0.05) & (lower_shadow / kline_range < 0.05)
        df['kline_s_marubozu_white'] = is_marubozu & is_green
        df['kline_s_marubozu_black'] = is_marubozu & is_red

        # --- 10. 上升三法 (Rising) / 下降三法 (Falling Three Methods) ---
        # 这是一个复杂的5日形态，这里实现一个简化但有效的版本
        day1_long_green = is_long_body_decent.shift(4) & is_green.shift(4)
        three_reds_in_body = (is_red.shift(3) & is_red.shift(2) & is_red.shift(1)) & \
                             (hi.shift(3) < hi.shift(4)) & (lo.shift(3) > lo.shift(4)) & \
                             (hi.shift(2) < hi.shift(4)) & (lo.shift(2) > lo.shift(4)) & \
                             (hi.shift(1) < hi.shift(4)) & (lo.shift(1) > lo.shift(4))
        day5_breakout_green = is_long_body_decent & is_green & (cl > cl.shift(4))
        df['kline_c_rising_three_methods'] = day1_long_green & three_reds_in_body & day5_breakout_green

        day1_long_red = is_long_body_decent.shift(4) & is_red.shift(4)
        three_greens_in_body = (is_green.shift(3) & is_green.shift(2) & is_green.shift(1)) & \
                               (hi.shift(3) < hi.shift(4)) & (lo.shift(3) > lo.shift(4)) & \
                               (hi.shift(2) < hi.shift(4)) & (lo.shift(2) > lo.shift(4)) & \
                               (hi.shift(1) < hi.shift(4)) & (lo.shift(1) > lo.shift(4))
        day5_breakout_red = is_long_body_decent & is_red & (cl < cl.shift(4))
        df['kline_c_falling_three_methods'] = day1_long_red & three_greens_in_body & day5_breakout_red

        # --- 11 & 12. 其他组合形态 ---
        # 向上跳空两只乌鸦 (看跌)
        df['kline_c_upside_gap_two_crows'] = is_green.shift(2) & (op.shift(1) > cl.shift(2)) & is_red.shift(1) & \
                                             (op > op.shift(1)) & (op < cl.shift(1)) & is_red & (cl < cl.shift(2))
        
        # 向下跳空并列阴线 (看涨)
        df['kline_c_downside_tasuki_gap'] = is_red.shift(2) & (op.shift(1) < cl.shift(2)) & is_red.shift(1) & \
                                            (op > cl.shift(1)) & (op < op.shift(1)) & is_green & (cl > op.shift(1))

        # 看涨反击线
        df['kline_c_bullish_counterattack'] = prev1['is_red'] & prev1['is_long_body_decent'] & is_green & is_long_body_decent & \
                                             (abs(cl - prev1['cl']) / cl < 0.001)
        # 看跌反击线
        df['kline_c_bearish_counterattack'] = prev1['is_green'] & prev1['is_long_body_decent'] & is_red & is_long_body_decent & \
                                              (abs(cl - prev1['cl']) / cl < 0.001)
        
        # 看涨分离线
        df['kline_c_bullish_separating_lines'] = prev1['is_red'] & is_green & is_marubozu & \
                                                 (abs(op - prev1['op']) / op < 0.001)
        # 看跌分离线
        df['kline_c_bearish_separating_lines'] = prev1['is_green'] & is_red & is_marubozu & \
                                                  (abs(op - prev1['op']) / op < 0.001)

        # print(f"    [调试-K线识别器]: 已完成所有形态分析，为DataFrame添加了新的布尔列。")
        return df

