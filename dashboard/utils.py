# dashboard/utils.py

def extract_score_details(analysis_text: str) -> str:
    """
    从完整的分析文本中提取【评分细则】部分。
    例如: "【信号类型】...【评分细则】(1)形态分: 50..." -> "(1)形态分: 50..."
    
    :param analysis_text: 包含评分细则的完整分析文本
    :return: 只包含评分细则的字符串，如果找不到则返回原始文本的后半部分或空字符串
    """
    if not isinstance(analysis_text, str):
        return ""
    
    marker = "【评分细则】"
    if marker in analysis_text:
        # 返回标记之后的所有内容
        return analysis_text.split(marker, 1)[1].strip()
    
    # 如果没有找到标记，可以根据情况返回一个默认值或部分文本
    return analysis_text # 或者返回 ""
