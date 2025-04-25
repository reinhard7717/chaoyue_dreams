from stock_models.stock_basic import StockInfo

# 1. 生成主键映射
code_map = {}
for stock in StockInfo.objects.all():
    old_code = stock.stock_code
    if old_code.endswith('.SH') or old_code.endswith('.SZ'):
        continue
    if old_code.startswith('6'):
        new_code = f"{old_code}.SH"
    else:
        new_code = f"{old_code}.SZ"
    code_map[old_code] = new_code

with open('stock_code_update.sql', 'w', encoding='utf-8') as f:
    # 2. 生成插入新主键的SQL
    f.write("-- 插入新主键StockInfo记录\n")
    for old_code, new_code in code_map.items():
        # 跳过已存在的新主键
        if StockInfo.objects.filter(stock_code=new_code).exists():
            continue
        old = StockInfo.objects.get(stock_code=old_code)
        def esc(val):
            if val is None:
                return 'NULL'
            return "'{}'".format(str(val).replace("'", "''"))
        f.write(
            f"INSERT INTO stock_info (stock_code, stock_name, area, industry, full_name, en_name, cn_spell, market_type, exchange, currency_type, list_status, list_date, delist_date, is_hs, actual_controller, actual_controller_type, circulating_shares) VALUES ({esc(new_code)}, {esc(old.stock_name)}, {esc(old.area)}, {esc(old.industry)}, {esc(old.full_name)}, {esc(old.en_name)}, {esc(old.cn_spell)}, {esc(old.market_type)}, {esc(old.exchange)}, {esc(old.currency_type)}, {esc(old.list_status)}, {esc(old.list_date)}, {esc(old.delist_date)}, {esc(old.is_hs)}, {esc(old.actual_controller)}, {esc(old.actual_controller_type)}, {old.circulating_shares if old.circulating_shares is not None else 'NULL'});\n"
        )

    # 3. 生成所有外键表的批量UPDATE SQL
    tables = [
        'new_stock_calendar',
        'st_stock_list',
        'company_info',
        'stock_belongs_index',
        'quarterly_profit',
        'stock_time_trade',
        'stock_analysis',
        'stock_realtime_data',
        'stock_level5_data',
        'stock_trade_detail',
        'stock_time_deal',
        'stock_price_percent',
        'stock_big_deal',
        'abnormal_movement',
    ]
    f.write("\n-- 批量更新所有外键表\n")
    for table in tables:
        for old_code, new_code in code_map.items():
            f.write(f"UPDATE {table} SET stock_id='{new_code}' WHERE stock_id='{old_code}';\n")

    # 4. 生成删除旧主键的SQL
    f.write("\n-- 删除旧主键StockInfo记录\n")
    for old_code in code_map.keys():
        f.write(f"DELETE FROM stock_info WHERE stock_code='{old_code}';\n")

print("SQL脚本已生成到 stock_code_update.sql")
