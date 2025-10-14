// 文件: static/js/stock_detail.js

// K线图配置函数
function getKlineChartOption(data) {
    return {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross'
            },
            // 部署“情报识别协议”，重构 formatter 函数
            formatter: function (params) {
                // 增加防御性编程，处理 params 可能为空的边缘情况
                if (!Array.isArray(params) || params.length === 0) {
                    return '';
                }
                let klineParams = null;
                let volumeParams = null;
                // 步骤一：情报识别。遍历所有数据包，根据 seriesName 确认其身份。
                params.forEach(param => {
                    if (param.seriesName === '日K') {
                        klineParams = param;
                    } else if (param.seriesName === '成交量') {
                        volumeParams = param;
                    }
                });
                // 步骤二：构建情报简报。
                // 从任意一个数据包中获取统一的日期信息。
                let tooltipHtml = `${params[0].axisValue}<br/>`;
                // 如果识别到了K线情报，则添加 OHLC 数据。
                if (klineParams && klineParams.data && Array.isArray(klineParams.data) && klineParams.data.length > 4) {
                    const ohlc = klineParams.data;
                    tooltipHtml += `开盘: <span style="font-weight:bold;">${ohlc[1].toFixed(2)}</span><br/>`;
                    tooltipHtml += `收盘: <span style="font-weight:bold;">${ohlc[2].toFixed(2)}</span><br/>`;
                    tooltipHtml += `最低: <span style="font-weight:bold; color: #14b143;">${ohlc[3].toFixed(2)}</span><br/>`;
                    tooltipHtml += `最高: <span style="font-weight:bold; color: #ef232a;">${ohlc[4].toFixed(2)}</span><br/>`;
                }
                // 如果识别到了成交量情报，则添加成交量数据。
                if (volumeParams && volumeParams.value !== undefined) {
                    // volumeParams.marker 是ECharts提供的小圆点标记
                    tooltipHtml += `${volumeParams.marker} ${volumeParams.seriesName}: <span style="font-weight:bold;">${volumeParams.value}</span>`;
                }
                return tooltipHtml;
            }
        },
        legend: {
            data: ['日K', '成交量'],
            textStyle: { color: '#ccc' }
        },
        grid: [
            { left: '10%', right: '8%', height: '50%' },
            { left: '10%', right: '8%', top: '65%', height: '16%' }
        ],
        xAxis: [
            {
                type: 'category',
                data: data.dates,
                axisLine: { lineStyle: { color: '#8392A5' } }
            },
            {
                type: 'category',
                gridIndex: 1,
                data: data.dates,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { show: false }
            }
        ],
        yAxis: [
            {
                scale: true,
                axisLine: { lineStyle: { color: '#8392A5' } },
                splitLine: { show: false }
            },
            {
                scale: true,
                gridIndex: 1,
                axisLabel: { show: false },
                axisLine: { show: false },
                axisTick: { show: false },
                splitLine: { show: false }
            }
        ],
        dataZoom: [
            { type: 'inside', xAxisIndex: [0, 1], start: 0, end: 100 },
            { show: true, xAxisIndex: [0, 1], type: 'slider', top: '90%', start: 0, end: 100 }
        ],
        series: [
            {
                name: '日K',
                type: 'candlestick',
                data: data.values,
                itemStyle: {
                    color: '#ef232a',
                    color0: '#14b143',
                    borderColor: '#ef232a',
                    borderColor0: '#14b143'
                }
            },
            {
                name: '成交量',
                type: 'bar',
                xAxisIndex: 1,
                yAxisIndex: 1,
                data: data.volumes,
                itemStyle: {
                    color: function (params) {
                        // 根据K线的涨跌决定成交量柱子的颜色
                        const ohlc = data.values[params.dataIndex];
                        return ohlc[1] >= ohlc[0] ? '#ef232a' : '#14b143';
                    }
                }
            }
        ]
    };
}

// 得分图配置函数
function getScoreChartOption(data) {
    return {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' }
        },
        // 增加图例以区分先知信号
        legend: {
            data: ['策略得分', '先知信号'],
            textStyle: { color: '#ccc' }
        },
        xAxis: {
            type: 'category',
            data: data.dates,
            axisLine: { lineStyle: { color: '#8392A5' } }
        },
        yAxis: {
            type: 'value',
            scale: true,
            axisLine: { lineStyle: { color: '#8392A5' } },
            splitLine: { lineStyle: { color: '#363636' } }
        },
        dataZoom: [
            { type: 'inside', start: 0, end: 100 },
            { show: true, type: 'slider', top: '90%', start: 0, end: 100 }
        ],
        // 增加系列和标记点
        series: [
            {
                name: '策略得分',
                type: 'line',
                smooth: true,
                data: data.scores,
                connectNulls: true, // 连接空值点
                itemStyle: { color: '#5470c6' },
                lineStyle: { width: 2 },
                markLine: {
                    silent: true,
                    data: [{ yAxis: 0, lineStyle: { color: '#999' } }]
                },
                // 在得分曲线上增加“先知信号”的标记点
                markPoint: {
                    symbol: 'pin', // 标记点样式
                    symbolSize: 50, // 标记点大小
                    label: {
                        fontSize: 10
                    },
                    data: data.prophet_signals || [] // 使用从后端传来的标记点数据
                }
            },
            // 一个空的“散点”系列，仅用于在图例中显示“先知信号”
            {
                name: '先知信号',
                type: 'scatter',
                itemStyle: {
                    color: '#ffc107'
                },
                data: []
            }
        ]
    };
}
