// 新建文件: static/js/stock_detail.js

document.addEventListener('DOMContentLoaded', function () {
    // 初始化K线图
    const klineChartDom = document.getElementById('kline-chart');
    if (klineChartDom) {
        const klineChart = echarts.init(klineChartDom, 'dark'); // 使用暗色主题
        const klineOption = getKlineChartOption(klineChartData);
        klineChart.setOption(klineOption);
    }

    // 初始化得分图
    const scoreChartDom = document.getElementById('score-chart');
    if (scoreChartDom) {
        const scoreChart = echarts.init(scoreChartDom, 'dark');
        const scoreOption = getScoreChartOption(scoreChartData);
        scoreChart.setOption(scoreOption);
    }
});

// K线图配置函数
function getKlineChartOption(data) {
    return {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' }
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
        series: [{
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
            }
        }]
    };
}
