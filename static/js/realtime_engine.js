// static\js\realtime_engine.js

document.addEventListener('DOMContentLoaded', function () {
    // =========================================================================
    // === 盘中引擎 (realtime_engine.html) 功能 ==============================
    // =========================================================================
    function initializeRealtimeEnginePage() {
        const signalStream = document.getElementById('signal-stream');
        // 卫兵子句：如果找不到核心元素，说明不是这个页面，直接退出
        if (!signalStream) {
            return;
        }
        console.log('正在初始化【盘中引擎实时监控】页面功能...');
        const signalCountSpan = document.getElementById('signal-count');
        const noSignalsMessage = document.getElementById('no-signals-message');
        // WebSocket 连接 (复用主控台的连接逻辑)
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsPath = `${wsProtocol}//${window.location.host}/ws/dashboard/`;
        let socket;
        function connectWebSocket() {
            console.log('正在尝试连接 WebSocket 以接收盘中引擎信号...');
            socket = new WebSocket(wsPath);
            socket.onopen = function (e) {
                console.log('WebSocket 连接已建立');
            };
            socket.onmessage = function (e) {
                const data = JSON.parse(e.data);
                // 我们只关心盘中引擎的信号更新
                if (data.type === 'intraday_signal_update') {
                    console.log('接收到盘中引擎信号:', data.payload);
                    addSignalCard(data.payload);
                }
            };
            socket.onclose = function (e) {
                console.error('WebSocket 连接意外关闭。5秒后尝试重新连接...', e.reason);
                setTimeout(connectWebSocket, 5000);
            };
        }
        function addSignalCard(signal) {
            // 如果“无信号”提示存在，则移除它
            if (noSignalsMessage && noSignalsMessage.style.display !== 'none') {
                noSignalsMessage.style.display = 'none';
            }
            const card = document.createElement('div');
            const signalTypeClass = signal.signal_type.toLowerCase().replace(/_/g, '-');
            card.className = `signal-card ${signalTypeClass}`;
            let icon = 'ℹ️';
            if (signal.signal_type === 'BUY') icon = '🚀';
            else if (signal.signal_type === 'TAKE_PROFIT_T') icon = '💰';
            else if (signal.signal_type === 'BUY_DIP_T') icon = '📥';
            else if (signal.signal_type === 'STOP_LOSS') icon = '🛑';
            // 解析时间
            const signalTime = new Date(signal.entry_time).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            card.innerHTML = `
            <div class="signal-icon">${icon}</div>
            <div class="signal-content">
                <div class="signal-header">
                    <span class="signal-stock">${signal.stock_code}</span>
                    <span class="signal-time">${signalTime}</span>
                </div>
                <div class="signal-reason">
                    ${signal.reason} @ ${parseFloat(signal.entry_price).toFixed(2)}
                </div>
                <div class="signal-playbook">
                    剧本: ${signal.playbook}
                </div>
            </div>
        `;
            // 将新卡片插入到最顶部
            signalStream.prepend(card);
            // 更新信号总数
            if (signalCountSpan) {
                signalCountSpan.textContent = parseInt(signalCount_span.textContent || '0') + 1;
            }
        }
        // 启动WebSocket连接
        connectWebSocket();
    }
});
