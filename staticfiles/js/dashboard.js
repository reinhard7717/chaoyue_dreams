document.addEventListener('DOMContentLoaded', function () {
    const favoriteMessagesList = document.querySelector('#favorite-stock-messages .message-list');
    const strategyMessagesList = document.querySelector('#strategy-messages .message-list');
    const searchInput = document.getElementById('stock-search-input');
    const searchResultsContainer = document.getElementById('search-results');
    const addFavoriteForm = document.getElementById('add-favorite-form');
    const favoritesTbody = document.getElementById('favorites-tbody');
    const favoritesLoading = document.getElementById('favorites-loading');
    const favoritesEmpty = document.getElementById('favorites-empty');
    const addFavoriteBtn = document.getElementById('add-favorite-btn'); // 获取添加按钮
    const MAX_FAV_MESSAGES = 20;
    const MAX_STRATEGY_MESSAGES = 30;
    // --- WebSocket 连接 ---
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsPath = `${wsProtocol}//${window.location.host}/ws/dashboard/`;
    let socket;
    function connectWebSocket() {
        console.log('Attempting to connect WebSocket...');
        socket = new WebSocket(wsPath);
        socket.onopen = function (e) {
            console.log('WebSocket connection established');
            // 连接成功后可以请求初始数据或等待推送
            // fetchInitialFavorites(); // 通过 API 获取初始列表
        };
        socket.onmessage = function (e) {
            const data = JSON.parse(e.data);
            console.log('Data received:', data);
            switch (data.type) {
                case 'favorite_message': // 自选股相关消息
                    addMessage(favoriteMessagesList, data.payload, MAX_FAV_MESSAGES);
                    break;
                case 'strategy_message': // 通用策略消息
                    addMessage(strategyMessagesList, data.payload, MAX_STRATEGY_MESSAGES);
                    break;
                case 'stock_tick_update': // 实时tick行情推送
                    updateStockRow(data.payload);
                    break;
                case 'favorites_update': // 全量刷新自选股列表
                    renderFavoritesTable(data.payload);
                    break;
                case 'favorite_added_with_data': // 单行更新自选股列表
                    addStockRow(data.payload);
                    break;
                default:
                    console.warn('Unknown message type:', data.type);
            }
        };
        socket.onclose = function (e) {
            console.error('WebSocket connection closed unexpectedly. Attempting to reconnect...', e.reason);
            // 设置延迟重连，避免服务器压力过大
            setTimeout(connectWebSocket, 5000); // 5秒后重试
        };
        socket.onerror = function (err) {
            console.error('WebSocket error:', err);
            // 错误发生时也会触发 onclose，所以重连逻辑在 onclose 中处理
        };
    }
    // --- 消息处理 ---
    function addMessage(listElement, messageData, maxSize) {
        if (!listElement) return;
        const li = document.createElement('li');
        // 根据 messageData 构建消息内容 (需要后端定义好 payload 结构)
        // 示例 payload: { timestamp: "10:45:01", code: "600036", name: "招商银行", signal_type: "buy", text: "策略A触发买入信号, 已挂单 35.50" }
        const timestamp = messageData.timestamp || new Date().toLocaleTimeString();
        const codeHtml = messageData.code ? `<span class="stock-code">${messageData.code}</span>` : '';
        const nameHtml = messageData.name ? `${messageData.name} - ` : '';
        const signalClass = messageData.signal_type ? `signal-${messageData.signal_type}` : 'signal-info'; // 默认为 info
        const signalHtml = `<span class="signal ${signalClass}">${messageData.text}</span>`;
        li.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${codeHtml} ${nameHtml} ${signalHtml}`;
        listElement.prepend(li); // 在列表顶部添加新消息
        // 保持列表大小
        while (listElement.children.length > maxSize) {
            listElement.removeChild(listElement.lastChild);
        }
    }
    // --- 股票搜索与添加 ---
    let searchDebounceTimer;
    searchInput.addEventListener('input', () => {
        clearTimeout(searchDebounceTimer);
        const query = searchInput.value.trim();
        if (query.length > 1) { // 至少输入2个字符才开始搜索
            searchDebounceTimer = setTimeout(() => {
                performSearch(query);
            }, 300); // 延迟 300ms 执行搜索
        } else {
            searchResultsContainer.innerHTML = ''; // 清空结果
            searchResultsContainer.style.display = 'none';
        }
    });
    // 阻止搜索表单的默认提交行为 (如果它是 <form>)
    if (addFavoriteForm) {
        addFavoriteForm.addEventListener('submit', (event) => {
            event.preventDefault();
        });
    }
    async function performSearch(query) {
        console.log(`Searching for: ${query}`);
        searchResultsContainer.innerHTML = '<div class="search-result-item empty">正在搜索...</div>'; // 显示加载状态
        searchResultsContainer.style.display = 'block'; // 显示容器
        try {
            // 使用 GET 请求，不需要 CSRF token
            const response = await fetch(`/dashboard/api/search/?q=${encodeURIComponent(query)}`, {
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'Accept': 'application/json', // 明确要求 JSON
                }
            });
            if (!response.ok) {
                // 尝试解析错误信息
                let errorMsg = `搜索失败 (HTTP ${response.status})`;
                try {
                    const errorData = await response.json();
                    errorMsg = Object.values(errorData).flat().join(' ') || errorMsg;
                } catch (e) { /* 忽略解析错误 */ }
                throw new Error(errorMsg);
            }
            const results = await response.json();
            displaySearchResults(results);
        } catch (error) {
            console.error('Search failed:', error);
            searchResultsContainer.innerHTML = `<div class="search-result-item error">${error.message || '搜索出错'}</div>`;
            searchResultsContainer.style.display = 'block';
        }
    }
    function displaySearchResults(results) {
        searchResultsContainer.innerHTML = ''; // 清空旧结果 (包括加载状态)
        if (results.length === 0) {
            searchResultsContainer.innerHTML = '<div class="search-result-item empty">未找到相关股票</div>';
        } else {
            results.forEach(stock => {
                const item = document.createElement('div');
                item.classList.add('search-result-item');
                item.innerHTML = `<strong>${stock.stock_code}</strong> ${stock.stock_name}`;
                item.dataset.stockCode = stock.stock_code; // 存储股票代码
                // --- 添加点击事件监听器 ---
                item.addEventListener('click', () => {
                    addFavoriteStock(stock.stock_code); // 点击结果直接添加
                    searchInput.value = ''; // 清空输入框
                    searchResultsContainer.style.display = 'none'; // 隐藏结果
                    searchResultsContainer.innerHTML = ''; // 清空内容
                });
                // --- 事件监听器结束 ---
                searchResultsContainer.appendChild(item);
            });
        }
        searchResultsContainer.style.display = 'block'; // 确保容器可见
    }
    // 点击页面其他地方隐藏搜索结果
    document.addEventListener('click', function (event) {
        if (!searchResultsContainer.contains(event.target) && event.target !== searchInput) {
            searchResultsContainer.style.display = 'none';
        }
    });
    // 处理表单提交（如果不用点击结果添加）
    addFavoriteForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // 阻止表单默认提交
        const stockCode = searchInput.value.trim(); // 直接使用输入框内容
        if (stockCode) {
            addFavoriteStock(stockCode);
            searchInput.value = ''; // 清空输入框
            searchResultsContainer.style.display = 'none'; // 隐藏结果
        }
    });
    async function addFavoriteStock(stockCode) {
        console.log(`正在尝试添加自选股: ${stockCode}`);
        // 可以在这里加一个简单的视觉反馈，比如按钮禁用，但因为是点击结果触发，可能不需要
        showNotification(`正在添加 ${stockCode}...`, 'info'); // 显示添加中提示
        try {
            const csrfToken = getCookie('csrftoken');
            if (!csrfToken) {
                throw new Error("无法获取 CSRF token，请刷新页面重试。");
            }
            const response = await fetch('/dashboard/api/favorites/', { // POST 到 ViewSet 的根 URL
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({ stock_code: stockCode }) // 发送 stock_code
            });
            if (response.ok) {
                const newFavorite = await response.json(); // 可能不需要解析响应体了
                console.log('Favorite added:', newFavorite);
                showNotification(`股票 ${stockCode} 添加成功！`, 'success');
                // --- 不再手动操作表格，等待 WebSocket 推送 ---
            } else {
                let errorMsg = `添加股票 ${stockCode} 失败`;
                try {
                    const errorData = await response.json();
                    // 尝试提取后端返回的具体错误信息
                    if (errorData.stock_code) {
                        errorMsg = errorData.stock_code[0];
                    } else if (errorData.detail) {
                        errorMsg = errorData.detail;
                    } else {
                        errorMsg = Object.values(errorData).flat().join(' ') || errorMsg;
                    }
                } catch (e) { /* 忽略解析错误 */ }
                throw new Error(errorMsg); // 抛出错误以便下面捕获
            }
        } catch (error) {
            console.error('添加自选股失败:', error);
            showNotification(error.message || `添加股票 ${stockCode} 时出错`, 'error');
        } finally {
            // 可以在这里恢复按钮状态（如果之前禁用了）
        }
    }
    // --- 添加单个股票行 ---
    function addStockRow(favData) {
        if (!favoritesTbody) return;
        // 检查是否已存在 (以防万一重复推送)
        const existRow = favoritesTbody.querySelector(`tr[data-favorite-id="${favData.id}"]`);
        if (existRow) {
            console.warn(`Stock row with favorite ID ${favData.id} already exists. 刷新内容。`);
            updateStockRow(favData);
            flashRow(existRow);
            return;
        }
        favoritesEmpty.style.display = 'none'; // 隐藏空状态提示
        favoritesLoading.style.display = 'none'; // 确保加载状态隐藏
        const row = document.createElement('tr');
        row.dataset.stockCode = favData.code;
        row.dataset.favoriteId = favData.id; // 设置 favorite ID
        const changePercent = favData.change_percent;
        let percentClass = '';
        if (changePercent > 0) percentClass = 'positive';
        else if (changePercent < 0) percentClass = 'negative';
        const signal = favData.signal || { type: 'hold', text: 'N/A' };
        const signalClass = `signal-${signal.type || 'hold'}`;
        row.innerHTML = `
            <td class="stock-code">${favData.code || 'N/A'}</td>
            <td class="stock-name">${favData.name || 'N/A'}</td>
            <td class="price">${formatNumber(favData.current_price, 2)}</td>
            <td class="change-percent ${percentClass}">${formatPercent(changePercent)}</td>
            <td class="volume">${formatVolume(favData.volume)}</td>
            <td class="strategy-signal"><span class="signal ${signalClass}">${signal.text || 'N/A'}</span></td>
            <td class="actions">
                <button class="btn btn-secondary btn-sm action-btn" data-action="detail">详情</button>
                <button class="btn btn-danger btn-sm action-btn" data-action="remove">移除</button>
            </td>
        `;
        favoritesTbody.appendChild(row); // 添加新行到表格末尾
        flashRow(row); // 给新行一个闪烁效果
    }
    // --- 移除单个股票行 ---
    function removeStockRow(favoriteId) {
        if (!favoritesTbody) return;
        const rowToRemove = favoritesTbody.querySelector(`tr[data-favorite-id="${favoriteId}"]`);
        if (rowToRemove) {
            rowToRemove.remove();
            console.log(`Removed row with favorite ID: ${favoriteId}`);
            // 检查表格是否变为空
            if (favoritesTbody.children.length === 0) {
                favoritesEmpty.style.display = 'block';
            }
        } else {
            console.warn(`Could not find row with favorite ID ${favoriteId} to remove.`);
        }
    }
    // --- 自选股表格渲染与更新 (需要修改以存储 ID) ---
    function renderFavoritesTable(favoritesData) {
        if (!favoritesTbody) return;
        favoritesTbody.innerHTML = '';
        if (!favoritesData || favoritesData.length === 0) {
            favoritesLoading.style.display = 'none';
            favoritesEmpty.style.display = 'block';
            return;
        }
        favoritesEmpty.style.display = 'none';
        favoritesLoading.style.display = 'none';
        favoritesData.forEach(fav => {
            // fav 结构现在包含 id:
            // { id: 1, code: '600036', name: '招商银行', latest_price: null, ... }
            const row = document.createElement('tr');
            row.dataset.stockCode = fav.code;
            row.dataset.favoriteId = fav.id; // <--- 存储 Favorite ID
            row.dataset.stockName = fav.name; // 新增
            const changePercent = fav.change_percent;
            let percentClass = '';
            if (changePercent > 0) percentClass = 'positive';
            else if (changePercent < 0) percentClass = 'negative';
            const signal = fav.signal || { type: 'hold', text: 'N/A' };
            const signalClass = `signal-${signal.type || 'hold'}`;
            row.innerHTML = `
                <td class="stock-code">${fav.code || 'N/A'}</td>
                <td class="stock-name">${fav.name || 'N/A'}</td>
                <td class="price">${formatNumber(fav.current_price, 2)}</td>
                <td class="change-percent ${percentClass}">${formatPercent(changePercent)}</td>
                <td class="volume">${formatVolume(fav.volume)}</td>
                <td class="strategy-signal"><span class="signal ${signalClass}">${signal.text || 'N/A'}</span></td>
                <td class="actions">
                    <button class="btn btn-secondary btn-sm action-btn" data-action="detail">详情</button>
                    <button class="btn btn-danger btn-sm action-btn" data-action="remove">移除</button>
                </td>
            `;
            favoritesTbody.appendChild(row);
        });
    }
    function updateStockRow(updateData) {
        // updateData 结构: { code: '600036', latest_price: 35.48, change_percent: 0.90, volume: 155000, signal: { type: 'hold', text: '持有中' } }
        if (!favoritesTbody) return;
        const row = favoritesTbody.querySelector(`tr[data-stock-code="${updateData.code}"]`);
        if (!row) return; // 如果行不存在，忽略
        const priceCell = row.querySelector('.price');
        const percentCell = row.querySelector('.change-percent');
        const volumeCell = row.querySelector('.volume');
        const signalCell = row.querySelector('.strategy-signal .signal');
        if (priceCell) priceCell.textContent = formatNumber(updateData.current_price, 2);
        if (volumeCell) volumeCell.textContent = formatVolume(updateData.volume);
        if (percentCell) {
            const changePercent = updateData.change_percent;
            percentCell.textContent = formatPercent(changePercent);
            percentCell.className = 'change-percent'; // Reset class
            if (changePercent > 0) percentCell.classList.add('positive');
            else if (changePercent < 0) percentCell.classList.add('negative');
        }
        if (signalCell && updateData.signal) {
            const signal = updateData.signal;
            signalCell.textContent = signal.text || 'N/A';
            signalCell.className = 'signal'; // Reset class
            signalCell.classList.add(`signal-${signal.type || 'hold'}`);
        }
        // 可以添加闪烁效果提示更新
        flashRow(row);
    }
    // --- 表格删除操作 ---
    favoritesTbody.addEventListener('click', function (event) {
        const target = event.target;
        // 确保点击的是移除按钮本身或其内部元素
        const removeButton = target.closest('button[data-action="remove"]');
        if (removeButton) {
            const row = removeButton.closest('tr');
            console.log("Row:", row.dataset.stockName);
            const stockCode = row.dataset.stockCode;
            const stockName = row.dataset.stockName;
            const favoriteId = row.dataset.favoriteId; // <--- 从行的 data 属性获取 ID
            // 确保获取到了 ID
            if (favoriteId) {
                removeFavoriteStock(stockCode, stockName, favoriteId, row);
            } else {
                console.error(`无法获取股票 ${stockCode} 的 Favorite ID`);
                showNotification(`移除股票 ${stockCode} 时出错 (缺少ID)`, 'error');
            }
        } else if (target.closest('button[data-action="detail"]')) {
            const row = target.closest('tr');
            const stockCode = row.dataset.stockCode;
            console.log(`Show details for ${stockCode}`);
            // window.location.href = `/stocks/${stockCode}/`; // 跳转逻辑
        }
    });
    async function removeFavoriteStock(stockCode, stockName, favoriteId, rowElement) {
        if (!confirm(`确定要从自选中移除 ${stockCode} - ${stockName} 吗？`)) {
            return;
        }
        console.log(`Attempting to remove favorite: ${stockCode} (ID: ${favoriteId})`);
        const removeButton = rowElement.querySelector('button[data-action="remove"]');
        if (removeButton) {
            removeButton.disabled = true; // 禁用按钮
            removeButton.textContent = '移除中...';
        }
        try {
            const csrfToken = getCookie('csrftoken');
            if (!csrfToken) {
                throw new Error("无法获取 CSRF token，请刷新页面重试。");
            }
            // DELETE 请求到具体的 favorite 实例 URL
            const response = await fetch(`/dashboard/api/favorites/${favoriteId}/`, {
                method: 'DELETE',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRFToken': csrfToken
                }
            });
            if (response.ok || response.status === 204) { // 204 No Content 也是成功
                console.log('Favorite remove request successful for:', stockCode);
                showNotification(`股票 ${stockCode} 移除请求已发送`, 'success');
                // --- 不再手动移除行，等待 WebSocket 推送 'favorite_removed' ---
            } else {
                let errorMsg = `移除股票 ${stockCode} 失败`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.detail || Object.values(errorData).flat().join(' ') || errorMsg;
                } catch (e) { /* 忽略解析错误 */ }
                throw new Error(errorMsg);
            }
        } catch (error) {
            console.error('Error removing favorite:', error);
            showNotification(error.message || `移除股票 ${stockCode} 时出错`, 'error');
            if (removeButton) { // 恢复按钮状态
                removeButton.disabled = false;
                removeButton.textContent = '移除';
            }
        }
    }
    // 辅助函数：根据 stockCode 查找 Favorite ID (需要调用 API)
    async function findFavoriteIdByCode(stockCode) {
        try {
            const response = await fetch('/dashboard/api/favorites/', {
                headers: { 'X-Requested-With': 'XMLHttpRequest' }
            });
            if (!response.ok) return null;
            const favorites = await response.json();
            const found = favorites.find(fav => fav.stock && fav.stock.stock_code === stockCode);
            return found ? found.id : null;
        } catch (error) {
            console.error("Error finding favorite ID:", error);
            return null;
        }
    }
    // --- 初始化 (使用模板传递的数据) ---
    function initializeDashboard() {
        if (typeof initialFavoritesData !== 'undefined') {
            renderFavoritesTable(initialFavoritesData); // 使用初始数据渲染
        } else {
            console.warn("Initial favorites data not found in template. Fetching via API.");
            fetchInitialFavorites(); // 如果模板没提供数据，则通过 API 获取
        }
        connectWebSocket();
    }
    // 保留 fetchInitialFavorites 作为备用或手动刷新
    async function fetchInitialFavorites() {
        if (!favoritesTbody) return;
        favoritesLoading.style.display = 'block';
        favoritesEmpty.style.display = 'none';
        favoritesTbody.innerHTML = '';
        try {
            const response = await fetch('/dashboard/api/favorites/', { // GET 请求获取列表
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'Accept': 'application/json',
                }
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const favorites = await response.json();
            // 格式化数据以匹配 renderFavoritesTable 的期望
            const formattedData = favorites.map(fav => ({
                id: fav.id, // <--- 确保 API 返回 ID
                code: fav.stock.stock_code,
                name: fav.stock.stock_name,
                current_price: null, // 初始设为 null
                high_price: null,
                low_price: null,
                open_price: null,
                prev_close_price: null,
                trade_time: null,
                volume: null,
                change_percent: null,
                signal: null,
            }));
            renderFavoritesTable(formattedData);
        } catch (error) {
            console.error('Failed to fetch initial favorites:', error);
            favoritesLoading.style.display = 'none';
            favoritesTbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color: red;">加载自选股列表失败</td></tr>';
        }
    }
    // --- 辅助函数 ---
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    function formatNumber(value, decimals = 2) {
        if (value === null || value === undefined || isNaN(value)) return '--';
        return parseFloat(value).toFixed(decimals);
    }
    function formatPercent(value) {
        if (value === null || value === undefined || isNaN(value)) return '--';
        const percent = parseFloat(value);
        const sign = percent > 0 ? '+' : '';
        return `${sign}${percent.toFixed(2)}%`;
    }
    function formatVolume(value) {
        if (value === null || value === undefined || isNaN(value)) return '--';
        return Number(value).toLocaleString(); // 强制转为数字
    }
    function flashRow(rowElement) {
        rowElement.classList.add('flash');
        setTimeout(() => {
            rowElement.classList.remove('flash');
        }, 500); // 闪烁 500ms
    }
    // 简单的通知函数 (可以替换为更美观的库)
    function showNotification(message, type = 'info') {
        console.log(`[${type.toUpperCase()}] ${message}`);
        // 这里可以添加代码将通知显示在页面上
        const notificationArea = document.getElementById('notification-area'); // 假设页面有这个区域
        if (notificationArea) {
            const div = document.createElement('div');
            div.className = `notification notification-${type}`;
            div.textContent = message;
            notificationArea.appendChild(div);
            setTimeout(() => div.remove(), 3000); // 3秒后自动消失
        }
    }
    // --- 启动 ---
    initializeDashboard();

});
