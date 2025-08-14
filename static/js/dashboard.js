// dashboard.js

document.addEventListener('DOMContentLoaded', function () {
    // =========================================================================
    // === 辅助函数 (全局可复用) ==============================================
    // =========================================================================
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

    function showNotification(message, type = 'info', duration = 3000) {
        let notificationArea = document.getElementById('notification-area');
        // 如果页面上没有通知区域，动态创建一个
        if (!notificationArea) {
            notificationArea = document.createElement('div');
            notificationArea.id = 'notification-area';
            document.body.appendChild(notificationArea);
        }
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notificationArea.appendChild(notification);
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 500);
        }, duration);
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
        return Number(value).toLocaleString();
    }

    // 给表格行添加闪烁效果
    function flashRow(rowElement, type = 'update') {
        const animationClass = type === 'add' ? 'flash-add' : 'flash-update';
        rowElement.classList.add(animationClass);
        setTimeout(() => {
            rowElement.classList.remove(animationClass);
        }, 600);
    }

    // =========================================================================
    // === 主控台页面 (home.html) 功能 =========================================
    // =========================================================================
    function initializeHomePage() {
        const addFavoriteForm = document.getElementById('add-favorite-form');
        const favoritesTbody = document.getElementById('favorites-tbody');

        // 卫兵子句：如果连最基本的自选列表tbody都找不到，直接退出，不执行任何主控台逻辑
        if (!favoritesTbody) {
            return;
        }

        console.log('正在初始化【主控台】页面功能...');

        const favoritesEmpty = document.getElementById('favorites-empty');

        // 检查自选股列表是否为空
        if (favoritesEmpty) {
            favoritesEmpty.style.display = favoritesTbody.children.length === 0 ? 'block' : 'none';
        }

        // WebSocket 连接
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsPath = `${wsProtocol}//${window.location.host}/ws/dashboard/`;
        let socket;

        function connectWebSocket() {
            console.log('正在尝试连接 WebSocket...');
            socket = new WebSocket(wsPath);

            socket.onopen = function (e) {
                console.log('WebSocket 连接已建立');
            };

            socket.onmessage = function (e) {
                const data = JSON.parse(e.data);
                console.log('接收到WebSocket数据:', data);

                switch (data.type) {
                    case 'stock_tick_update':
                        updateStockRow(data.payload);
                        break;
                    case 'favorites_update':
                        renderFavoritesTable(data.payload);
                        break;
                    case 'favorite_added_with_data':
                        addStockRow(data.payload);
                        break;
                    case 'favorite_removed':
                        removeStockRow(data.payload.id);
                        break;
                    default:
                        console.warn('未知的消息类型:', data.type);
                }
            };

            socket.onclose = function (e) {
                console.error('WebSocket 连接意外关闭。正在尝试重新连接...', e.reason);
                setTimeout(connectWebSocket, 5000);
            };

            socket.onerror = function (err) {
                console.error('WebSocket 错误:', err);
            };
        }

        function addStockRow(favData) {
            const existRow = favoritesTbody.querySelector(`tr[data-id="${favData.id}"]`);
            if (existRow) {
                updateStockRow(favData);
                return;
            }
            if (favoritesEmpty) favoritesEmpty.style.display = 'none';

            const row = document.createElement('tr');
            row.dataset.stockCode = favData.code;
            row.dataset.id = favData.id;
            row.dataset.stockName = favData.name;

            row.innerHTML = `
                <td class="stock-code">${favData.code || 'N/A'}</td>
                <td class="stock-name">${favData.name || 'N/A'}</td>
                <td class="price" data-field="current_price">--</td>
                <td class="change-percent" data-field="change_percent">--</td>
                <td class="volume" data-field="volume">--</td>
                <td class="strategy-signal" data-field="signal">--</td>
                <td class="actions">
                    <button class="btn btn-danger btn-sm action-btn remove-favorite-btn" data-action="remove" data-id="${favData.id}">移除</button>
                </td>
            `;
            favoritesTbody.appendChild(row);
            flashRow(row, 'add');
            updateStockRow(favData);
        }

        function removeStockRow(favoriteId) {
            const rowToRemove = favoritesTbody.querySelector(`tr[data-id="${favoriteId}"]`);
            if (rowToRemove) {
                rowToRemove.classList.add('flash-remove');
                setTimeout(() => {
                    rowToRemove.remove();
                    if (favoritesTbody.children.length === 0 && favoritesEmpty) {
                        favoritesEmpty.style.display = 'block';
                    }
                }, 300);
            }
        }

        function renderFavoritesTable(favoritesData) {
            favoritesTbody.innerHTML = '';
            if (!favoritesData || favoritesData.length === 0) {
                if (favoritesEmpty) favoritesEmpty.style.display = 'block';
                return;
            }
            if (favoritesEmpty) favoritesEmpty.style.display = 'none';
            favoritesData.forEach(fav => addStockRow(fav));
        }

        function updateStockRow(updateData) {
            const row = updateData.id
                ? favoritesTbody.querySelector(`tr[data-id="${updateData.id}"]`)
                : favoritesTbody.querySelector(`tr[data-stock-code="${updateData.code}"]`);
            if (!row) return;

            const priceCell = row.querySelector('[data-field="current_price"]');
            const percentCell = row.querySelector('[data-field="change_percent"]');
            const volumeCell = row.querySelector('[data-field="volume"]');

            if (priceCell && updateData.current_price !== undefined) priceCell.textContent = formatNumber(updateData.current_price, 2);
            if (volumeCell && updateData.volume !== undefined) volumeCell.textContent = formatVolume(updateData.volume);

            if (percentCell && updateData.change_percent !== undefined) {
                const changePercent = updateData.change_percent;
                percentCell.textContent = formatPercent(changePercent);
                percentCell.className = 'change-percent';
                if (changePercent > 0) percentCell.classList.add('positive');
                else if (changePercent < 0) percentCell.classList.add('negative');
            }
            flashRow(row, 'update');
        }

        favoritesTbody.addEventListener('click', async function (event) {
            const removeButton = event.target.closest('button[data-action="remove"]');
            if (!removeButton) return;

            const row = removeButton.closest('tr');
            const stockCode = row.dataset.stockCode;
            const stockName = row.dataset.stockName;
            const favoriteId = row.dataset.id;

            if (favoriteId && confirm(`确定要从自选中移除 ${stockCode} - ${stockName} 吗？`)) {
                removeButton.disabled = true;
                removeButton.textContent = '移除中...';
                try {
                    const csrfToken = getCookie('csrftoken');
                    const response = await fetch(`/dashboard/api/favorites/${favoriteId}/`, {
                        method: 'DELETE',
                        headers: { 'X-Requested-With': 'XMLHttpRequest', 'X-CSRFToken': csrfToken }
                    });
                    if (response.ok || response.status === 204) {
                        showNotification(`股票 ${stockCode} 已移除`, 'success');
                        removeStockRow(favoriteId); // 即时反馈
                    } else {
                        throw new Error('移除失败');
                    }
                } catch (error) {
                    showNotification(`移除股票 ${stockCode} 时出错`, 'error');
                    removeButton.disabled = false;
                    removeButton.textContent = '移除';
                }
            }
        });

        const searchInput = document.getElementById('stock-search-input');
        if (searchInput) {
            const searchResultsContainer = document.getElementById('search-results');
            let debounceTimer;
            let selectedStockCode = null;

            searchInput.addEventListener('keyup', (event) => {
                const query = searchInput.value.trim();
                clearTimeout(debounceTimer);

                if (query.length === 0) {
                    searchResultsContainer.innerHTML = '';
                    searchResultsContainer.style.display = 'none';
                    selectedStockCode = null;
                    return;
                }

                debounceTimer = setTimeout(async () => {
                    console.log(`[Search] 正在搜索: ${query}`);
                    try {
                        const response = await fetch(`/dashboard/api/search/?q=${encodeURIComponent(query)}`);
                        if (!response.ok) {
                            throw new Error('网络响应错误');
                        }
                        const stocks = await response.json();
                        renderSearchResults(stocks);
                    } catch (error) {
                        console.error('[Search] 搜索API请求失败:', error);
                        searchResultsContainer.innerHTML = '<div class="search-result-item">搜索出错，请稍后重试。</div>';
                        searchResultsContainer.style.display = 'block';
                    }
                }, 300);
            });

            function renderSearchResults(stocks) {
                searchResultsContainer.innerHTML = '';
                if (stocks.length === 0) {
                    searchResultsContainer.innerHTML = '<div class="search-result-item">未找到相关股票</div>';
                } else {
                    stocks.forEach(stock => {
                        const item = document.createElement('div');
                        item.className = 'search-result-item';
                        item.textContent = `${stock.stock_code} - ${stock.stock_name}`;
                        item.dataset.stockCode = stock.stock_code;
                        item.dataset.stockName = stock.stock_name;
                        searchResultsContainer.appendChild(item);
                    });
                }
                searchResultsContainer.style.display = 'block';
            }

            searchResultsContainer.addEventListener('click', (event) => {
                const targetItem = event.target.closest('.search-result-item');
                if (targetItem && targetItem.dataset.stockCode) {
                    console.log(`[Search] 选中了: ${targetItem.dataset.stockCode}`);
                    searchInput.value = `${targetItem.dataset.stockCode} - ${targetItem.dataset.stockName}`;
                    selectedStockCode = targetItem.dataset.stockCode;
                    searchResultsContainer.innerHTML = '';
                    searchResultsContainer.style.display = 'none';
                }
            });

            if (addFavoriteForm) {
                addFavoriteForm.addEventListener('submit', async (event) => {
                    event.preventDefault();

                    if (!selectedStockCode) {
                        showNotification('请先从搜索结果中选择一只股票', 'warning');
                        return;
                    }

                    const addButton = document.getElementById('add-favorite-btn');
                    addButton.disabled = true;
                    addButton.innerHTML = '<span class="icon">+</span> 添加中...';

                    try {
                        const csrfToken = getCookie('csrftoken');
                        const response = await fetch('/dashboard/api/favorites/', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-Requested-With': 'XMLHttpRequest',
                                'X-CSRFToken': csrfToken
                            },
                            body: JSON.stringify({ stock_code: selectedStockCode })
                        });

                        // 【代码修改】API返回的是一个包含 detail 信息的对象，而不是序列化的 Favorite 对象
                        const responseData = await response.json();

                        if (response.ok) {
                            // 【代码修改】直接使用后端返回的友好提示信息
                            showNotification(responseData.detail || `股票 ${selectedStockCode} 操作成功！`, 'success');
                            searchInput.value = '';
                            selectedStockCode = null;
                            // 刷新页面以看到新添加的自选股（简单有效的方案）
                            setTimeout(() => window.location.reload(), 1000);
                        } else {
                            const errorMsg = responseData.detail || (responseData.stock_code ? `代码: ${responseData.stock_code[0]}` : '添加失败，请检查该股票是否已在自选列表中');
                            showNotification(errorMsg, 'error');
                        }
                    } catch (error) {
                        console.error('[Favorite Add] 添加自选失败:', error);
                        showNotification('网络错误，添加失败', 'error');
                    } finally {
                        addButton.disabled = false;
                        addButton.innerHTML = '<span class="icon">+</span> 添加到自选';
                    }
                });
            }

            document.addEventListener('click', (event) => {
                if (addFavoriteForm && !addFavoriteForm.contains(event.target)) {
                    searchResultsContainer.style.display = 'none';
                }
            });
        }

        connectWebSocket();
    }

    // =========================================================================
    // === 策略监控中心 (trend_following_list.html) 功能 =======================
    // =========================================================================
    function initializeTrendListPage() {
        const tableBody = document.getElementById('trend-table-body');
        if (!tableBody) return; // 卫兵子句

        console.log('正在初始化【策略监控中心】页面功能...');

        // --- 折叠功能 ---
        tableBody.addEventListener('click', function (event) {
            const toggleBtn = event.target.closest('.toggle-playbooks');
            if (toggleBtn) {
                event.preventDefault();
                const list = toggleBtn.closest('.playbook-container').querySelector('.playbook-list');
                if (!list) return;
                list.classList.toggle('expanded');
                const isExpanded = list.classList.contains('expanded');
                toggleBtn.textContent = isExpanded ? toggleBtn.dataset.textCollapse : toggleBtn.dataset.textExpand;
            }
        });

        // --- 添加自选功能 ---
        const favoriteStockCodes = new Set();

        function updateButtonState(button, isFavorite, isLoading = false) {
            const icon = button.querySelector('.btn-icon');
            const text = button.querySelector('.btn-text');
            button.disabled = isLoading || isFavorite;
            button.classList.remove('is-favorite', 'is-loading');

            if (isLoading) {
                button.classList.add('is-loading');
                if (icon) icon.textContent = '...';
                if (text) text.textContent = '处理中';
            } else if (isFavorite) {
                button.classList.add('is-favorite');
                if (icon) icon.textContent = '✓';
                if (text) text.textContent = '已添加';
            } else {
                if (icon) icon.textContent = '+';
                if (text) text.textContent = '添加自选';
            }
        }

        async function initializeFavoriteButtons() {
            try {
                const response = await fetch('/dashboard/api/favorites/', {
                    headers: { 'X-Requested-With': 'XMLHttpRequest', 'Accept': 'application/json' }
                });
                if (!response.ok) throw new Error('获取自选股列表失败');

                const favorites = await response.json();
                favorites.forEach(fav => favoriteStockCodes.add(fav.stock.stock_code));

                const allButtons = tableBody.querySelectorAll('.add-to-favorites-btn');
                allButtons.forEach(button => {
                    const stockCode = button.dataset.stockCode;
                    if (favoriteStockCodes.has(stockCode)) {
                        updateButtonState(button, true);
                    }
                });
            } catch (error) {
                console.error('初始化自选按钮失败:', error);
                showNotification('无法加载自选状态，请刷新重试', 'error');
            }
        }

        async function handleAddFavorite(event) {
            const button = event.target.closest('.add-to-favorites-btn');
            if (!button || button.disabled) return;

            const stockCode = button.dataset.stockCode;

            // 1. 防御性检查：现在只检查 stockCode
            if (!stockCode) {
                showNotification('无法获取股票代码，操作已取消', 'error');
                console.error('错误：点击了添加自选按钮，但未能从 data-stock-code 属性中获取到值。');
                return;
            }

            // 2. 更新按钮为加载状态 
            updateButtonState(button, false, true);

            try {
                const csrfToken = getCookie('csrftoken');
                const response = await fetch('/dashboard/api/favorites/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest',
                        'X-CSRFToken': csrfToken
                    },
                    // 3. 【核心修改】请求体中只包含 stock_code
                    body: JSON.stringify({
                        stock_code: stockCode,
                    })
                });

                const responseData = await response.json(); // 无论成功失败，都先解析响应体

                if (response.ok) {
                    // 4. 使用后端返回的友好提示信息
                    showNotification(responseData.detail || `股票 ${stockCode} 操作成功！`, 'success');
                    favoriteStockCodes.add(stockCode);
                    updateButtonState(button, true);
                } else {
                    // 5. 同样使用后端返回的错误信息
                    const errorMsg = responseData.detail || `添加 ${stockCode} 失败`;
                    showNotification(errorMsg, 'error');
                    // 如果错误是因为已存在，也应该将按钮更新为“已添加”状态
                    if (response.status === 400 && errorMsg.includes('已存在')) {
                        updateButtonState(button, true);
                    } else {
                        updateButtonState(button, false);
                    }
                }
            } catch (error) {
                showNotification('网络错误，请稍后重试', 'error');
                updateButtonState(button, false);
            }
        }

        // 使用事件委托来处理所有“添加自选”按钮的点击事件
        tableBody.addEventListener('click', handleAddFavorite);

        // 初始化
        initializeFavoriteButtons();
    }

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

    // =========================================================================
    // === 页面路由和启动 ======================================================
    // =========================================================================
    console.log('[JS] DOMContentLoaded 事件触发，开始页面路由判断...');
    if (document.getElementById('favorites-tbody')) {
        console.log('[JS] 检测到 "favorites-tbody"，判定为【主控台】页面。');
        initializeHomePage();
    } else if (document.getElementById('trend-table-body')) {
        console.log('[JS] 检测到 "trend-table-body"，判定为【策略监控中心】页面。');
        initializeTrendListPage();
    } else if (document.getElementById('fav-trend-table-body')) {
        console.log('[JS] 检测到 "fav-trend-table-body"，判定为【自选股监控】页面。');
        initializeFavTrendListPage();
    } else if (document.getElementById('signal-stream')) {
        console.log('[JS] 检测到 "signal-stream"，判定为【盘中引擎实时监控】页面。');
        initializeRealtimeEnginePage();
    } else {
        console.warn('[JS] 未匹配到任何已知页面的主要元素ID，没有执行任何页面专属的初始化函数。');
    }

});
