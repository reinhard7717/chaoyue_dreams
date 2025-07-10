// dashboard.js

document.addEventListener('DOMContentLoaded', function() {

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
        console.log('正在初始化【主控台】页面功能...');

        const favoritesTbody = document.getElementById('favorites-tbody');
        const favoritesEmpty = document.getElementById('favorites-empty');
        
        // 检查自选股列表是否为空
        if (favoritesTbody && favoritesEmpty) {
            favoritesEmpty.style.display = favoritesTbody.children.length === 0 ? 'block' : 'none';
        }

        // WebSocket 连接
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsPath = `${wsProtocol}//${window.location.host}/ws/dashboard/`;
        let socket;

        function connectWebSocket() {
            console.log('正在尝试连接 WebSocket...');
            socket = new WebSocket(wsPath);

            socket.onopen = function(e) {
                console.log('WebSocket 连接已建立');
            };

            socket.onmessage = function(e) {
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

            socket.onclose = function(e) {
                console.error('WebSocket 连接意外关闭。正在尝试重新连接...', e.reason);
                setTimeout(connectWebSocket, 5000);
            };

            socket.onerror = function(err) {
                console.error('WebSocket 错误:', err);
            };
        }

        function addStockRow(favData) {
            if (!favoritesTbody) return;
            const existRow = favoritesTbody.querySelector(`tr[data-id="${favData.id}"]`);
            if (existRow) {
                updateStockRow(favData);
                return;
            }
            if(favoritesEmpty) favoritesEmpty.style.display = 'none';
        
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
            if (!favoritesTbody) return;
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
            if (!favoritesTbody) return;
            favoritesTbody.innerHTML = '';
            if (!favoritesData || favoritesData.length === 0) {
                if(favoritesEmpty) favoritesEmpty.style.display = 'block';
                return;
            }
            if(favoritesEmpty) favoritesEmpty.style.display = 'none';
            favoritesData.forEach(fav => addStockRow(fav));
        }

        function updateStockRow(updateData) {
            if (!favoritesTbody) return;
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

        if (favoritesTbody) {
            favoritesTbody.addEventListener('click', async function(event) {
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
        }
        
        // 启动WebSocket
        connectWebSocket();
    }

    // =========================================================================
    // === 策略监控中心 (trend_following_list.html) 功能 =======================
    // =========================================================================
    function initializeTrendListPage() {
        console.log('正在初始化【策略监控中心】页面功能...');

        const tableBody = document.getElementById('trend-table-body');
        if (!tableBody) return;

        // --- 折叠功能 ---
        tableBody.addEventListener('click', function(event) {
            const toggleBtn = event.target.closest('.toggle-playbooks');
            if (!toggleBtn) return;
            event.preventDefault(); 
            const list = toggleBtn.closest('.playbook-container').querySelector('.playbook-list');
            if (!list) return;
            list.classList.toggle('expanded');
            const isExpanded = list.classList.contains('expanded');
            toggleBtn.textContent = isExpanded ? toggleBtn.dataset.textCollapse : toggleBtn.dataset.textExpand;
        });

        // --- 添加自选功能 ---
        const favoriteStockCodes = new Set();

        // 更新按钮状态的辅助函数
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

        // 页面加载时，获取自选股列表并初始化按钮状态
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

        // 处理点击“添加自选”按钮的事件
        async function handleAddFavorite(event) {
            const button = event.target.closest('.add-to-favorites-btn');
            if (!button || button.disabled) return;

            const stockCode = button.dataset.stockCode;
            updateButtonState(button, false, true); // 设置为加载中状态

            try {
                const csrfToken = getCookie('csrftoken');
                const response = await fetch('/dashboard/api/favorites/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest',
                        'X-CSRFToken': csrfToken
                    },
                    body: JSON.stringify({ stock_code: stockCode })
                });

                if (response.ok) {
                    showNotification(`股票 ${stockCode} 添加成功！`, 'success');
                    favoriteStockCodes.add(stockCode);
                    updateButtonState(button, true); // 设置为已添加状态
                } else {
                    const errorData = await response.json();
                    const errorMsg = errorData.detail || Object.values(errorData).flat().join(' ') || `添加 ${stockCode} 失败`;
                    showNotification(errorMsg, 'error');
                    updateButtonState(button, false); // 恢复为初始状态
                }
            } catch (error) {
                showNotification('网络错误，请稍后重试', 'error');
                updateButtonState(button, false); // 恢复为初始状态
            }
        }

        // 使用事件委托来处理所有按钮的点击事件
        tableBody.addEventListener('click', handleAddFavorite);
        
        // 初始化
        initializeFavoriteButtons();
    }

    // =========================================================================
    // === 自选股监控 (fav_trend_following_list.html) 功能 ====================
    // =========================================================================
    function initializeFavTrendListPage() {
        // ▼▼▼【代码修改】: 增加调试日志 ▼▼▼
        console.log('--- [JS] 正在初始化【自选股监控】页面功能 (initializeFavTrendListPage) ---');
        // ▲▲▲【代码修改结束】▲▲▲

        const tableBody = document.getElementById('fav-trend-table-body');
        if (!tableBody) {
            console.error('[JS] 错误：在自选股监控页面未找到 ID 为 "fav-trend-table-body" 的元素。');
            return;
        }

        // --- 整合原有的折叠功能 和 新增移除自选股功能 ---
        tableBody.addEventListener('click', async function(event) {
            const toggleBtn = event.target.closest('.toggle-playbooks');
            const removeButton = event.target.closest('button[data-action="remove"]');

            // 处理折叠按钮
            if (toggleBtn) {
                console.log('[JS] 折叠/展开按钮被点击。');
                event.preventDefault(); 
                const list = toggleBtn.closest('.playbook-container').querySelector('.playbook-list');
                if (!list) return;
                list.classList.toggle('expanded');
                const isExpanded = list.classList.contains('expanded');
                toggleBtn.textContent = isExpanded ? toggleBtn.dataset.textCollapse : toggleBtn.dataset.textExpand;
                return; // 处理完折叠后，不再继续执行
            }

            // 处理移除按钮
            if (removeButton) {
                // ▼▼▼【代码修改】: 增加调试日志 ▼▼▼
                console.log('[JS] 移除按钮被点击。');
                // ▲▲▲【代码修改结束】▲▲▲
                const favoriteId = removeButton.dataset.id;
                const stockCode = removeButton.dataset.stockCode;
                const stockName = removeButton.dataset.stockName;

                if (favoriteId && confirm(`确定要从自选中移除 ${stockCode} - ${stockName} 吗？`)) {
                    removeButton.disabled = true;
                    removeButton.innerHTML = '...';

                    try {
                        const csrfToken = getCookie('csrftoken');
                        if (!csrfToken) throw new Error('无法获取CSRF令牌');
                        
                        console.log(`[JS] 正在发送 DELETE 请求到 /dashboard/api/favorites/${favoriteId}/`);
                        const response = await fetch(`/dashboard/api/favorites/${favoriteId}/`, {
                            method: 'DELETE',
                            headers: {
                                'X-Requested-With': 'XMLHttpRequest',
                                'X-CSRFToken': csrfToken
                            }
                        });

                        if (response.ok || response.status === 204) {
                            console.log('[JS] 后端成功响应，正在从界面移除该行。');
                            showNotification(`股票 ${stockCode} 已成功移除`, 'success');
                            const rowToRemove = removeButton.closest('tr');
                            if (rowToRemove) {
                                rowToRemove.classList.add('flash-remove');
                                setTimeout(() => {
                                    rowToRemove.remove();
                                    if (tableBody.children.length === 0) {
                                        const colspan = tableBody.previousElementSibling.rows[0].cells.length;
                                        tableBody.innerHTML = `<tr><td colspan="${colspan}" style="text-align: center; padding: 20px;">自选股列表已清空。</td></tr>`;
                                    }
                                }, 300);
                            }
                        } else {
                            const errorData = await response.json().catch(() => ({}));
                            const errorMsg = errorData.detail || `移除股票 ${stockCode} 失败 (状态码: ${response.status})`;
                            throw new Error(errorMsg);
                        }
                    } catch (error) {
                        console.error('[JS] 移除自选股时发生错误:', error);
                        showNotification(error.message, 'error');
                        removeButton.disabled = false;
                        removeButton.innerHTML = '&times;';
                    }
                }
                return; // 处理完移除后，不再继续执行
            }
        });
    }

    // =========================================================================
    // === 页面路由和启动 ======================================================
    // =========================================================================
    // ▼▼▼【代码修改】: 将路由逻辑从多个if改为if/else if结构，并增加调试日志 ▼▼▼
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
    } else {
        console.warn('[JS] 未匹配到任何已知页面的主要元素ID，没有执行任何页面专属的初始化函数。');
    }

});