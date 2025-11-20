// static\js\home.js

document.addEventListener('DOMContentLoaded', function () {
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
            // 假设 flashRow 是全局可用的
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
            // 假设 formatNumber, formatVolume, formatPercent, flashRow 是全局可用的
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
            // 假设 getCookie 和 showNotification 是全局可用的
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
            const viewDetailBtn = document.getElementById('view-detail-btn'); // 新增行: 获取“查看详情”按钮
            let debounceTimer;
            let selectedStockCode = null;
            searchInput.addEventListener('keyup', (event) => {
                const query = searchInput.value.trim();
                clearTimeout(debounceTimer);
                if (query.length === 0) {
                    searchResultsContainer.innerHTML = '';
                    searchResultsContainer.style.display = 'none';
                    selectedStockCode = null;
                    // 新增开始: 当输入框清空时，禁用“查看详情”按钮
                    if (viewDetailBtn) {
                        viewDetailBtn.classList.add('disabled');
                        viewDetailBtn.href = '#';
                    }
                    // 新增结束
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
                    // 新增开始: 当选中股票时，激活“查看详情”按钮并设置链接
                    if (viewDetailBtn) {
                        viewDetailBtn.href = `/dashboard/stock/${selectedStockCode}/`;
                        viewDetailBtn.classList.remove('disabled');
                    }
                    // 新增结束
                }
            });
            if (addFavoriteForm) {
                addFavoriteForm.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    if (!selectedStockCode) {
                        // 假设 showNotification 是全局可用的
                        showNotification('请先从搜索结果中选择一只股票', 'warning');
                        return;
                    }
                    const addButton = document.getElementById('add-favorite-btn');
                    addButton.disabled = true;
                    addButton.innerHTML = '<span class="icon">+</span> 添加中...';
                    try {
                        // 假设 getCookie 是全局可用的
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
                        const responseData = await response.json();
                        if (response.ok) {
                            // 假设 showNotification 是全局可用的
                            showNotification(responseData.detail || `股票 ${selectedStockCode} 操作成功！`, 'success');
                            searchInput.value = '';
                            selectedStockCode = null;
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
    // 在这里调用上面定义的初始化函数，以确保它能够执行
    initializeHomePage();
});
