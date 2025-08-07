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

        // ▼▼▼【代码修改】: 增加保护性检查，确保只在搜索框存在时才执行相关逻辑 ▼▼▼
        const searchInput = document.getElementById('stock-search-input');
        // 只有当 searchInput 元素存在时，才初始化所有搜索和添加相关的逻辑
        if (searchInput) {
            const searchResultsContainer = document.getElementById('search-results');
            let debounceTimer;
            let selectedStockCode = null;

            // 1. 监听搜索框的输入事件
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

            // 2. 渲染搜索结果
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

            // 3. 监听搜索结果容器的点击事件（事件委托）
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

            // 4. 监听表单提交事件 (确保 addFavoriteForm 存在)
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

                        if (response.ok) {
                            const newFavorite = await response.json();
                            showNotification(`股票 ${newFavorite.stock.stock_code} 添加成功！`, 'success');
                            searchInput.value = '';
                            selectedStockCode = null;
                        } else {
                            const errorData = await response.json();
                            const errorMsg = errorData.detail || (errorData.stock_code ? `代码: ${errorData.stock_code[0]}` : '添加失败，请检查该股票是否已在自选列表中');
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

            // 点击页面其他地方，隐藏搜索结果
            document.addEventListener('click', (event) => {
                if (addFavoriteForm && !addFavoriteForm.contains(event.target)) {
                    searchResultsContainer.style.display = 'none';
                }
            });
        }
        // ▲▲▲【代码修改结束】▲▲▲

        // 启动WebSocket
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

            // --- 代码修改开始 ---
            // [修改原因] 修复报错，确保JS读取的属性名 data-signal-id 与HTML中定义的一致
            const signalId = button.dataset.signalId; // 从 logId 改为 signalId

            // 防御性检查
            if (!signalId) {
                // 这里的报错信息也同步更新
                showNotification('无法获取信号ID，操作已取消', 'error');
                console.error('错误：点击了添加自选按钮，但未能从 data-signal-id 属性中获取到值。');
                return;
            }
            // --- 代码修改结束 ---

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
                    body: JSON.stringify({
                        stock_code: stockCode,
                        signal_id: signalId // API需要的是 signal_id
                    })
                });

                if (response.ok) {
                    showNotification(`股票 ${stockCode} 添加成功！`, 'success');
                    favoriteStockCodes.add(stockCode);
                    updateButtonState(button, true);
                } else {
                    const errorData = await response.json();
                    const errorMsg = errorData.detail || Object.values(errorData).flat().join(' ') || `添加 ${stockCode} 失败`;
                    showNotification(errorMsg, 'error');
                    updateButtonState(button, false);
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
    // === 自选股监控 (fav_trend_following_list.html) 功能 ====================
    // =========================================================================
    function initializeFavTrendListPage() {
        // 调试点 1: 确认函数是否被调用
        console.log('[调试点 1] initializeFavTrendListPage 函数已开始执行。');

        const tableBody = document.getElementById('fav-trend-table-body');
        if (!tableBody) {
            console.error('[错误] 页面中未找到 ID 为 "fav-trend-table-body" 的元素，初始化失败！');
            return;
        }
        console.log('[调试点 2] 已成功获取到 tableBody 元素:', tableBody);

        const modalOverlay = document.getElementById('transaction-modal-overlay');
        const modalTitle = document.getElementById('transaction-modal-title');
        const modalCloseBtn = document.getElementById('transaction-modal-close-btn');
        const transactionListTbody = document.getElementById('transaction-list-tbody');
        const transactionListLoading = document.getElementById('transaction-list-loading');
        const addTransactionForm = document.getElementById('add-transaction-form');
        const formTrackerIdInput = document.getElementById('form-tracker-id');

        tableBody.addEventListener('click', async function (event) {
            console.log('[调试点 3] tableBody 内发生点击事件。被点击的原始元素是:', event.target);
            const manageButton = event.target.closest('.manage-transactions-btn');
            const removeButton = event.target.closest('.remove-position-btn');
            console.log('[调试点 4] closest() 查找结果:', { manageButton, removeButton });

            if (manageButton) {
                console.log('[分支 1] 检测到“管理”按钮被点击。');
                event.preventDefault();
                const trackerId = manageButton.dataset.trackerId;
                const stockName = manageButton.dataset.stockName;
                console.log('[调试点 5] 从“管理”按钮获取的数据:', { trackerId, stockName });
                if (!trackerId) {
                    showNotification('操作失败：缺少 tracker ID。', 'error');
                    return;
                }
                openTransactionModal(trackerId, stockName);
                return;
            }

            if (removeButton) {
                console.log('[分支 2] 检测到“删除”按钮被点击。');
                event.preventDefault();
                const favoriteId = removeButton.dataset.favId;
                const stockCode = removeButton.dataset.stockCode;
                console.log('[调试点 6] 从“删除”按钮获取的数据:', { favoriteId, stockCode });
                if (!favoriteId) {
                    showNotification('操作失败：缺少 favorite ID (data-fav-id)。', 'error');
                    return;
                }
                if (confirm(`确定要从自选列表中移除 ${stockCode} 吗？\n注意：这只会移除自选星标，不会删除您的交易记录。`)) {
                    removeButton.disabled = true;
                    removeButton.textContent = '...';
                    try {
                        const csrfToken = getCookie('csrftoken');
                        const response = await fetch(`/dashboard/api/favorites/${favoriteId}/`, {
                            method: 'DELETE',
                            headers: { 'X-Requested-With': 'XMLHttpRequest', 'X-CSRFToken': csrfToken }
                        });
                        if (response.ok || response.status === 204) {
                            showNotification(`股票 ${stockCode} 已成功从自选中移除`, 'success');
                            const rowToRemove = removeButton.closest('tr');
                            if (rowToRemove) {
                                rowToRemove.classList.add('flash-remove');
                                setTimeout(() => rowToRemove.remove(), 300);
                            }
                        } else {
                            throw new Error('删除失败，请刷新后重试');
                        }
                    } catch (error) {
                        showNotification(error.message, 'error');
                        removeButton.disabled = false;
                        removeButton.textContent = '×';
                    }
                }
                return;
            }
            console.log('[调试信息] 点击未命中任何目标按钮。');
        });

        async function openTransactionModal(trackerId, stockName) {
            console.log(`[Modal调试 1] 进入 openTransactionModal 函数。接收到 trackerId: ${trackerId}, stockName: ${stockName}`);

            // 检查关键的 modalOverlay 元素是否存在
            if (!modalOverlay) {
                console.error('[Modal错误] 无法找到 modalOverlay 元素 (ID: transaction-modal-overlay)！模态框无法显示。');
                showNotification('页面结构错误，无法打开管理窗口。', 'error');
                return;
            }

            modalTitle.textContent = `管理 [${stockName}] 的交易流水`;
            formTrackerIdInput.value = trackerId;

            console.log('[Modal调试 2] 准备显示模态框，设置 style.display = "flex"');
            modalOverlay.style.display = 'flex';
            console.log('[Modal调试 3] 已设置 style.display。当前 modalOverlay 的 display 状态是:', window.getComputedStyle(modalOverlay).display);

            transactionListLoading.style.display = 'block';
            transactionListTbody.innerHTML = '';

            try {
                console.log(`[Modal调试 4] 准备发起 fetch 请求获取交易流水: /dashboard/api/transactions/?tracker_id=${trackerId}`);
                const response = await fetch(`/dashboard/api/transactions/?tracker_id=${trackerId}`);
                console.log('[Modal调试 5] fetch 请求已收到响应:', response);

                if (!response.ok) {
                    throw new Error(`获取交易流水失败 (状态: ${response.status})`);
                }
                const transactions = await response.json();
                console.log('[Modal调试 6] 成功解析交易数据:', transactions);
                renderTransactionList(transactions);
            } catch (error) {
                console.error('[Modal错误] 加载交易流水时出错:', error);
                showNotification(error.message, 'error');
                transactionListTbody.innerHTML = `<tr><td colspan="5">加载失败: ${error.message}</td></tr>`;
            } finally {
                transactionListLoading.style.display = 'none';
                console.log('[Modal调试 7] 函数执行完毕。');
            }
        }

        function renderTransactionList(transactions) {
            transactionListTbody.innerHTML = '';
            if (transactions.length === 0) {
                transactionListTbody.innerHTML = `<tr><td colspan="5">暂无交易记录</td></tr>`;
                return;
            }
            transactions.forEach(tx => {
                const row = document.createElement('tr');
                const txDate = new Date(tx.transaction_date).toISOString().split('T')[0];
                row.innerHTML = `
                    <td>${tx.transaction_type === 'BUY' ? '买入' : '卖出'}</td>
                    <td>${txDate}</td>
                    <td>${formatNumber(tx.price, 2)}</td>
                    <td>${formatVolume(tx.quantity)}</td>
                    <td class="actions"><button class="btn btn-sm btn-danger delete-transaction-btn" data-tx-id="${tx.id}">删除</button></td>
                `;
                transactionListTbody.appendChild(row);
            });
        }

        function closeTransactionModal() { modalOverlay.style.display = 'none'; }
        modalCloseBtn.addEventListener('click', closeTransactionModal);
        modalOverlay.addEventListener('click', (event) => { if (event.target === modalOverlay) closeTransactionModal(); });

        addTransactionForm.addEventListener('submit', async function (event) {
            event.preventDefault();
            const submitBtn = document.getElementById('add-transaction-submit-btn');
            submitBtn.disabled = true;
            submitBtn.textContent = '处理中...';
            const formData = new FormData(addTransactionForm);
            const data = Object.fromEntries(formData.entries());
            data.transaction_date = new Date(data.transaction_date).toISOString();
            try {
                const csrfToken = getCookie('csrftoken');
                const response = await fetch('/dashboard/api/transactions/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
                    body: JSON.stringify(data),
                });
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || '添加失败');
                }
                showNotification('交易添加成功！快照正在后台更新...', 'success');
                addTransactionForm.reset();
                openTransactionModal(data.tracker, modalTitle.textContent.split('[')[1].split(']')[0]);
            } catch (error) {
                showNotification(error.message, 'error');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = '确认添加';
            }
        });

        transactionListTbody.addEventListener('click', async function (event) {
            const deleteBtn = event.target.closest('.delete-transaction-btn');
            if (!deleteBtn) return;
            const txId = deleteBtn.dataset.txId;
            if (!confirm('确定要删除这条交易记录吗？此操作会重新计算持仓成本和历史快照。')) return;
            deleteBtn.disabled = true;
            deleteBtn.textContent = '...';
            try {
                const csrfToken = getCookie('csrftoken');
                const response = await fetch(`/dashboard/api/transactions/${txId}/`, {
                    method: 'DELETE',
                    headers: { 'X-CSRFToken': csrfToken },
                });
                if (!response.ok && response.status !== 204) throw new Error('删除失败');
                showNotification('交易删除成功！快照正在后台更新...', 'success');
                openTransactionModal(formTrackerIdInput.value, modalTitle.textContent.split('[')[1].split(']')[0]);
            } catch (error) {
                showNotification(error.message, 'error');
                deleteBtn.disabled = false;
                deleteBtn.textContent = '删除';
            }
        });
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
