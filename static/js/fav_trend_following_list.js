// static\js\fav_trend_following_list.js

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // 判断这个 cookie 字符串是否以我们想要的名字开头
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
// 将所有代码包裹在 DOMContentLoaded 事件监听器中
// 这确保了在执行JS代码时，页面上的所有HTML元素（如表格、按钮）都已经加载完毕
document.addEventListener('DOMContentLoaded', function () {
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
        // 将所有模态框相关的DOM元素获取操作放在函数顶层，确保它们在整个函数作用域内可用
        const modalOverlay = document.getElementById('transaction-modal-overlay');
        const modalContainer = document.getElementById('transaction-modal-container');
        const modalTitle = document.getElementById('transaction-modal-title');
        const modalCloseBtn = document.getElementById('transaction-modal-close-btn');
        const transactionListTbody = document.getElementById('transaction-list-tbody');
        const transactionListLoading = document.getElementById('transaction-list-loading');
        const addTransactionForm = document.getElementById('add-transaction-form');
        // 使用事件委托处理表格行的所有点击事件
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
        // 辅助函数：给元素添加闪烁效果
        function flashElement(element) {
            if (!element) return;
            element.classList.add('flash-attention');
            setTimeout(() => {
                element.classList.remove('flash-attention');
            }, 700); // 动画持续时间
        }
        // 核心函数：打开并填充交易管理模态框
        async function openTransactionModal(trackerId, stockName) {
            console.log(`[Modal调试 1] 进入 openTransactionModal 函数。接收到 trackerId: ${trackerId}, stockName: ${stockName}`);
            if (!modalOverlay || !modalContainer) {
                console.error('[Modal错误] 无法找到模态框核心元素！模态框无法显示。');
                showNotification('页面结构错误，无法打开管理窗口。', 'error');
                return;
            }
            // 更新UI并存储状态
            modalTitle.textContent = `管理 [${stockName}] 的交易流水`;
            modalContainer.dataset.trackerId = trackerId;
            modalContainer.dataset.stockName = stockName;
            document.getElementById('form-tracker-id').value = trackerId;
            // 显示模态框和加载状态
            modalOverlay.style.display = 'flex';
            transactionListLoading.style.display = 'block';
            transactionListTbody.innerHTML = '';
            // 异步获取交易数据
            try {
                console.log(`[Modal调试 4] 准备发起 fetch 请求获取交易流水: /dashboard/api/transactions/?tracker_id=${trackerId}`);
                const response = await fetch(`/dashboard/api/transactions/?tracker_id=${trackerId}`);
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
        // 辅助函数：渲染交易列表
        function renderTransactionList(transactions) {
            transactionListTbody.innerHTML = ''; // 清空旧内容
            if (transactions.length === 0) {
                const emptyRow = document.createElement('tr');
                const emptyCell = document.createElement('td');
                emptyCell.colSpan = 5;
                emptyCell.style.textAlign = 'center';
                emptyCell.style.padding = '20px';
                emptyCell.innerHTML = '暂无交易记录，请在下方表单中添加您的第一笔买入交易。';
                emptyRow.appendChild(emptyCell);
                transactionListTbody.appendChild(emptyRow);
                const addTransactionWrapper = document.querySelector('.add-transaction-wrapper');
                flashElement(addTransactionWrapper);
            } else {
                transactions.forEach(tx => {
                    const row = document.createElement('tr');
                    const txDate = new Date(tx.transaction_date).toISOString().split('T')[0];
                    row.innerHTML = `
                        <td>${tx.transaction_type === 'BUY' ? '买入' : '卖出'}</td>
                        <td>${txDate}</td>
                        <td>${formatNumber(tx.price, 2)}</td>
                        <td>${formatVolume(tx.quantity)}</td>
                        <td class="actions">
                            <button class="btn btn-sm btn-danger delete-transaction-btn" data-tx-id="${tx.id}">删除</button>
                        </td>
                    `;
                    transactionListTbody.appendChild(row);
                });
            }
        }
        // 辅助函数：关闭模态框
        function closeTransactionModal() {
            if (modalOverlay) {
                modalOverlay.style.display = 'none';
            }
        }
        // 为模态框的关闭按钮和遮罩层添加事件监听
        if (modalCloseBtn) {
            modalCloseBtn.addEventListener('click', closeTransactionModal);
        }
        if (modalOverlay) {
            modalOverlay.addEventListener('click', (event) => { if (event.target === modalOverlay) closeTransactionModal(); });
        }
        // 为新增交易表单添加提交事件监听
        if (addTransactionForm) {
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
                    // --- 修改代码开始 ---
                    // 原来的逻辑是刷新模态框，这会让用户困惑。
                    // 新逻辑是直接关闭模态框，提供清晰的操作完成反馈。
                    // const currentTrackerId = modalContainer.dataset.trackerId;
                    // const currentStockName = modalContainer.dataset.stockName;
                    // if (currentTrackerId && currentStockName) {
                    //     openTransactionModal(currentTrackerId, currentStockName);
                    // }
                    closeTransactionModal(); // 直接调用关闭模态框函数
                    // --- 修改代码结束 ---
                } catch (error) {
                    showNotification(error.message, 'error');
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = '确认添加';
                }
            });
        }
        // 为交易列表（用于删除）添加事件委托
        if (transactionListTbody) {
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
                    // 从 modalContainer 的 dataset 中获取可靠的状态来刷新模态框
                    const currentTrackerId = modalContainer.dataset.trackerId;
                    const currentStockName = modalContainer.dataset.stockName;
                    if (currentTrackerId && currentStockName) {
                        openTransactionModal(currentTrackerId, currentStockName);
                    }
                } catch (error) {
                    showNotification(error.message, 'error');
                    deleteBtn.disabled = false;
                    deleteBtn.textContent = '删除';
                }
            });
        }
        // WebSocket连接，用于接收后台任务完成的通知并刷新页面
        function connectFavListWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsPath = `${wsProtocol}//${window.location.host}/ws/dashboard/`;
            console.log('[FavList WS] 正在尝试连接 WebSocket:', wsPath);
            const socket = new WebSocket(wsPath);
            socket.onopen = function (e) {
                console.log('[FavList WS] WebSocket 连接成功！');
            };
            socket.onmessage = function (e) {
                const data = JSON.parse(e.data);
                console.log('[FavList WS] 收到WebSocket消息:', data);
                if (data.type === 'snapshot_rebuilt') {
                    console.log('[FavList WS] 接收到快照重建完成信号，准备刷新页面...');
                    showNotification('持仓数据已更新，页面即将刷新...', 'info', 1500);
                    setTimeout(function () {
                        window.location.reload();
                    }, 1000);
                }
            };
            socket.onclose = function (e) {
                console.error('[FavList WS] WebSocket 连接已关闭。代码:', e.code, '原因:', e.reason, '5秒后尝试重连...');
                setTimeout(connectFavListWebSocket, 5000);
            };
            socket.onerror = function (err) {
                console.error('[FavList WS] WebSocket 发生错误:', err);
            };
        }
        // 启动该页面的WebSocket连接
        connectFavListWebSocket();
    }
    // 调用上面定义的初始化函数
    initializeFavTrendListPage();

});
