// static\js\trend_following_list.js

document.addEventListener('DOMContentLoaded', function () {
    // =========================================================================
    // === 策略监控中心 (trend_following_list.html) 功能 =======================
    // =========================================================================
    function initializeTrendListPage() {
        const tableBody = document.getElementById('trend-table-body');
        if (!tableBody) return; // 卫兵子句

        console.log('正在初始化【策略监控中心】页面功能...');

        // --- 折叠功能 ---
        // 将两个不同的事件处理逻辑分开，避免冲突
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
                // 假设 getCookie 和 showNotification 是在 dashboard.js 中定义的全局函数
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

            if (!stockCode) {
                showNotification('无法获取股票代码，操作已取消', 'error');
                console.error('错误：点击了添加自选按钮，但未能从 data-stock-code 属性中获取到值。');
                return;
            }

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
                    })
                });

                const responseData = await response.json();

                if (response.ok) {
                    showNotification(responseData.detail || `股票 ${stockCode} 操作成功！`, 'success');
                    favoriteStockCodes.add(stockCode);
                    updateButtonState(button, true);
                } else {
                    const errorMsg = responseData.detail || `添加 ${stockCode} 失败`;
                    showNotification(errorMsg, 'error');
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

        // 为添加自选按钮单独设置一个事件监听器
        tableBody.addEventListener('click', handleAddFavorite);

        // 初始化
        initializeFavoriteButtons();
    }

    // 在这里调用上面定义的初始化函数，以确保它能够执行
    initializeTrendListPage();
});
