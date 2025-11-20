// static/js/trend_following_list.js

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

document.addEventListener('DOMContentLoaded', function () {
    // =========================================================================
    // === 策略监控中心 (trend_following_list.html) 功能 =======================
    // =========================================================================
    function initializeTrendListPage() {
        const tableBody = document.getElementById('trend-table-body');
        if (!tableBody) return; // 卫兵子句
        console.log('正在初始化【策略监控中心】页面功能...');
        document.querySelectorAll('.playbook-container').forEach(container => {
            const list = container.querySelector('.playbook-list');
            const toggleBtn = container.querySelector('.toggle-playbooks');
            if (list && toggleBtn) {
                // 1. 检查内容是否真的溢出，如果不溢出则隐藏“展开”按钮
                // 使用一个小的阈值（例如5px）来处理计算偏差
                if (list.scrollHeight <= 58 + 5) {
                    toggleBtn.style.display = 'none';
                } else {
                    // 2. 确保初始状态是折叠的，并设置正确的按钮文本
                    list.classList.remove('expanded');
                    toggleBtn.textContent = toggleBtn.dataset.textExpand;
                    toggleBtn.style.display = 'inline-block'; // 确保按钮可见
                }
            }
        });
        // --- 添加自选功能相关变量和函数 (逻辑不变, 仅调整了 handleAddFavorite 的参数) ---
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
                if (typeof showNotification === 'function') {
                    showNotification('无法加载自选状态，请刷新重试', 'error');
                }
            }
        }
        async function handleAddFavorite(button) {
            const stockCode = button.dataset.stockCode;
            if (!stockCode) {
                if (typeof showNotification === 'function') showNotification('无法获取股票代码，操作已取消', 'error');
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
                    if (typeof showNotification === 'function') showNotification(responseData.detail || `股票 ${stockCode} 操作成功！`, 'success');
                    favoriteStockCodes.add(stockCode);
                    updateButtonState(button, true);
                } else {
                    const errorMsg = responseData.detail || `添加 ${stockCode} 失败`;
                    if (typeof showNotification === 'function') showNotification(errorMsg, 'error');
                    if (response.status === 400 && errorMsg.includes('已存在')) {
                        updateButtonState(button, true);
                    } else {
                        updateButtonState(button, false);
                    }
                }
            } catch (error) {
                if (typeof showNotification === 'function') showNotification('网络错误，请稍后重试', 'error');
                updateButtonState(button, false);
            }
        }
        tableBody.addEventListener('click', function (event) {
            const toggleBtn = event.target.closest('.toggle-playbooks');
            const favBtn = event.target.closest('.add-to-favorites-btn');
            if (toggleBtn) {
                event.preventDefault();
                const list = toggleBtn.closest('.playbook-container').querySelector('.playbook-list');
                if (!list) return;
                list.classList.toggle('expanded');
                const isExpanded = list.classList.contains('expanded');
                toggleBtn.textContent = isExpanded ? toggleBtn.dataset.textCollapse : toggleBtn.dataset.textExpand;
                return; // 处理完折叠后，不再继续
            }
            if (favBtn) {
                if (favBtn.disabled) return;
                event.preventDefault();
                handleAddFavorite(favBtn); // 传递按钮元素
                return; // 处理完添加自选后，不再继续
            }
        });
        // 初始化
        initializeFavoriteButtons();
    }
    // 在这里调用上面定义的初始化函数，以确保它能够执行
    initializeTrendListPage();
});
