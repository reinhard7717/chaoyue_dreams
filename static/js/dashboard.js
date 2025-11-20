// dashboard.js

// =========================================================================
// === 辅助函数 (全局可复用) ==============================================
// 将这些函数定义移到 DOMContentLoaded 监听器外部，使其成为全局函数
// 这样其他JS文件（如 fav_trend_following_list.js）才能调用它们
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
// === 页面初始化逻辑 ======================================================
// 将页面初始化逻辑保留在 DOMContentLoaded 监听器内部
// 确保这些代码在页面完全加载后才执行
// =========================================================================
document.addEventListener('DOMContentLoaded', function () {
    // 封装一个用于初始化主控台搜索功能的函数
    // 这样可以将页面特定的逻辑隔离开，提高代码的可读性和健壮性
    function initializeSearchFunctionality() {
        // 错误根源：尝试获取只在主控台页面存在的搜索输入框
        const searchInput = document.getElementById('searchInput');
        // 关键修复：添加一个 "卫兵子句"
        // 在尝试为 searchInput 添加任何事件监听器之前，首先检查它是否存在。
        // 如果 searchInput 为 null（即当前页面不是主控台页面），则函数直接返回，不执行后续代码。
        if (!searchInput) {
            // 在非主控台页面，这个日志会打印，是正常行为
            console.log('[dashboard.js] Search input not found on this page. Skipping search initialization.');
            return;
        }
        // 只有在 searchInput 元素存在时，才会执行以下代码
        console.log('[dashboard.js] Search input found. Initializing search functionality...');
        // 假设这里是您原来版本中为搜索框、结果容器等添加事件监听器的代码
        // 例如:
        // const searchResultsContainer = document.querySelector('.search-results-container');
        // searchInput.addEventListener('input', function() {
        //     // 您的搜索逻辑
        // });
        // ... 其他只应在主控台页面执行的JS代码
    }
    // 调用初始化函数
    initializeSearchFunctionality();
    // 此处可以安全地放置其他全局初始化代码
    console.log('[dashboard.js] DOMContentLoaded event fired and processed without fatal errors.');
});

