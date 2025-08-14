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
});
