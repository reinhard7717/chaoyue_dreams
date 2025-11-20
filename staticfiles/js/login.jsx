import React, { useState } from 'react';
import ReactDOM from 'react-dom';

/**
 * 登录表单组件
 * @param {Object} props 组件属性
 * @returns {JSX.Element} 登录表单组件
 */
const LoginForm = (props) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(true);
  const [error, setError] = useState('');
  const { csrfToken, loginUrl } = props;

  /**
   * 处理表单提交
   * @param {Event} e 事件对象
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username || !password) {
      setError('用户名和密码不能为空');
      return;
    }
    // 提交表单
    document.getElementById('login-form').submit();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 bg-white p-10 rounded-xl shadow-2xl">
        <div className="text-center">
          <div className="flex justify-center">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full flex items-center justify-center text-white text-2xl font-bold shadow-lg">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M12.395 2.553a1 1 0 00-1.45-.385c-.345.23-.614.558-.822.88-.214.33-.403.713-.57 1.116-.334.804-.614 1.768-.84 2.734a31.365 31.365 0 00-.613 3.58 2.64 2.64 0 01-.945-1.067c-.328-.68-.398-1.534-.398-2.654A1 1 0 005.05 6.05c-.867 0-1.563.64-1.563 1.17 0 1.323.526 2.337 1.262 3.004.737.669 1.7 1.054 2.28 1.125.062.14.125.28.187.42.267.6.532 1.188.792 1.752a19.89 19.89 0 001.113 2.12c.45.746.9 1.421 1.44 2.01.54.589 1.172 1.082 1.938 1.394.767.312 1.695.472 2.763.472 1.067 0 1.995-.16 2.763-.472.766-.312 1.399-.805 1.938-1.394.54-.589.99-1.264 1.44-2.01.44-.747.83-1.522 1.112-2.12.284-.598.547-1.185.793-1.753.062-.139.124-.279.187-.419.076-.17.147-.339.213-.504a1 1 0 00.172-1.016c-.125-.282-.396-.568-.822-.88-.424-.312-.99-.614-1.655-.88-.665-.267-1.43-.49-2.22-.668-.317-.07-.629-.131-.929-.184-.317-.055-.609-.103-.866-.144-.256-.041-.48-.075-.665-.106-.183-.03-.321-.054-.403-.072a1 1 0 00-1.064.588c-.05.113-.083.24-.104.374-.21.134-.038.275-.05.414-.012.14-.02.285-.025.43-.004.143-.006.277-.006.4v.268c.002.168.008.336.02.5L9.3 11.9l-.834-.1c-.086-.01-.154-.023-.204-.036a.482.482 0 01-.119-.046.402.402 0 01-.098-.082.276.276 0 01-.075-.11 1.578 1.578 0 01-.104-.458c-.01-.158-.017-.371-.017-.638 0-.259.006-.479.017-.638.011-.158.046-.313.104-.458a.276.276 0 01.075-.11.402.402 0 01.098-.082.482.482 0 01.119-.046c.05-.013.118-.025.204-.036l.834-.1c.023-.95.045-.176.068-.24.022-.063.047-.116.074-.164.336-.59.598-1.28.808-2.09.21-.81.366-1.708.502-2.676.374-2.661.62-4.937.38-5.42-.241-.482-1.074-.76-2.31-.76-.043 0-.09 0-.134.002-.175.006-.361.023-.56.06v3.37c0 1.12-.07 1.973-.398 2.654-.327.68-.783 1.126-1.4 1.315-.49.147-.97.246-1.426.337-.457.09-.886.18-1.284.337C5.309 6.95 6.1 7 7 7c.25 0 .492-.014.712-.04.22-.027.418-.074.58-.134.254-.094.448-.225.583-.41.135-.183.228-.425.292-.694.064-.27.1-.613.115-1.037.013-.425.02-.892.02-1.41v-.4c0-1.326-.076-2.247-.233-2.634-.158-.387-.485-.592-1.05-.592-.063 0-.121.005-.175.016-.317.063-.61.197-.926.43-.317.233-.59.514-.833.857a4.425 4.425 0 00-.53.966c-.134.33-.232.685-.317 1.07z" clipRule="evenodd" />
              </svg>
            </div>
          </div>
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            超越梦想
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            中国A股股票策略计算系统
          </p>
        </div>
        <form className="mt-8 space-y-6" id="login-form" method="POST" action={loginUrl}>
          <input type="hidden" name="csrfmiddlewaretoken" value={csrfToken} />
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="username" className="sr-only">用户名</label>
              <input
                id="username"
                name="username"
                type="text"
                autoComplete="username"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="用户名"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="password" className="sr-only">密码</label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="密码"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>
          {error && (
            <div className="text-red-500 text-sm mt-2">
              {error}
            </div>
          )}
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                id="remember_me"
                name="remember_me"
                type="checkbox"
                className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
              />
              <label htmlFor="remember_me" className="ml-2 block text-sm text-gray-900">
                记住我
              </label>
            </div>
          </div>
          <div>
            <button
              type="submit"
              onClick={handleSubmit}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              <span className="absolute left-0 inset-y-0 flex items-center pl-3">
                <svg className="h-5 w-5 text-indigo-500 group-hover:text-indigo-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                </svg>
              </span>
              登录
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// 在DOM加载完成后初始化React组件
document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('react-login');
  if (container) {
    const csrfToken = container.getAttribute('data-csrf-token');
    const loginUrl = container.getAttribute('data-login-url');
    ReactDOM.render(
      <LoginForm csrfToken={csrfToken} loginUrl={loginUrl} />,
      container
    );
  }
}); 