# users\views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.views import LoginView
from .forms import UserLoginForm, UserProfileForm, FavoriteStockForm
from .models import FavoriteStock, UserProfile
from stock_models.stock_basic import StockInfo
from django.contrib.auth.forms import UserCreationForm


@login_required # 确保用户已登录
def user_home_view(request):
    # 后续可以传递初始数据到模板
    context = {}
    return render(request, 'users/home.html', context)

def favorite_list_view(request):
    # 后续可以传递初始数据到模板
    context = {}
    return render(request, 'users/favorite_list.html', context)

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()  # 保存普通用户
            print("新用户注册成功")  # 调试信息
            return redirect(reverse('users:login'))  # 注册成功后跳转到 /users/login/
    else:
        form = UserCreationForm()
    return render(request, 'users/register.html', {'form': form})













def home(request):
    """
    主页视图
    """
    if not request.user.is_authenticated:
        return redirect('login')
    # 获取用户的自选股
    favorite_stocks = FavoriteStock.objects.filter(user=request.user)
    context = {
        'favorite_stocks': favorite_stocks,
    }
    return render(request, 'users/home.html', context)

class CustomLoginView(LoginView):
    """
    自定义登录视图
    """
    template_name = 'users/login.html'
    authentication_form = UserLoginForm
    redirect_authenticated_user = True
    def form_valid(self, form):
        """
        表单验证成功的处理
        """
        remember_me = form.cleaned_data.get('remember_me')
        if not remember_me:
            # 如果用户未勾选"记住我"，设置session过期时间为关闭浏览器
            self.request.session.set_expiry(0)
        # 记录用户登录IP
        user = form.get_user()
        if hasattr(user, 'profile'):
            user.profile.last_login_ip = self.request.META.get('REMOTE_ADDR')
            user.profile.save(update_fields=['last_login_ip'])
        else:
            # 如果用户没有资料，创建一个
            profile = UserProfile.objects.create(
                user=user,
                last_login_ip=self.request.META.get('REMOTE_ADDR')
            )
        return super().form_valid(form)

@login_required
def profile_view(request):
    """
    用户个人资料页面
    """
    # 确保用户有资料，如果没有则创建
    if not hasattr(request.user, 'profile'):
        profile = UserProfile.objects.create(user=request.user)
    else:
        profile = request.user.profile
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, _('个人资料更新成功！'))
            return redirect('profile')
    else:
        form = UserProfileForm(instance=profile)
    return render(request, 'users/profile.html', {'form': form})

@login_required
def favorite_stock_list(request):
    """
    自选股列表页面
    """
    favorite_stocks = FavoriteStock.objects.filter(user=request.user)
    # 获取所有标签（去重）
    tags = set()
    for stock in favorite_stocks:
        if stock.tags:
            for tag in stock.tags.split():
                tags.add(tag)
    return render(request, 'users/favorite_stock_list.html', {
        'favorite_stocks': favorite_stocks,
        'tags': tags
    })

@login_required
def add_favorite_stock(request):
    """
    添加自选股页面
    """
    if request.method == 'POST':
        form = FavoriteStockForm(request.POST)
        if form.is_valid():
            favorite_stock = form.save(commit=False)
            favorite_stock.user = request.user
            favorite_stock.save()
            messages.success(request, _(f'已成功添加自选股：{favorite_stock.stock.stock_name}'))
            return redirect('favorite_stock_list')
    else:
        form = FavoriteStockForm()
    # 获取所有股票列表
    stocks = StockInfo.objects.all().order_by('stock_code')
    return render(request, 'users/favorite_stock_form.html', {
        'form': form, 
        'title': '添加自选股',
        'stocks': stocks
    })

@login_required
def edit_favorite_stock(request, pk):
    """
    编辑自选股页面
    """
    favorite_stock = get_object_or_404(FavoriteStock, pk=pk, user=request.user)
    if request.method == 'POST':
        form = FavoriteStockForm(request.POST, instance=favorite_stock)
        if form.is_valid():
            form.save()
            messages.success(request, _(f'已成功更新自选股：{favorite_stock.stock.stock_name}'))
            return redirect('favorite_stock_list')
    else:
        form = FavoriteStockForm(instance=favorite_stock)
    # 获取所有股票列表
    stocks = StockInfo.objects.all().order_by('stock_code')
    return render(request, 'users/favorite_stock_form.html', {
        'form': form, 
        'title': '编辑自选股',
        'stocks': stocks
    })

@login_required
def delete_favorite_stock(request, pk):
    """
    删除自选股
    """
    favorite_stock = get_object_or_404(FavoriteStock, pk=pk, user=request.user)
    stock_name = favorite_stock.stock.stock_name
    favorite_stock.delete()
    messages.success(request, _(f'已成功删除自选股：{stock_name}'))
    return redirect('favorite_stock_list')
