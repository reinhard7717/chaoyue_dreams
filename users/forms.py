from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.utils.translation import gettext_lazy as _
from .models import FavoriteStock, UserProfile

User = get_user_model()

class UserLoginForm(AuthenticationForm):
    """
    用户登录表单
    """
    username = forms.CharField(widget=forms.TextInput(attrs={
        'class': 'form-control', # 应用你在 theme.css 中定义的样式类
        'placeholder': '请输入用户名' # 添加占位符
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'form-control', # 应用你在 theme.css 中定义的样式类
        'placeholder': '请输入密码'
    }))
    remember_me = forms.BooleanField(
        label=_('记住我'),
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    error_messages = {
        'invalid_login': _(
            "请输入正确的用户名和密码。注意区分大小写。"
        ),
        'inactive': _("该账号已被禁用。"),
    }
    class Meta:
        model = User
        fields = ('username', 'password', 'remember_me')

class UserRegistrationForm(UserCreationForm):
    """
    用户注册表单（主要用于管理员创建用户）
    """
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入用户名'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': '请输入邮箱'}),
        }

class UserProfileForm(forms.ModelForm):
    """
    用户资料编辑表单
    """
    first_name = forms.CharField(
        label=_('名'),
        max_length=150,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    last_name = forms.CharField(
        label=_('姓'),
        max_length=150,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    email = forms.EmailField(
        label=_('电子邮箱'),
        required=False,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    class Meta:
        model = UserProfile
        fields = ('phone', 'avatar', 'bio', 'email_notification')
        widgets = {
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'bio': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'email_notification': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    def __init__(self, *args, **kwargs):
        """
        初始化用户资料表单
        """
        instance = kwargs.get('instance')
        if instance and hasattr(instance, 'user'):
            initial = kwargs.get('initial', {})
            initial.update({
                'first_name': instance.user.first_name,
                'last_name': instance.user.last_name,
                'email': instance.user.email,
            })
            kwargs['initial'] = initial
        super().__init__(*args, **kwargs)
    def save(self, commit=True):
        """
        保存用户资料
        """
        profile = super().save(commit=False)
        # 同时更新User模型中的字段
        if hasattr(profile, 'user'):
            profile.user.first_name = self.cleaned_data['first_name']
            profile.user.last_name = self.cleaned_data['last_name']
            profile.user.email = self.cleaned_data['email']
            if commit:
                profile.user.save()
        if commit:
            profile.save()
        return profile

class FavoriteStockForm(forms.ModelForm):
    """
    自选股表单
    """
    class Meta:
        model = FavoriteStock
        fields = ('stock', 'note', 'tags', 'is_pinned')
        widgets = {
            'stock': forms.Select(attrs={'class': 'form-control'}),
            'note': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入备注信息'}),
            'tags': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入标签，多个标签用空格分隔'}),
            'is_pinned': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        } 