from django.urls import path
from .views import *
from django.contrib.auth.views import LogoutView


from . import views

urlpatterns = [
    path('', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('home/', home_view, name='homepage'),
    path('predict/', predict, name='predict'),
    path('bargraph/', bargraph, name='bargraph'),
    path('piechart/', piechart, name='piechart'),
    path('check-tweet/', views.check_tweet, name='check_tweet'),
    path('view_remote_users/', views.view_all_users, name='view_all_users'),
    path('userdata/', views.userdata, name='userdata'),
    path('logout/', LogoutView.as_view(next_page='login'), name='logout'),
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('report/resolve/<int:report_id>/', views.resolve_report, name='resolve_report'),
    path('download/csv/', views.download_csv_report, name='download_csv_report'),
    path('status-chart/', views.generate_account_status_chart, name='account_status_chart'),
]