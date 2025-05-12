from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Account, Report

class AccountAdmin(admin.ModelAdmin):
    list_display = ('username', 'status', 'flagged', 'profile_link')
    list_filter = ('status', 'flagged')
    search_fields = ('username', 'profile_link')

class ReportAdmin(admin.ModelAdmin):
    list_display = ('account', 'created_at', 'resolved')
    list_filter = ('resolved',)
    search_fields = ('account__username',)

admin.site.register(Account, AccountAdmin)
admin.site.register(Report, ReportAdmin)
