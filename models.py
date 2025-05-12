from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    mobile = models.CharField(max_length=20)
    country = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    city = models.CharField(max_length=100)

    def __str__(self):
        return self.user.username

from django.db import models

class Account(models.Model):
    username = models.CharField(max_length=255)
    profile_link = models.URLField(null=True, blank=True)
    status = models.CharField(max_length=50, choices=[('Original', 'Original'), ('Fake', 'Fake'), ('Unknown', 'Unknown')])
    flagged = models.BooleanField(default=False)

    def __str__(self):
        return self.username


class Report(models.Model):
    account = models.ForeignKey(Account, related_name='reports', on_delete=models.CASCADE)
    report_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)

    def __str__(self):
        return f"Report for {self.account.username}"
