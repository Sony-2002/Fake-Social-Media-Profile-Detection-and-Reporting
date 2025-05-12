from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('homepage')  # redirect to your homepage
        else:
            messages.error(request, 'Invalid username or password')
    return render(request, 'login.html')


from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from .forms import CustomUserCreationForm
from .models import UserProfile

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Save profile data
            UserProfile.objects.create(
                user=user,
                mobile=form.cleaned_data.get('mobile'),
                country=form.cleaned_data.get('country'),
                state=form.cleaned_data.get('state'),
                city=form.cleaned_data.get('city')
            )
            login(request, user)
            return redirect('homepage')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})


from django.shortcuts import render
from .models import UserProfile
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404

def view_all_users(request):
    profiles = UserProfile.objects.select_related('user').all()
    return render(request, 'view_all_users.html', {'profiles': profiles})

@login_required
def userdata(request):
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    print(profile, created)  # Check if it's actually creating
    return render(request, 'userdata.html', {'profile': profile})


def home_view(request):
    return render(request, 'home.html')

import csv
import os
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings

def predict(request):
    return render(request, 'predict.html')

def bargraph(request):
    return render(request, 'bargraph.html')

def piechart(request):
    return render(request, 'piechart.html')


from django.shortcuts import render
import os
import csv
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
import os
import csv
from django.conf import settings
from django.core.files.storage import default_storage
from .models import Account, Report

def check_tweet(request):
    result = None
    matched_accounts = []
    report = None  # Initialize the report variable

    if request.method == 'POST':
        username = request.POST.get('username', '').strip().lower()
        profile_link = request.POST.get('profile_link', '').strip().lower()
        uploaded_file = request.FILES.get('csv_file')

        # Load reference tweets.csv
        reference_path = os.path.join(settings.BASE_DIR, 'main', 'static', 'social_media_model.csv')
        reference_data = []

        try:
            with open(reference_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    reference_data.append({
                        'username': row.get('username', '').strip().lower(),
                        'profile_link': row.get('profile_link', '').strip().lower(),
                        'message_type': row.get('message_type', 'Unknown')
                    })
        except FileNotFoundError:
            result = "social_media_model.csv not found."
            return render(request, 'predict.html', {'result': result})

        # If CSV uploaded
        if uploaded_file:
            try:
                temp_path = default_storage.save('tmp/' + uploaded_file.name, uploaded_file)
                abs_path = os.path.join(settings.MEDIA_ROOT, temp_path)

                with open(abs_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)

                    for row in reader:
                        u_name = row.get('username', '').strip().lower()
                        p_link = row.get('profile_link', '').strip().lower()
                        matched = False

                        for ref in reference_data:
                            if u_name and u_name == ref['username']:
                                matched_accounts.append({
                                    'username': u_name,
                                    'profile_link': ref['profile_link'],
                                    'status': ref['message_type']
                                })
                                matched = True
                                break
                            elif p_link and p_link == ref['profile_link']:
                                matched_accounts.append({
                                    'username': ref['username'],
                                    'profile_link': p_link,
                                    'status': ref['message_type']
                                })
                                matched = True
                                break

                        if not matched:
                            matched_accounts.append({
                                'username': u_name,
                                'profile_link': p_link,
                                'status': 'Unknown'
                            })

                result = f"Matched {len(matched_accounts)} accounts."

                # Generate the report summary and save it to the database
                report = generate_report(matched_accounts)

                # Save the report for each matched account
                for account in matched_accounts:
                    # Find the Account object (assumes Account exists with the username or profile link)
                    account_obj = Account.objects.filter(username=account['username']).first()

                    if account_obj:
                        # Create and save the report
                        report_text = generate_report([account])
                        Report.objects.create(account=account_obj, report_text=report_text)

            except Exception as e:
                result = f"Error processing uploaded CSV: {str(e)}"

        elif username or profile_link:
            for ref in reference_data:
                if username and username == ref['username']:
                    result = ref['message_type']
                    break
                elif profile_link and profile_link == ref['profile_link']:
                    result = ref['message_type']
                    break
            if not result:
                result = "Unknown"
            
            # Generate the report summary and save it to the database
            if result != "Unknown":
                report = generate_report([{'username': username, 'profile_link': profile_link, 'status': result}])

                # Save the report for the found account
                account_obj = Account.objects.filter(username=username).first()
                if account_obj:
                    report_text = generate_report([{'username': username, 'profile_link': profile_link, 'status': result}])
                    Report.objects.create(account=account_obj, report_text=report_text)

        else:
            result = "No input provided."

    return render(request, 'predict.html', {'result': result, 'matched_accounts': matched_accounts, 'report': report})


from keras.models import load_model
import os
import numpy as np
import pandas as pd
import joblib
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render

def check_tweets(request):
    result = None
    matched_accounts = []
    model_path = os.path.join(settings.BASE_DIR, 'main', 'model_train', 'social_media_model.h5')
    scaler_path = os.path.join(settings.BASE_DIR, 'main', 'model_train', 'scaler.pkl')
    encoders_path = os.path.join(settings.BASE_DIR, 'main', 'model_train', 'label_encoders.pkl')

    # model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)

    if request.method == 'POST':
        username = request.POST.get('username', '').strip().lower()
        profile_link = request.POST.get('profile_link', '').strip().lower()
        uploaded_file = request.FILES.get('csv_file')

        # Case 1: If a username is provided (manual input)
        if username and not uploaded_file:
            try:
                # Dummy data for prediction — DO NOT include 'username' field
                dummy_row = {
                    'profile pic': 1,
                    'nums/length username': len(username),
                    'fullname words': 1,
                    'nums/length fullname': len(username),
                    'name==username': 1,
                    'description length': 10,
                    'external url': 0,
                    'private': 0,
                    '#posts': 50,
                    '#followers': 1000,
                    '#follows': 200,
                }

                row_data = pd.DataFrame([dummy_row])

                # Encode categorical values
                for col in row_data.columns:
                    if row_data[col].dtype == 'object' or row_data[col].dtype == 'bool':
                        le = label_encoders.get(col)
                        if le:
                            row_data[col] = le.transform(row_data[col].astype(str).str.lower())

                # Scale
                row_scaled = scaler.transform(row_data)

                # Predict
                prediction = model.predict(row_scaled)
                label = np.argmax(prediction)

                status = 'Fake' if label == 1 else 'Real'
                matched_accounts.append({
                    'username': username,
                    'profile_link': profile_link or 'N/A',
                    'status': status
                })
                result = f"Prediction: {status} account based on username."

            except Exception as e:
                result = f"Error: {str(e)}"

        # Case 2: CSV upload with or without username/profile link
        elif uploaded_file:
            try:
                temp_path = default_storage.save('tmp/' + uploaded_file.name, uploaded_file)
                abs_path = os.path.join(settings.MEDIA_ROOT, temp_path)
                df = pd.read_csv(abs_path)

                df.columns = [col.strip().lower() for col in df.columns]
                df['username'] = df['username'].str.lower()
                df['profile link'] = df['profile link'].str.lower()

                found_row = None
                if username:
                    found_row = df[df['username'] == username]
                if (found_row is None or found_row.empty) and profile_link:
                    found_row = df[df['profile link'] == profile_link]

                if found_row is not None and not found_row.empty:
                    row = found_row.iloc[0]
                    features = ['profile pic', 'nums/length username', 'fullname words',
                                'nums/length fullname', 'name==username', 'description length',
                                'external url', 'private', '#posts', '#followers', '#follows']
                    
                    row_data = pd.DataFrame([row[features]])

                    for col in row_data.columns:
                        if row_data[col].dtype == 'object' or row_data[col].dtype == 'bool':
                            le = label_encoders.get(col)
                            if le:
                                row_data[col] = le.transform(row_data[col].astype(str).str.lower())

                    row_scaled = scaler.transform(row_data)
                    prediction = model.predict(row_scaled)
                    label = np.argmax(prediction)

                    status = 'Fake' if label == 1 else 'Real'
                    matched_accounts.append({
                        'username': row.get('username', 'unknown'),
                        'profile_link': row.get('profile link', 'unknown'),
                        'status': status
                    })
                    result = f"Prediction: {status} account based on data from uploaded CSV."

                else:
                    result = "User not found in uploaded CSV."

            except Exception as e:
                result = f"Error: {str(e)}"

        else:
            result = "Please upload a CSV file or enter a username for prediction."

    return render(request, 'predict.html', {
        'result': result,
        'matched_accounts': matched_accounts
    })


def generate_report(matched_accounts):
    """Generates a report indicating if the account is likely fake or original."""
    
    report_lines = []
    
    for account in matched_accounts:
        username = account.get('username', '')
        profile_link = account.get('profile_link', '')
        status = account.get('status', 'Unknown')

        if status == 'Fake':
            explanation = f"The account with username '{username}' and profile link '{profile_link}' is marked as Fake due to no matching data in social media."
        elif status == 'Original':
            explanation = f"The account with username '{username}' and profile link '{profile_link}' is marked as Original as it matches known data from the in social media."
        else:
            explanation = f"The account with username '{username}' and profile link '{profile_link}' could not be verified and is marked as Unknown."

        report_lines.append(f"Account: {username}\nStatus: {status}\nExplanation: {explanation}\n")
    
    return "\n".join(report_lines)





from django.shortcuts import render
from .models import Account, Report

def admin_dashboard(request):
    # Get flagged accounts
    flagged_accounts = Account.objects.filter(flagged=True)
    
    # Get unresolved reports
    unresolved_reports = Report.objects.filter(resolved=False)
    
    return render(request, 'admin_dashboard.html', {
        'flagged_accounts': flagged_accounts,
        'unresolved_reports': unresolved_reports
    })

def resolve_report(request, report_id):
    report = Report.objects.get(id=report_id)
    if request.method == 'POST':
        report.resolved = True
        report.save()
        return redirect('admin_dashboard')
    return render(request, 'resolve_report.html', {'report': report})


import plotly.express as px
import pandas as pd
from django.db import models
from django.shortcuts import render
from .models import Account

def generate_account_status_chart(request):
    # Get counts grouped by status
    account_data = Account.objects.values('status').annotate(count=models.Count('id'))

    # Convert queryset to DataFrame
    df = pd.DataFrame(list(account_data))

    if df.empty:
        return render(request, 'account_status_chart.html', {'chart_html': '<h4>No data available.</h4>'})

    # Create chart
    chart = px.bar(df, x='status', y='count', title="Account Status Distribution")
    chart_html = chart.to_html(full_html=False)

    return render(request, 'account_status_chart.html', {'chart_html': chart_html})

import csv
from django.http import HttpResponse
from .models import Account

def download_csv_report(request):
    # Create an HTTP response with CSV content
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="account_report.csv"'

    writer = csv.writer(response)
    writer.writerow(['Username', 'Profile Link', 'Status', 'Flagged'])

    # Get data from the database
    accounts = Account.objects.all()

    for account in accounts:
        writer.writerow([account.username, account.profile_link, account.status, account.flagged])

    return response


