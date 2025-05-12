import csv
from django.core.management.base import BaseCommand
from main.models import Account  # Replace 'main' with your app name if different

class Command(BaseCommand):
    help = 'Import accounts from CSV'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']

        with open(csv_file, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                username = row.get('username')
                profile_link = row.get('profile_link')
                status = row.get('status', 'Unknown')
                flagged = row.get('flagged', 'False').lower() in ['true', '1', 'yes']

                account, created = Account.objects.get_or_create(
                    username=username,
                    defaults={
                        'profile_link': profile_link,
                        'status': status if status in ['Original', 'Fake', 'Unknown'] else 'Unknown',
                        'flagged': flagged,
                    }
                )

                if created:
                    self.stdout.write(self.style.SUCCESS(f'Created: {username}'))
                else:
                    self.stdout.write(self.style.WARNING(f'Skipped (already exists): {username}'))
