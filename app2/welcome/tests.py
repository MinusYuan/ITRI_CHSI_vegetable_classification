from django.test import TestCase
from .models import log_history
from datetime import datetime, timedelta

# Create your tests here.
class FunctionTests(TestCase):
    def test_nested_function_ordering(self):
        log_history.objects.create(filename="1.jpg", written_time=datetime.utcnow())
        log_history.objects.create(filename="2.jpg", written_time=(datetime.utcnow() + timedelta(seconds=15)))
        log_history.objects.create(filename="3.jpg", written_time=(datetime.utcnow() + timedelta(seconds=22)))

        logs = log_history.objects.order_by("-written_time")
        self.assertQuerysetEqual(
            logs,
            [
                "3.jpg",
                "2.jpg",
                "1.jpg",
            ],
            lambda a: a.filename,
        )
