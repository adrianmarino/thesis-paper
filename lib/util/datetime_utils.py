from datetime import datetime


class DateTimeDiff:
    def __init__(self, days, hours, minutes, seconds):
        self.days    = days
        self.hours   = hours
        self.minutes = minutes 
        self.seconds = seconds

    def __str__(self): return f'{self.days} days -> {self.hours} hours -> {self.minutes} minutes -> {self.seconds} seconds'
    def __repr__(self): return str(self)


class Days:
    def __init__(self, value): self.value = value
    
    def to_hours(self): return self.value * 24

    def to_minutes(self): return self.to_hours() * 60

    def to_seconds(self): return self.to_minutes() * 60


class Seconds:
    def __init__(self, value): self.value = value
    
    def to_hours(self): return self.value // 60

    def to_minutes(self): return self.value // 3600 

    def to_days(self): return self.value // 3600
    




class DateTimeUtils:
    @staticmethod
    def now(): return datetime.now()


    @classmethod
    def diff_with_now(clz, datetime): return clz.diff(datetime, clz.now())
    
    @staticmethod
    def diff(datetime_a, datetime_b):
        diff = datetime_b - datetime_a

        days = Days(diff.days)

        hours   = days.to_hours()   + diff.seconds // 3600
        minutes = days.to_minutes() + diff.seconds // 60
        seconds = days.to_seconds() + diff.seconds

        return DateTimeDiff(days.value, hours, minutes, seconds)
