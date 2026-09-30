"""Calendar-window and rate regression cases use real SQLite query semantics."""
import sqlite3
from datetime import datetime, timezone

import pytest
from app import analytics


@pytest.mark.parametrize(('values','expected'), [
    ([7]*7, 'stable'), ([7]*9, 'stable'), ([7]*30, 'stable'),
    ([7,7,7,6,6,6,6], 'decreasing'), ([6,6,6,7,7,7,7], 'increasing'),
    ([0]*7, 'unknown'), ([1]*3, 'unknown'), ([1,1,2,2], 'increasing'),
])
def test_trend_compares_daily_rates(values, expected):
    assert analytics._trend(values) == expected


@pytest.mark.parametrize('window_days', [7, 30, 90])
def test_calendar_window_includes_first_day_midnight(monkeypatch, window_days):
    from datetime import timedelta
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 30, 15, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(analytics, 'datetime', FixedDatetime)
    first = (FixedDatetime.now() - timedelta(days=window_days-1)).replace(hour=0, minute=0)
    dates = [(first-timedelta(seconds=1)).isoformat(), first.isoformat(), first.replace(hour=8).isoformat(), FixedDatetime.now().isoformat()]
    class DB:
        def __init__(self):
            self.conn=sqlite3.connect(':memory:');self.conn.row_factory=sqlite3.Row
            self.conn.executescript('''CREATE TABLE events(id,status,confidence,explanation,first_seen_at,last_seen_at,home_id,event_type,care_recipient_id);
            CREATE TABLE summaries(id,subject_user_id,care_recipient_id,status,trend,explanation,limitations,created_at,model_version,home_id);''')
            for i, date in enumerate(dates):
                self.conn.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)',(str(i),'needs_review',.8,'test',date,date,'home','fall_suspected',None))
                self.conn.execute('INSERT INTO summaries VALUES (?,?,?,?,?,?,?,?,?,?)',(str(i),None,None,'ok','stable','test','[]',date,'test','home'))
        def many(self, sql, params):
            return [dict(r) for r in self.conn.execute(sql,params)]
    result=analytics.care_analytics(DB(),'home',window_days=window_days)
    assert result['fall']['total_signals']==3
    assert result['daily_check_in']['total']==3
    assert result['fall']['by_day'][0]['count']==2
    assert result['daily_check_in']['by_day'][0]['count']==2
    assert result['event_counts']=={'fall_suspected':3}
