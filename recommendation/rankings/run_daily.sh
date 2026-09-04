#!/bin/sh
# 하루 한 번. crontab: 0 9 * * * /home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/rankings/run_daily.sh
cd /home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation
/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/.venv/bin/python rankings/collect.py >> rankings/data/collect.log 2>&1
