#!/bin/bash
# Kill orchestrator after ~7 more generations (~120 min from now)
# Gen 3 is running, need gens 4-10 = 7 more × ~16 min = 112 min + buffer
ORCH_PID=1152048
WAIT_MINUTES=120

echo "Watchdog: will kill PID $ORCH_PID in $WAIT_MINUTES minutes (~gen 10)"
echo "Started at $(date)"

sleep $(( WAIT_MINUTES * 60 ))

echo "Time's up at $(date). Killing orchestrator..."
taskkill //F //PID $ORCH_PID 2>/dev/null
sleep 10
# Kill remaining workers
wmic process where "name='python.exe' and WorkingSetSize>400000000" call terminate 2>/dev/null
echo "Done."
