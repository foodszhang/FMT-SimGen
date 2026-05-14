#!/bin/bash
# Check MCX progress for uniform_1000_20k

TOTAL=1000
COMPLETED=$(ls data/uniform_1000_20k/samples/sample_*/sample_*.jnii 2>/dev/null | wc -l)
REMAINING=$((TOTAL - COMPLETED))
PERCENT=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc)

echo "=== MCX Progress ==="
echo "Completed: $COMPLETED / $TOTAL ($PERCENT%)"
echo "Remaining: $REMAINING"
echo ""
echo "=== Latest Log ==="
tail -5 mcx_uniform_1000_20k.log 2>/dev/null || echo "No log file found"
echo ""
echo "=== Process Status ==="
pgrep -f "run_all.py.*uniform_1000_20k" > /dev/null && echo "MCX pipeline is running" || echo "MCX pipeline not running"
