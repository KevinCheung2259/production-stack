#!/bin/bash
if [[ $# -ne 1 ]]; then
    echo "Usage $0 <router port>"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run router.py from the correct directory
python3 "$SCRIPT_DIR/app.py" --port "$1" \
    --service-discovery static \
    --static-backends "http://localhost:9004,http://localhost:9001,http://localhost:9002,http://localhost:9003" \
    --static-models "Nitral-AI/Captain-Eris_Violet-V0.420-12B,Nitral-AI/Captain-Eris_Violet-V0.420-12B,Nitral-AI/Captain-Eris_Violet-V0.420-12B,Nitral-AI/Captain-Eris_Violet-V0.420-12B" \
    --engine-stats-interval 15 \
    --log-stats \
    --routing-logic cache_aware_load_balancing \
    --session-key "X-Flow-Conversation-Id"

    #--routing-logic roundrobin

    #--routing-logic session \
    #--session-key "x-user-id" \
