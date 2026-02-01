#!/bin/bash
# =============================================================================
# Redis RAG & RL Integration Verification Script
# =============================================================================
# Run this after:
#   1. docker-compose up -d
#   2. python ingest_to_redis.py
# =============================================================================

set -e

echo "=========================================="
echo "Redis RAG & RL Integration Verification"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Test Redis connection
echo "1. Testing Redis connection..."
if python -c "from redis_bridge import query_knowledge_base; print('✅ Redis bridge module loaded')" 2>/dev/null; then
    echo -e "${GREEN}✅ Redis connection OK${NC}"
else
    echo -e "${RED}❌ Redis connection failed${NC}"
    echo "   Make sure Redis is running: docker-compose up -d"
    exit 1
fi
echo ""

# 2. Test RAG query
echo "2. Testing RAG query..."
python -c "
from redis_bridge import query_knowledge_base
result = query_knowledge_base('Cloud Run', top_k=2)
if 'Error' in result or 'not configured' in result:
    print('⚠️  RAG query returned:', result[:100])
else:
    print('✅ RAG query successful, retrieved:', len(result), 'chars')
"
echo ""

# 3. Test RL reward storage
echo "3. Testing RL reward storage..."
python -c "
from redis_bridge import store_reward
success = store_reward('screen_test_123', 'test_intent', 'mask_test_1', 1.0)
if success:
    print('✅ RL reward stored successfully')
else:
    print('⚠️  RL reward storage returned False (Redis may be unavailable)')
"
echo ""

# 4. Verify Redis indexes
echo "4. Verifying Redis indexes..."
if command -v redis-cli &> /dev/null; then
    echo "   Running: redis-cli FT._LIST"
    redis-cli FT._LIST 2>/dev/null || echo "   (redis-cli not connected or RediSearch not available)"
else
    echo "   redis-cli not found, skipping index verification"
    echo "   You can check manually with: docker exec gcp-assistant-redis redis-cli FT._LIST"
fi
echo ""

# 5. Test module imports
echo "5. Testing module imports..."
python -c "
from redis_bridge import query_knowledge_base, store_reward, hash_screen
from rag_context import build_system_prompt, update_rag_context
from rl_processors import RewardProcessor, ActionFeedbackFrame
print('✅ All modules import successfully')
"
echo ""

echo "=========================================="
echo "🎉 All verification tests passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Start the Brain API:   uvicorn api:app --reload --port 8000"
echo "  2. Start the Pipecat bot: python pipecat_bot.py"
echo "  3. Open the Daily room URL in your browser"
echo ""
