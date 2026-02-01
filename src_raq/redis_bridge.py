"""
Redis client bridge for Pipecat RAG integration.

This module provides a clean interface to the Redis-based RAG and RL
functionality implemented in brain.py. All functions are safe to call
even if Redis is unavailable - they gracefully degrade.

Public API:
    - query_knowledge_base(user_query, top_k) - RAG vector search
    - get_learned_action(screen_hash, intent) - Best RL action lookup
    - store_reward(screen_hash, intent, sam_mask_id, reward) - Store RL reward
    - hash_screen(image_data) - Screen perceptual hashing
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Import Redis brain functions from local brain.py
try:
    from brain import (
        retrieve_iam_steps,
        generate_screen_hash,
        update_reward,
        get_best_action
    )
    REDIS_AVAILABLE = True
    logger.info("✓ Redis brain module loaded successfully")
except ImportError as e:
    logger.warning(f"Redis brain module not available: {e}")
    logger.warning("Falling back to placeholder mode")
    REDIS_AVAILABLE = False


# =============================================================================
# RAG Vector Search
# =============================================================================

def query_knowledge_base(user_query: str, top_k: int = 2) -> str:
    """
    Query the Redis vector database for relevant domain knowledge.
    
    Args:
        user_query: User's voice transcript or intent
        top_k: Number of results to return
        
    Returns:
        Formatted string with retrieved knowledge chunks
    """
    if not REDIS_AVAILABLE:
        return "Redis not configured. Using placeholder knowledge."
    
    try:
        result = retrieve_iam_steps(user_query, top_k=top_k)
        
        if not result.get("results"):
            return "No specific knowledge found for this query."
        
        # Format results into readable text
        chunks = []
        for i, item in enumerate(result["results"], 1):
            text = item.get("text", "")
            similarity = item.get("similarity", 0.0)
            chunks.append(f"[Source {i}] (relevance: {similarity:.2f}) {text[:500]}...")
        
        return "\n\n".join(chunks)
    
    except Exception as e:
        logger.error(f"Failed to query knowledge base: {e}")
        return "Error retrieving knowledge from Redis."


# =============================================================================
# Reinforcement Learning / Episodic Memory
# =============================================================================

def get_learned_action(screen_hash: str, intent: str) -> Optional[Dict]:
    """
    Get the best learned action for a given screen state and intent.
    
    Args:
        screen_hash: Perceptual hash of the current screen
        intent: User's intent (e.g., "create_bucket")
        
    Returns:
        Dict with best_mask_id and score, or None if no data
    """
    if not REDIS_AVAILABLE:
        return None
    
    try:
        return get_best_action(screen_hash, intent)
    except Exception as e:
        logger.error(f"Failed to get best action: {e}")
        return None


def store_reward(
    screen_hash: str, 
    intent: str, 
    sam_mask_id: str, 
    reward: float
) -> bool:
    """
    Store a reward for a specific action.
    
    Args:
        screen_hash: Perceptual hash of the screen
        intent: User's intent
        sam_mask_id: Identifier for the clicked mask
        reward: Reward value (positive or negative)
        
    Returns:
        True if successful
    """
    if not REDIS_AVAILABLE:
        logger.warning("Redis not available, cannot store reward")
        return False
    
    try:
        update_reward(screen_hash, intent, sam_mask_id, reward)
        logger.info(f"Stored reward {reward} for {sam_mask_id} on screen {screen_hash[:8]}")
        return True
    except Exception as e:
        logger.error(f"Failed to store reward: {e}")
        return False


# =============================================================================
# Screen Hashing
# =============================================================================

def hash_screen(image_data) -> Optional[str]:
    """
    Generate perceptual hash for a screen image.
    
    Args:
        image_data: PIL Image, bytes, or base64 string
        
    Returns:
        Perceptual hash string, or None if error
    """
    if not REDIS_AVAILABLE:
        return None
    
    try:
        return generate_screen_hash(image_data)
    except Exception as e:
        logger.error(f"Failed to hash screen: {e}")
        return None
