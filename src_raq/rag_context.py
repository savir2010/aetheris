"""
Dynamic RAG context management for Pipecat LLM prompts.

This module provides thread-safe global context management and 
system prompt building that combines:
1. Domain Knowledge (RAG from Redis vector search)
2. Episodic Memory (RL from learned actions)

Public API:
    - update_rag_context(new_context) - Thread-safe context update
    - get_rag_context() - Get current RAG context
    - clear_rag_context() - Clear the context
    - build_system_prompt(template, user_query) - Combine RAG + RL for prompts
"""

import threading
import logging

logger = logging.getLogger(__name__)

# ============================================
# GLOBAL RAG CONTEXT
# ============================================

# Global variable holding current RAG context
current_rag_context: str = ""

# Thread-safe lock for context updates
_rag_context_lock = threading.Lock()

# ============================================
# PUBLIC API
# ============================================

def update_rag_context(new_context: str) -> None:
    """
    Update the global RAG context (thread-safe)
    
    Args:
        new_context: New context string to inject into LLM prompts
    """
    global current_rag_context
    
    with _rag_context_lock:
        old_length = len(current_rag_context)
        current_rag_context = new_context
        logger.info(f"RAG context updated: {old_length} → {len(new_context)} chars")


def get_rag_context() -> str:
    """Get current RAG context (thread-safe)"""
    with _rag_context_lock:
        return current_rag_context


def clear_rag_context() -> None:
    """Clear the RAG context"""
    update_rag_context("")


def update_reward_store(state_key: str, reward: float):
    """
    Update the episodic reward store based on user behavioral feedback
    """
    logger.info(f"Episodic memory updated for {state_key}: Reward {reward}")
    # Actual storage happens via redis_bridge.store_reward()


def get_current_screen_hash() -> str:
    """Hook to get the latest vision-based screen identifier"""
    # Placeholder - will be wired to screen capture processor
    return "current_screen_sha256"


def build_system_prompt(template: str, user_query: str = "") -> str:
    """
    Final Aggregator: Combine Knowledge (RAG) + Episodic Memory (RL)
    
    Args:
        template: System prompt template with {rag_context} placeholder
        user_query: Optional user query for vector search (if empty, uses global context)
        
    Returns:
        Complete system prompt with RAG and RL context injected
    """
    from redis_bridge import query_knowledge_base, get_learned_action
    
    # 1. Get Domain Knowledge from Redis Vector Search
    if user_query:
        # Use vector search for specific query
        domain_knowledge = query_knowledge_base(user_query, top_k=2)
    else:
        # Fall back to global RAG context
        domain_knowledge = get_rag_context()
    
    if not domain_knowledge or domain_knowledge.strip() == "":
        domain_knowledge = "No specific context provided."
    
    # 2. Get Episodic Knowledge (Learned Actions from RL)
    # Note: This requires screen_hash and intent from the current conversation context
    # For now, we use a placeholder. In production, you would:
    # - Get screen_hash from vision_hooks.hash_screen()
    # - Extract intent from the LLM's understanding
    
    episodic_memory = ""
    try:
        # Example: get_learned_action(screen_hash, intent)
        # best_action = get_learned_action("current_hash", "user_intent")
        # if best_action:
        #     episodic_memory = f"LEARNED STRATEGY: Click target {best_action['best_mask_id']} (confidence: {best_action['score']:.2f})"
        
        # Placeholder until we wire screen capture + intent extraction
        episodic_memory = ""
    except Exception as e:
        logger.error(f"Failed to retrieve episodic memory: {e}")
        episodic_memory = ""
    
    # 3. Inject Combined Intelligence into Template
    context_block = f"{domain_knowledge}\n\n{episodic_memory}" if episodic_memory else domain_knowledge
    
    prompt = template.format(rag_context=context_block)
    logger.debug("Successfully aggregated RAG + RL for System Prompt")
    
    return prompt
