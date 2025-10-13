"""
Memory Management Module for MCPAgent

This module provides intelligent context management to reduce token usage in API calls.
It implements automatic summarization and trimming of conversation history while preserving
important context for the AI model.

Key Features:
- Token estimation using tiktoken (GPT-style tokenization)
- Automatic summarization of older messages when approaching token limits
- Preservation of system messages and recent context
- Smart compression of tool call results
- Fallback token estimation when tiktoken is unavailable

Author: RudraModi360
Date: 2025-10-13
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class MemoryManager:
    """
    Manages conversation context with automatic summarization and token limiting.
    
    This class helps prevent context overflow by:
    1. Estimating token count for each message
    2. Summarizing older messages when approaching token limits
    3. Keeping recent messages intact for better context continuity
    4. Preserving system messages throughout the conversation
    
    Attributes:
        max_tokens (int): Maximum allowed tokens for the context window
        summary_trigger_ratio (float): When to trigger summarization (0.0-1.0)
        keep_recent_messages (int): Number of recent messages to keep intact
        messages (List[Dict]): The conversation history
        summarization_count (int): Number of times summarization has occurred
    """
    
    def __init__(
        self,
        max_tokens: int = 6000,
        summary_trigger_ratio: float = 0.75,
        keep_recent_messages: int = 6
    ):
        """
        Initialize the MemoryManager.
        
        Args:
            max_tokens: Maximum tokens allowed in context (default: 6000)
            summary_trigger_ratio: Percentage of max_tokens that triggers summarization (default: 0.75)
            keep_recent_messages: Number of recent messages to preserve during summarization (default: 6)
        """
        self.max_tokens = max_tokens
        self.summary_trigger_ratio = summary_trigger_ratio
        self.keep_recent_messages = keep_recent_messages
        self.messages: List[Dict[str, Any]] = []
        self.summarization_count = 0
        
        # Try to use tiktoken for accurate token counting, fallback to heuristic
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            self.use_tiktoken = True
        except ImportError:
            self.tokenizer = None
            self.use_tiktoken = False
            print("Warning: tiktoken not available. Using heuristic token estimation.")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Uses tiktoken when available for accurate counting, otherwise falls back
        to a heuristic (words * 1.3 + characters / 4).
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
            
        if self.use_tiktoken and self.tokenizer:
            # Accurate token counting using tiktoken
            return len(self.tokenizer.encode(text))
        else:
            # Fallback heuristic: ~1.3 tokens per word + characters/4 for punctuation
            # This is a rough approximation based on empirical data
            word_count = len(text.split())
            char_count = len(text)
            return int(word_count * 1.3 + char_count / 4)
    
    def estimate_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        Estimate tokens for a complete message including metadata.
        
        Accounts for:
        - Role tokens
        - Content tokens
        - Tool call overhead
        - JSON structure overhead
        
        Args:
            message: The message dictionary to estimate
            
        Returns:
            Estimated token count for the entire message
        """
        total = 0
        
        # Role adds ~1 token
        if "role" in message:
            total += 1
        
        # Content tokens
        if "content" in message and message["content"]:
            if isinstance(message["content"], str):
                total += self.estimate_tokens(message["content"])
            else:
                # Handle non-string content (shouldn't normally happen)
                total += self.estimate_tokens(str(message["content"]))
        
        # Tool calls add significant overhead
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                # Function name, arguments, and JSON structure
                total += self.estimate_tokens(tool_call.function.name)
                total += self.estimate_tokens(tool_call.function.arguments)
                total += 10  # Overhead for tool call structure
        
        # Tool call response overhead
        if "tool_call_id" in message:
            total += 5  # Overhead for tool response structure
        
        # Add small overhead for message structure (role, formatting, etc.)
        total += 4
        
        return total
    
    def get_total_tokens(self) -> int:
        """
        Calculate total tokens across all messages.
        
        Returns:
            Total estimated token count for all messages
        """
        return sum(self.estimate_message_tokens(msg) for msg in self.messages)
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the conversation history.
        
        Simply appends the message - trimming happens separately
        via get_trimmed_messages() before API calls.
        
        Args:
            message: Message dictionary to add
        """
        self.messages.append(message)
    
    def _compress_tool_result(self, content: str, max_length: int = 200) -> str:
        """
        Compress lengthy tool results while preserving key information.
        
        Strategies:
        1. If content is JSON, extract key fields
        2. If content is long text, truncate with summary
        3. Preserve error messages completely
        
        Args:
            content: Tool result content to compress
            max_length: Maximum length for compressed content
            
        Returns:
            Compressed content string
        """
        if len(content) <= max_length:
            return content
        
        # Check if it's an error - preserve errors completely
        if "error" in content.lower() or "fail" in content.lower():
            return content
        
        # Try to parse as JSON and extract key information
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # Keep only important keys
                important_keys = ["success", "result", "output", "error", "status", "path", "name"]
                compressed = {k: v for k, v in data.items() if k in important_keys}
                compressed_str = json.dumps(compressed)
                if len(compressed_str) <= max_length:
                    return compressed_str
            elif isinstance(data, list):
                # For lists, show count and first few items
                return f"[List with {len(data)} items, first few: {json.dumps(data[:3])}...]"
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback: truncate with ellipsis
        return content[:max_length] + f"... [truncated, original length: {len(content)}]"
    
    def _create_summary(self, messages_to_summarize: List[Dict[str, Any]]) -> str:
        """
        Create a concise summary of older messages.
        
        Focuses on:
        - User requests and intents
        - Key tool executions and results
        - Important decisions or outcomes
        - Errors or issues encountered
        
        Args:
            messages_to_summarize: List of messages to summarize
            
        Returns:
            Summary string
        """
        summary_parts = []
        user_requests = []
        tool_actions = []
        assistant_responses = []
        errors = []
        
        for msg in messages_to_summarize:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                # Capture user intents
                if content:
                    truncated = content[:100] + "..." if len(content) > 100 else content
                    user_requests.append(truncated)
            
            elif role == "assistant":
                # Capture key assistant responses (non-tool-calling)
                if content and not hasattr(msg, "tool_calls"):
                    truncated = content[:100] + "..." if len(content) > 100 else content
                    assistant_responses.append(truncated)
                
                # Track tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_actions.append(tc.function.name)
            
            elif role == "tool":
                # Check for errors in tool results
                if content and ("error" in content.lower() or "fail" in content.lower()):
                    errors.append(f"Tool error: {content[:80]}")
        
        # Build summary
        if user_requests:
            summary_parts.append(f"User asked about: {'; '.join(user_requests[:3])}")
        
        if tool_actions:
            unique_tools = list(set(tool_actions))
            summary_parts.append(f"Tools used: {', '.join(unique_tools[:5])}")
        
        if errors:
            summary_parts.append(f"Issues encountered: {'; '.join(errors[:2])}")
        
        if assistant_responses:
            summary_parts.append(f"Key responses: {'; '.join(assistant_responses[:2])}")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        summary = f"[Summary #{self.summarization_count + 1} at {timestamp}] " + " | ".join(summary_parts)
        
        return summary if summary_parts else "[Previous conversation context]"
    
    def summarize_old_context(self) -> None:
        """
        Compress older messages into a summary to reduce token count.
        
        Strategy:
        1. Always preserve system message (index 0)
        2. Keep the last N messages intact (recent context)
        3. Summarize everything in between
        4. Replace summarized messages with a single summary message
        
        This reduces token count by ~60-80% while preserving context.
        """
        if len(self.messages) <= self.keep_recent_messages + 1:
            # Not enough messages to summarize (system + recent)
            return
        
        # Separate messages into: system, old (to summarize), and recent (to keep)
        system_message = self.messages[0]  # Always the system message
        
        # Calculate split point
        split_point = len(self.messages) - self.keep_recent_messages
        if split_point <= 1:
            # Not enough old messages to summarize
            return
        
        messages_to_summarize = self.messages[1:split_point]
        recent_messages = self.messages[split_point:]
        
        # Create summary of old messages
        summary_text = self._create_summary(messages_to_summarize)
        summary_message = {
            "role": "system",
            "content": summary_text,
            "is_summary": True  # Mark as summary for potential filtering
        }
        
        # Rebuild messages list: [system, summary, recent...]
        self.messages = [system_message, summary_message] + recent_messages
        self.summarization_count += 1
        
        # Log the compression result
        old_tokens = sum(self.estimate_message_tokens(msg) for msg in messages_to_summarize)
        new_tokens = self.estimate_message_tokens(summary_message)
        reduction = ((old_tokens - new_tokens) / old_tokens * 100) if old_tokens > 0 else 0
        
        print(f"✓ Context summarized: {len(messages_to_summarize)} messages ({old_tokens} tokens) "
              f"→ 1 summary ({new_tokens} tokens) | {reduction:.1f}% reduction")
    
    def get_trimmed_messages(self) -> List[Dict[str, Any]]:
        """
        Get the message list, automatically summarizing if needed.
        
        This is the main method to call before making API requests.
        It ensures the context stays within token limits.
        
        Process:
        1. Calculate current token count
        2. If over threshold, trigger summarization
        3. Repeat if still over limit
        4. Return trimmed message list
        
        Returns:
            List of messages ready for API call, within token limits
        """
        # Check if we need to summarize
        current_tokens = self.get_total_tokens()
        trigger_threshold = int(self.max_tokens * self.summary_trigger_ratio)
        
        # May need multiple rounds of summarization for very long contexts
        max_summarization_rounds = 3
        rounds = 0
        
        while current_tokens > trigger_threshold and rounds < max_summarization_rounds:
            if len(self.messages) <= self.keep_recent_messages + 1:
                # Can't summarize further without losing recent context
                print(f"⚠ Warning: Token limit reached ({current_tokens}/{self.max_tokens}) "
                      f"but can't summarize further without losing recent messages.")
                break
            
            self.summarize_old_context()
            current_tokens = self.get_total_tokens()
            rounds += 1
        
        # Final check and warning
        if current_tokens > self.max_tokens:
            print(f"⚠ Warning: Context ({current_tokens} tokens) exceeds max ({self.max_tokens}). "
                  f"API call may fail or be truncated.")
        
        return self.messages
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        Clear conversation history.
        
        Args:
            keep_system: If True, preserve the system message
        """
        if keep_system and self.messages:
            system_msg = self.messages[0] if self.messages[0].get("role") == "system" else None
            self.messages = [system_msg] if system_msg else []
        else:
            self.messages = []
        
        self.summarization_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory state.
        
        Returns:
            Dictionary with memory statistics
        """
        total_tokens = self.get_total_tokens()
        return {
            "total_messages": len(self.messages),
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": f"{(total_tokens / self.max_tokens * 100):.1f}%",
            "summarization_count": self.summarization_count,
            "recent_messages_kept": self.keep_recent_messages,
            "using_tiktoken": self.use_tiktoken
        }
    
    def __repr__(self) -> str:
        """String representation of MemoryManager state."""
        stats = self.get_stats()
        return (f"MemoryManager(messages={stats['total_messages']}, "
                f"tokens={stats['total_tokens']}/{stats['max_tokens']}, "
                f"summarizations={stats['summarization_count']})")

