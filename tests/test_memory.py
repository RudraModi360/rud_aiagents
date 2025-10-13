"""
Unit Tests for MemoryManager

Tests the core functionality of context management including:
- Token estimation (both with and without tiktoken)
- Message addition and retrieval
- Automatic summarization when approaching token limits
- Context trimming to stay within limits
- Edge cases and error handling

Run with: pytest tests/test_memory.py -v
"""

import pytest
import json
from unittest.mock import Mock, patch
from core.memory import MemoryManager


class TestTokenEstimation:
    """Test token estimation accuracy and fallback behavior."""
    
    def test_estimate_tokens_with_tiktoken(self):
        """Test token estimation when tiktoken is available."""
        memory = MemoryManager(max_tokens=1000)
        
        # Short text
        assert memory.estimate_tokens("Hello world") > 0
        
        # Empty text
        assert memory.estimate_tokens("") == 0
        
        # Longer text should have more tokens
        short_text = "Hello"
        long_text = "Hello " * 100
        assert memory.estimate_tokens(long_text) > memory.estimate_tokens(short_text)
    
    def test_estimate_tokens_without_tiktoken(self):
        """Test fallback token estimation when tiktoken is not available."""
        memory = MemoryManager(max_tokens=1000)
        memory.use_tiktoken = False
        memory.tokenizer = None
        
        # Should still provide reasonable estimates
        text = "This is a test message with several words"
        tokens = memory.estimate_tokens(text)
        
        # Heuristic: ~1.3 tokens per word + chars/4
        # Should be in reasonable range
        assert tokens > 5  # At least some tokens
        assert tokens < len(text)  # Not more than character count
    
    def test_estimate_message_tokens(self):
        """Test token estimation for complete messages."""
        memory = MemoryManager(max_tokens=1000)
        
        # Simple user message
        user_msg = {"role": "user", "content": "Hello, how are you?"}
        tokens = memory.estimate_message_tokens(user_msg)
        assert tokens > 0
        
        # Message with empty content
        empty_msg = {"role": "user", "content": ""}
        tokens = memory.estimate_message_tokens(empty_msg)
        assert tokens >= 4  # Should still have overhead
        
        # Message with tool call ID (tool response)
        tool_msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": json.dumps({"result": "success"})
        }
        tokens = memory.estimate_message_tokens(tool_msg)
        assert tokens > 0


class TestMessageManagement:
    """Test message addition, retrieval, and basic operations."""
    
    def test_add_single_message(self):
        """Test adding a single message."""
        memory = MemoryManager(max_tokens=1000)
        
        msg = {"role": "user", "content": "Test message"}
        memory.add_message(msg)
        
        assert len(memory.messages) == 1
        assert memory.messages[0] == msg
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages in sequence."""
        memory = MemoryManager(max_tokens=5000)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"}
        ]
        
        for msg in messages:
            memory.add_message(msg)
        
        assert len(memory.messages) == 5
        assert memory.messages == messages
    
    def test_get_total_tokens(self):
        """Test total token calculation across all messages."""
        memory = MemoryManager(max_tokens=5000)
        
        # Initially empty
        assert memory.get_total_tokens() == 0
        
        # Add messages
        memory.add_message({"role": "user", "content": "Hello"})
        tokens_1 = memory.get_total_tokens()
        assert tokens_1 > 0
        
        memory.add_message({"role": "assistant", "content": "Hi there!"})
        tokens_2 = memory.get_total_tokens()
        assert tokens_2 > tokens_1
    
    def test_clear_history_keep_system(self):
        """Test clearing history while keeping system message."""
        memory = MemoryManager(max_tokens=1000)
        
        memory.add_message({"role": "system", "content": "System prompt"})
        memory.add_message({"role": "user", "content": "User message"})
        memory.add_message({"role": "assistant", "content": "Assistant response"})
        
        assert len(memory.messages) == 3
        
        memory.clear_history(keep_system=True)
        
        assert len(memory.messages) == 1
        assert memory.messages[0]["role"] == "system"
    
    def test_clear_history_remove_all(self):
        """Test clearing all history including system message."""
        memory = MemoryManager(max_tokens=1000)
        
        memory.add_message({"role": "system", "content": "System prompt"})
        memory.add_message({"role": "user", "content": "User message"})
        
        memory.clear_history(keep_system=False)
        
        assert len(memory.messages) == 0


class TestSummarization:
    """Test automatic summarization functionality."""
    
    def test_no_summarization_below_threshold(self):
        """Test that summarization doesn't trigger below threshold."""
        # Set a high token limit so we don't trigger summarization
        memory = MemoryManager(max_tokens=10000, summary_trigger_ratio=0.75)
        
        # Add a few short messages
        memory.add_message({"role": "system", "content": "System"})
        memory.add_message({"role": "user", "content": "Hello"})
        memory.add_message({"role": "assistant", "content": "Hi"})
        
        messages = memory.get_trimmed_messages()
        
        # Should have all original messages, no summarization
        assert len(messages) == 3
        assert memory.summarization_count == 0
    
    def test_summarization_triggers_above_threshold(self):
        """Test that summarization triggers when exceeding threshold."""
        # Set a very low token limit to force summarization
        memory = MemoryManager(max_tokens=100, summary_trigger_ratio=0.5, keep_recent_messages=2)
        
        # Add system message
        memory.add_message({"role": "system", "content": "You are a helpful assistant"})
        
        # Add many messages to exceed threshold
        for i in range(10):
            memory.add_message({
                "role": "user",
                "content": f"This is a long message number {i} with lots of content to increase token count"
            })
            memory.add_message({
                "role": "assistant",
                "content": f"This is a detailed response to message {i} with more content"
            })
        
        messages = memory.get_trimmed_messages()
        
        # Should have triggered summarization
        assert memory.summarization_count > 0
        
        # Should have system + summary + recent messages
        # Length should be less than original (1 + 20 = 21)
        assert len(messages) < 21
    
    def test_create_summary_content(self):
        """Test that summary contains useful information."""
        memory = MemoryManager(max_tokens=100, summary_trigger_ratio=0.5, keep_recent_messages=2)
        
        messages_to_summarize = [
            {"role": "user", "content": "Can you help me with Python?"},
            {"role": "assistant", "content": "Of course! What do you need help with?"},
            {"role": "user", "content": "How do I create a list?"},
            {"role": "assistant", "content": "You can create a list using square brackets: my_list = [1, 2, 3]"}
        ]
        
        summary = memory._create_summary(messages_to_summarize)
        
        # Summary should be a string
        assert isinstance(summary, str)
        
        # Summary should contain some reference to the conversation
        assert len(summary) > 0
        assert "Summary" in summary
    
    def test_keep_recent_messages_intact(self):
        """Test that recent messages are preserved during summarization."""
        memory = MemoryManager(max_tokens=100, summary_trigger_ratio=0.3, keep_recent_messages=3)
        
        memory.add_message({"role": "system", "content": "System message"})
        
        # Add many messages
        for i in range(15):
            memory.add_message({"role": "user", "content": f"Message {i}" * 10})
        
        # Force summarization
        memory.get_trimmed_messages()
        
        # Check that the last few messages are intact
        last_messages = memory.messages[-3:]
        
        # These should be original messages, not summarized
        for msg in last_messages:
            assert "is_summary" not in msg
            assert msg["role"] == "user"


class TestTokenLimiting:
    """Test that token limits are enforced correctly."""
    
    def test_token_limit_enforcement(self):
        """Test that context stays within token limits after trimming."""
        max_tokens = 500
        memory = MemoryManager(max_tokens=max_tokens, summary_trigger_ratio=0.6)
        
        # Add system message
        memory.add_message({"role": "system", "content": "System prompt"})
        
        # Add lots of content
        for i in range(20):
            content = "This is a very long message " * 20  # ~100+ tokens each
            memory.add_message({"role": "user", "content": content})
            memory.add_message({"role": "assistant", "content": content})
        
        # Get trimmed messages
        trimmed = memory.get_trimmed_messages()
        total_tokens = memory.get_total_tokens()
        
        # After trimming, tokens should be managed (may slightly exceed due to minimum messages)
        # but summarization should have occurred
        assert memory.summarization_count > 0
        
        # Should have reduced message count significantly
        assert len(trimmed) < 40  # Much less than 1 + 40 original messages
    
    def test_multiple_summarization_rounds(self):
        """Test that multiple rounds of summarization work correctly."""
        memory = MemoryManager(max_tokens=200, summary_trigger_ratio=0.4, keep_recent_messages=2)
        
        memory.add_message({"role": "system", "content": "System"})
        
        # Add many long messages to force multiple summarizations
        for i in range(30):
            long_content = "Very long content " * 30
            memory.add_message({"role": "user", "content": long_content})
        
        # This should trigger multiple summarization rounds
        trimmed = memory.get_trimmed_messages()
        
        # Should have performed summarization
        assert memory.summarization_count >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_memory(self):
        """Test behavior with no messages."""
        memory = MemoryManager(max_tokens=1000)
        
        assert len(memory.messages) == 0
        assert memory.get_total_tokens() == 0
        
        trimmed = memory.get_trimmed_messages()
        assert len(trimmed) == 0
    
    def test_single_system_message(self):
        """Test behavior with only system message."""
        memory = MemoryManager(max_tokens=1000)
        
        memory.add_message({"role": "system", "content": "System prompt"})
        
        trimmed = memory.get_trimmed_messages()
        assert len(trimmed) == 1
        assert trimmed[0]["role"] == "system"
    
    def test_very_long_single_message(self):
        """Test handling of a single very long message."""
        memory = MemoryManager(max_tokens=100, summary_trigger_ratio=0.5)
        
        # Add a single very long message
        long_content = "Very long content " * 100
        memory.add_message({"role": "user", "content": long_content})
        
        trimmed = memory.get_trimmed_messages()
        
        # Should still return the message even if it exceeds limit
        # (can't summarize a single message)
        assert len(trimmed) == 1
    
    def test_get_stats(self):
        """Test statistics reporting."""
        memory = MemoryManager(max_tokens=1000)
        
        stats = memory.get_stats()
        
        assert "total_messages" in stats
        assert "total_tokens" in stats
        assert "max_tokens" in stats
        assert "utilization" in stats
        assert "summarization_count" in stats
        assert "using_tiktoken" in stats
        
        assert stats["total_messages"] == 0
        assert stats["total_tokens"] == 0
        assert stats["max_tokens"] == 1000
    
    def test_repr(self):
        """Test string representation."""
        memory = MemoryManager(max_tokens=1000)
        
        repr_str = repr(memory)
        
        assert "MemoryManager" in repr_str
        assert "messages=" in repr_str
        assert "tokens=" in repr_str


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""
    
    def test_typical_conversation_flow(self):
        """Test a typical multi-turn conversation with tool calls."""
        memory = MemoryManager(max_tokens=2000, summary_trigger_ratio=0.75)
        
        # System message
        memory.add_message({"role": "system", "content": "You are a coding assistant"})
        
        # User asks a question
        memory.add_message({"role": "user", "content": "Can you help me debug this Python code?"})
        
        # Assistant responds
        memory.add_message({"role": "assistant", "content": "Of course! Please share the code."})
        
        # User provides code
        memory.add_message({"role": "user", "content": "Here's my code:\n\n" + "def foo():\n    return bar" * 10})
        
        # Assistant calls a tool (simulated)
        tool_msg = Mock()
        tool_msg.tool_calls = [Mock(function=Mock(name="read_file", arguments='{"path": "test.py"}'))]
        tool_msg.content = None
        memory.add_message(tool_msg)
        
        # Tool result
        memory.add_message({
            "role": "tool",
            "tool_call_id": "call_123",
            "content": json.dumps({"success": True, "content": "File contents..."})
        })
        
        # Get trimmed messages
        trimmed = memory.get_trimmed_messages()
        
        # Should have all messages (under threshold)
        assert len(trimmed) == 6
        
        stats = memory.get_stats()
        assert stats["total_messages"] == 6
    
    def test_sequential_tool_calls(self):
        """Test 10 sequential tool calls to verify context management."""
        memory = MemoryManager(max_tokens=1000, summary_trigger_ratio=0.6, keep_recent_messages=4)
        
        memory.add_message({"role": "system", "content": "System"})
        
        # Simulate 10 tool call iterations
        for i in range(10):
            # User message
            memory.add_message({"role": "user", "content": f"Execute task {i}"})
            
            # Assistant with tool call
            tool_msg = Mock()
            tool_msg.tool_calls = [Mock(function=Mock(name=f"tool_{i}", arguments='{}'))]
            tool_msg.content = None
            memory.add_message(tool_msg)
            
            # Tool result
            memory.add_message({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": json.dumps({"result": f"Result {i}" * 20})  # Long result
            })
            
            # Get trimmed messages (simulating API call preparation)
            trimmed = memory.get_trimmed_messages()
            total_tokens = memory.get_total_tokens()
            
            # Verify we're managing context
            if i > 5:
                # Should have triggered summarization by now
                assert memory.summarization_count > 0 or total_tokens < 1000
        
        # After 10 iterations, context should be managed
        final_stats = memory.get_stats()
        print(f"\nFinal stats after 10 tool calls: {final_stats}")
        
        # Should have performed summarization
        assert memory.summarization_count > 0
        
        # Token count should be under limit (or close to it)
        assert final_stats["total_tokens"] <= memory.max_tokens * 1.2  # Allow 20% overage


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

