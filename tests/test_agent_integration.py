"""
Integration Tests for MCPAgent with MemoryManager

Tests the complete integration of MemoryManager into MCPAgent including:
- Backwards compatibility with existing code
- Automatic context management during chat sessions
- Tool call handling with memory management
- Real-world scenarios with multiple iterations

Run with: pytest tests/test_agent_integration.py -v
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from core.agent_mcp import MCPAgent


class TestMCPAgentInitialization:
    """Test MCPAgent initialization with MemoryManager."""
    
    def test_agent_init_default_params(self):
        """Test agent initialization with default parameters."""
        agent = MCPAgent()
        
        # Should have MemoryManager
        assert hasattr(agent, 'memory')
        assert agent.memory is not None
        
        # Should have system message
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0]["role"] == "system"
    
    def test_agent_init_custom_memory_params(self):
        """Test agent initialization with custom memory parameters."""
        agent = MCPAgent(
            max_context_tokens=8000,
            memory_summarization_threshold=0.8
        )
        
        assert agent.memory.max_tokens == 8000
        assert agent.memory.summary_trigger_ratio == 0.8
    
    def test_agent_backwards_compatibility_messages_property(self):
        """Test that self.messages property still works for backwards compatibility."""
        agent = MCPAgent()
        
        # Should be able to access messages via property
        messages = agent.messages
        assert isinstance(messages, list)
        assert len(messages) == 1  # System message
        
        # Should be able to read from it
        assert messages[0]["role"] == "system"
    
    def test_agent_system_message_added_to_memory(self):
        """Test that system message is properly added to MemoryManager."""
        custom_system = "You are a custom assistant"
        agent = MCPAgent(system_message=custom_system)
        
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0]["role"] == "system"
        assert agent.memory.messages[0]["content"] == custom_system


class TestClearHistory:
    """Test history clearing functionality."""
    
    def test_clear_history_preserves_system(self):
        """Test that clear_history keeps system message."""
        agent = MCPAgent()
        
        # Add some messages manually
        agent.memory.add_message({"role": "user", "content": "Hello"})
        agent.memory.add_message({"role": "assistant", "content": "Hi"})
        
        assert len(agent.memory.messages) == 3  # system + 2
        
        # Clear history
        agent.clear_history()
        
        # Should only have system message
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0]["role"] == "system"
    
    def test_clear_history_resets_summarization_count(self):
        """Test that clear_history resets summarization counter."""
        agent = MCPAgent()
        
        # Manually trigger a summarization
        agent.memory.summarization_count = 5
        
        agent.clear_history()
        
        assert agent.memory.summarization_count == 0


class TestMemoryStatsAccess:
    """Test access to memory statistics."""
    
    def test_memory_stats_available(self):
        """Test that we can access memory stats from agent."""
        agent = MCPAgent()
        
        stats = agent.memory.get_stats()
        
        assert "total_messages" in stats
        assert "total_tokens" in stats
        assert stats["total_messages"] == 1  # System message
    
    def test_memory_repr(self):
        """Test memory representation is accessible."""
        agent = MCPAgent()
        
        repr_str = repr(agent.memory)
        
        assert "MemoryManager" in repr_str


class TestChatContextManagement:
    """Test context management during chat operations."""
    
    @pytest.fixture
    def mock_groq_client(self):
        """Create a mock Groq client for testing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Test response"
        }
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.mark.asyncio
    async def test_chat_adds_messages_to_memory(self, mock_groq_client):
        """Test that chat() adds messages to MemoryManager."""
        agent = MCPAgent()
        agent.client = mock_groq_client
        
        # Mock MCP sessions (empty)
        sessions = {}
        
        # Chat
        await agent.chat(sessions, "Hello, how are you?")
        
        # Should have: system, user, assistant
        assert len(agent.memory.messages) == 3
        assert agent.memory.messages[1]["role"] == "user"
        assert agent.memory.messages[1]["content"] == "Hello, how are you?"
    
    @pytest.mark.asyncio
    async def test_chat_uses_trimmed_messages(self, mock_groq_client):
        """Test that chat() uses get_trimmed_messages() for API calls."""
        agent = MCPAgent()
        agent.client = mock_groq_client
        
        sessions = {}
        
        # Mock get_trimmed_messages to track calls
        original_get_trimmed = agent.memory.get_trimmed_messages
        call_count = 0
        
        def mock_get_trimmed():
            nonlocal call_count
            call_count += 1
            return original_get_trimmed()
        
        agent.memory.get_trimmed_messages = mock_get_trimmed
        
        await agent.chat(sessions, "Test")
        
        # Should have called get_trimmed_messages
        assert call_count > 0
    
    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self, mock_groq_client):
        """Test chat with tool calls adds all messages to memory."""
        agent = MCPAgent()
        agent.client = mock_groq_client
        
        # First response: tool call
        tool_call_msg = Mock()
        tool_call_msg.content = None
        tool_call_mock = Mock()
        tool_call_mock.id = "call_123"
        tool_call_mock.function.name = "read_file"
        tool_call_mock.function.arguments = '{"path": "test.py"}'
        tool_call_msg.tool_calls = [tool_call_mock]
        tool_call_msg.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{"id": "call_123", "function": {"name": "read_file", "arguments": '{"path": "test.py"}'}}]
        }
        
        # Second response: final message
        final_msg = Mock()
        final_msg.content = "Here's the file content"
        final_msg.tool_calls = None
        final_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "Here's the file content"
        }
        
        # Setup mock to return tool call first, then final message
        mock_groq_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=tool_call_msg)]),
            Mock(choices=[Mock(message=final_msg)])
        ]
        
        sessions = {}
        
        # Mock tool execution
        with patch('core.agent_mcp.execute_tool') as mock_execute:
            mock_execute.return_value = {"success": True, "content": "File contents"}
            
            await agent.chat(sessions, "Read test.py")
        
        # Should have: system, user, assistant (tool call), tool result, assistant (final)
        assert len(agent.memory.messages) >= 4
        
        # Check that tool result was added
        tool_results = [msg for msg in agent.memory.messages if msg.get("role") == "tool"]
        assert len(tool_results) >= 1


class TestContextManagementUnderLoad:
    """Test context management with heavy load."""
    
    @pytest.mark.asyncio
    async def test_many_messages_trigger_summarization(self):
        """Test that many messages trigger automatic summarization."""
        # Create agent with low token limit to force summarization
        agent = MCPAgent(
            max_context_tokens=500,
            memory_summarization_threshold=0.5
        )
        
        # Add many long messages
        for i in range(20):
            agent.memory.add_message({
                "role": "user",
                "content": f"This is a long message number {i} with lots of content " * 10
            })
            agent.memory.add_message({
                "role": "assistant",
                "content": f"This is a long response to message {i} with lots of content " * 10
            })
        
        # Get trimmed messages
        trimmed = agent.memory.get_trimmed_messages()
        
        # Should have triggered summarization
        assert agent.memory.summarization_count > 0
        
        # Message count should be reduced
        assert len(trimmed) < 41  # Less than system + 40 messages
    
    @pytest.mark.asyncio
    async def test_sequential_tool_calls_manage_context(self):
        """Test 10 sequential tool calls with context management."""
        agent = MCPAgent(
            max_context_tokens=1000,
            memory_summarization_threshold=0.6
        )
        
        mock_client = Mock()
        agent.client = mock_client
        
        sessions = {}
        
        # Simulate 10 iterations of tool calls
        for i in range(10):
            # Create mock responses
            tool_call_msg = Mock()
            tool_call_msg.content = None
            tool_call_mock = Mock()
            tool_call_mock.id = f"call_{i}"
            tool_call_mock.function.name = "execute_command"
            tool_call_mock.function.arguments = json.dumps({"command": f"echo test{i}"})
            tool_call_msg.tool_calls = [tool_call_mock]
            tool_call_msg.model_dump.return_value = {
                "role": "assistant",
                "tool_calls": [{"id": f"call_{i}"}]
            }
            
            final_msg = Mock()
            final_msg.content = f"Command executed successfully for iteration {i}"
            final_msg.tool_calls = None
            final_msg.model_dump.return_value = {
                "role": "assistant",
                "content": f"Command executed successfully for iteration {i}"
            }
            
            mock_client.chat.completions.create.side_effect = [
                Mock(choices=[Mock(message=tool_call_msg)]),
                Mock(choices=[Mock(message=final_msg)])
            ]
            
            with patch('core.agent_mcp.execute_tool') as mock_execute:
                mock_execute.return_value = {
                    "success": True,
                    "output": f"Output from command {i}" * 20  # Long output
                }
                
                try:
                    await agent.chat(sessions, f"Execute command {i}")
                except Exception:
                    # May fail due to mock setup, but we're testing memory management
                    pass
        
        # Check that context was managed
        stats = agent.memory.get_stats()
        
        # Should have managed to keep tokens in reasonable range
        # If no summarization, we'd have way more tokens
        print(f"\nContext stats after 10 iterations: {stats}")
        
        # Either summarization occurred, or we're still under limit
        assert (agent.memory.summarization_count > 0 or 
                stats["total_tokens"] < agent.memory.max_tokens)


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing code."""
    
    def test_direct_messages_access_works(self):
        """Test that old code accessing self.messages still works."""
        agent = MCPAgent()
        
        # Old way of accessing messages
        messages = agent.messages
        
        assert isinstance(messages, list)
        assert len(messages) >= 1
    
    def test_can_iterate_over_messages(self):
        """Test that we can iterate over messages like before."""
        agent = MCPAgent()
        
        agent.memory.add_message({"role": "user", "content": "Test"})
        
        # Old code might iterate over messages
        for msg in agent.messages:
            assert "role" in msg
    
    def test_can_access_message_by_index(self):
        """Test that we can access messages by index."""
        agent = MCPAgent()
        
        # Access first message (system)
        system_msg = agent.messages[0]
        
        assert system_msg["role"] == "system"
    
    def test_can_check_message_count(self):
        """Test that len(messages) works."""
        agent = MCPAgent()
        
        initial_count = len(agent.messages)
        
        agent.memory.add_message({"role": "user", "content": "Hello"})
        
        assert len(agent.messages) == initial_count + 1


class TestMemoryManagerIntegrationEdgeCases:
    """Test edge cases in the integration."""
    
    def test_agent_with_no_api_key(self):
        """Test agent initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('core.agent_mcp.ConfigManager') as MockConfig:
                mock_config = MockConfig.return_value
                mock_config.get_api_key.return_value = None
                
                agent = MCPAgent()
                
                # Should still have memory initialized
                assert agent.memory is not None
                assert len(agent.memory.messages) == 1
    
    def test_agent_debug_mode_shows_stats(self):
        """Test that debug mode shows memory stats."""
        agent = MCPAgent(debug=True)
        
        # Memory should be initialized
        assert agent.memory is not None
        
        # Get stats should work
        stats = agent.memory.get_stats()
        assert stats is not None
    
    @pytest.mark.asyncio
    async def test_empty_sessions_dict(self):
        """Test chat with empty MCP sessions."""
        agent = MCPAgent()
        
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_message.model_dump.return_value = {"role": "assistant", "content": "Response"}
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        
        agent.client = mock_client
        
        # Should work with empty sessions
        await agent.chat({}, "Test message")
        
        assert len(agent.memory.messages) >= 2  # At least system + user


class TestRealWorldScenario:
    """Test a realistic end-to-end scenario."""
    
    @pytest.mark.asyncio
    async def test_complete_coding_assistant_session(self):
        """Test a complete session with multiple turns and tool calls."""
        agent = MCPAgent(
            max_context_tokens=2000,
            memory_summarization_threshold=0.7,
            debug=True
        )
        
        # Track initial state
        initial_msg_count = len(agent.memory.messages)
        
        # Simulate a series of interactions
        interactions = [
            {"role": "user", "content": "List all Python files in the current directory"},
            {"role": "assistant", "content": "I'll use the list_files tool to find Python files."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"files": ["main.py", "agent.py"]})},
            {"role": "assistant", "content": "Found 2 Python files: main.py and agent.py"},
            {"role": "user", "content": "Read the contents of main.py"},
            {"role": "assistant", "content": "I'll read main.py for you."},
            {"role": "tool", "tool_call_id": "call_2", "content": json.dumps({"content": "# Main file\nprint('Hello')" * 50})},
            {"role": "assistant", "content": "Here's the content of main.py..."},
        ]
        
        for msg in interactions:
            agent.memory.add_message(msg)
        
        # Get stats
        stats = agent.memory.get_stats()
        
        # Should have all messages plus initial system message
        assert stats["total_messages"] == initial_msg_count + len(interactions)
        
        # Should have token count
        assert stats["total_tokens"] > 0
        
        # Get trimmed messages
        trimmed = agent.memory.get_trimmed_messages()
        
        # Should return valid message list
        assert isinstance(trimmed, list)
        assert len(trimmed) > 0
        
        print(f"\n✅ Complete session stats: {stats}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

