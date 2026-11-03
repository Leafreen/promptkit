'''Agent 重构测试：内部 Session、history 兼容接口、fork()。'''
from .fake_client import make_agent


def test_agent_holds_session_and_ask_appends():
    agent = make_agent(["回复"])
    assert agent.session.length == 0
    assert agent.ask("你好") == "回复"
    assert [m.content for m in agent.session.messages] == ["你好", "回复"]


def test_history_view_supports_index_len_iteration_and_equality():
    agent = make_agent([])
    agent.history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    assert len(agent.history) == 2
    assert agent.history[0]["content"] == "a"
    assert [m["content"] for m in agent.history] == ["a", "b"]
    assert agent.history == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    assert list(agent.history) == agent.get_history()


def test_history_append_is_visible_to_session_and_rebuild():
    agent = make_agent([])
    agent.history.append({"role": "user", "content": "旧"})
    assert agent.get_history() == [{"role": "user", "content": "旧"}]
    agent.rebuild()
    assert agent.messages[-1]["content"] == "旧"


def test_get_history_returns_copy_and_keeps_extra_keys():
    agent = make_agent([])
    agent.history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "search"}}],
        }
    ]
    history = agent.get_history()
    assert history[0]["tool_calls"][0]["id"] == "c1"
    history.clear()
    assert len(agent.history) == 1


def test_reset_clears_history():
    agent = make_agent([])
    agent.history = [{"role": "user", "content": "旧"}]
    agent.reset()
    assert agent.history == []
    assert agent.get_history() == []


def test_load_history_replace_and_append():
    agent = make_agent([])
    agent.history = [{"role": "user", "content": "旧"}]
    assert agent.load_history([{"role": "user", "content": "新"}]) == 1
    assert [m["content"] for m in agent.history] == ["新"]

    assert agent.load_history([{"role": "assistant", "content": "尾"}], replace=False) == 1
    assert [m["content"] for m in agent.history] == ["新", "尾"]


def test_fork_shares_client_but_has_independent_session():
    agent = make_agent(["父回复", "子回复"])
    agent.ask("父问题")
    child = agent.fork()

    assert child.client is agent.client
    assert child.session is not agent.session
    assert child.session.session_id != agent.session.session_id
    assert [m.content for m in child.session.messages] == ["父问题", "父回复"]

    child.ask("子问题")
    assert [m.content for m in agent.session.messages] == ["父问题", "父回复"]
    assert [m.content for m in child.session.messages] == ["父问题", "父回复", "子问题", "子回复"]


def test_fork_does_not_change_default_agent():
    from promptkit.agent import Agent

    agent = make_agent([])
    Agent.set_default(agent)
    child = agent.fork()
    assert Agent.default() is agent
    assert child.messages == [*child.start_prompts, *child.get_history()]
