from src.env import Relation, TemporalGame


class TestTemporalGame:
    def test_temporal_game_initialization(self):
        game = TemporalGame()
        assert game._data is not None
        assert game._doc is None
        assert game._context is None
        assert game._entity_pairs is None
        assert game._timeline is None

    def test_temporal_game_reset(self):
        game = TemporalGame()
        state, info = game.reset()

        assert isinstance(state, dict)
        assert "context" in state
        assert "entity_pairs" in state
        assert "timeline" in state

        assert isinstance(info, dict)
        assert "id" in info

        assert game._doc is not None
        assert game._context is not None
        assert game._entity_pairs is not None
        assert game._timeline is not None

    def test_temporal_game_step_valid_action(self):
        game = TemporalGame()
        state, _ = game.reset()

        entity_pair = state["entity_pairs"][0]

        action = Relation(
            source=entity_pair["source"],
            target=entity_pair["target"],
            type="<",
        )

        state, reward, terminated, truncated, info = game.step(action)

        assert isinstance(state, dict)
        assert isinstance(reward, int)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        assert not terminated
        assert not truncated
        assert reward > 0

    def test_temporal_game_termination(self):
        game = TemporalGame()
        state, _ = game.reset(42)

        action1 = Relation(source="start t0", target="start t26", type="<")
        action2 = Relation(source="start t26", target="start ei143", type="<")
        action3 = Relation(source="start t0", target="start ei143", type=">")

        for action in [action1, action2, action3]:
            state, reward, terminated, truncated, info = game.step(action)

        assert terminated

    def test_temporal_game_step_entity_pair_removal(self):
        game = TemporalGame()
        state, _ = game.reset()

        initial_entity_pairs_count = len(state["entity_pairs"])
        entity_pair = state["entity_pairs"][0]

        action = Relation(
            source=entity_pair["source"],
            target=entity_pair["target"],
            type="<",
        )
        state, reward, terminated, truncated, info = game.step(action)

        assert len(state["entity_pairs"]) == initial_entity_pairs_count - 1
