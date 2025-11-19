from __future__ import annotations

import torch
import torch.nn as nn

from mighty.mighty_models.sac import SACModel


class TestSACModel:
    def test_init(self):
        """Test initialization of SAC model."""
        sac = SACModel(obs_size=8, action_size=3, activation="tanh")

        assert sac.obs_size == 8, "Obs size should be 8"
        assert sac.action_size == 3, "Action size should be 3"
        assert sac.activation == "tanh", "Passed activation should be tanh"
        assert sac.log_std_min == -20, "Default log_std_min should be -20"
        assert sac.log_std_max == 2, "Default log_std_max should be 2"
        assert sac.continuous_action is True, "SAC should always be continuous"

        # Check network structure - updated for feature extractor + head architecture
        assert hasattr(sac, "policy_feature_extractor"), (
            "Should have policy feature extractor"
        )
        assert hasattr(sac, "policy_head"), "Should have policy head"
        assert isinstance(sac.policy_net, nn.Sequential), (
            "Policy network should be Sequential"
        )

        # Check Q-networks
        assert hasattr(sac, "q_feature_extractor1"), "Should have Q1 feature extractor"
        assert hasattr(sac, "q_head1"), "Should have Q1 head"
        assert isinstance(sac.q_net1, nn.Sequential), "Q-network 1 should be Sequential"
        assert isinstance(sac.q_net2, nn.Sequential), "Q-network 2 should be Sequential"

        # Check target networks
        assert isinstance(sac.target_q_net1, nn.Sequential), (
            "Target Q-network 1 should be Sequential"
        )
        assert isinstance(sac.target_q_net2, nn.Sequential), (
            "Target Q-network 2 should be Sequential"
        )
        assert hasattr(sac, "value_function_module"), (
            "Should have value function module wrapper"
        )

        # Check that target networks have gradients disabled
        for param in sac.target_q_feature_extractor1.parameters():
            assert not param.requires_grad, (
                "Target Q1 feature extractor parameters should not require gradients"
            )
        for param in sac.target_q_head1.parameters():
            assert not param.requires_grad, (
                "Target Q1 head parameters should not require gradients"
            )
        for param in sac.target_q_feature_extractor2.parameters():
            assert not param.requires_grad, (
                "Target Q2 feature extractor parameters should not require gradients"
            )
        for param in sac.target_q_head2.parameters():
            assert not param.requires_grad, (
                "Target Q2 head parameters should not require gradients"
            )

        # Check that live networks have gradients enabled
        for param in sac.q_feature_extractor1.parameters():
            assert param.requires_grad, (
                "Q1 feature extractor parameters should require gradients"
            )
        for param in sac.q_head1.parameters():
            assert param.requires_grad, "Q1 head parameters should require gradients"
        for param in sac.q_feature_extractor2.parameters():
            assert param.requires_grad, (
                "Q2 feature extractor parameters should require gradients"
            )
        for param in sac.q_head2.parameters():
            assert param.requires_grad, "Q2 head parameters should require gradients"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        sac = SACModel(
            obs_size=4,
            action_size=2,
            hidden_sizes=[128, 64],
            activation="tanh",
            log_std_min=-10.0,
            log_std_max=1.0,
        )

        assert sac.obs_size == 4, "Custom obs size should be 4"
        assert sac.action_size == 2, "Custom action size should be 2"
        assert sac.hidden_sizes == [128, 64], "Custom hidden sizes should be [128, 64]"
        assert sac.activation == "tanh", "Custom activation should be tanh"
        assert sac.log_std_min == -10.0, "Custom log_std_min should be -10.0"
        assert sac.log_std_max == 1.0, "Custom log_std_max should be 1.0"

    def test_kwargs_configuration(self):
        """Test initialization with kwargs-based configuration."""
        head_kwargs = {"hidden_sizes": [128], "activation": "relu"}
        feature_extractor_kwargs = {
            "obs_shape": 6,
            "activation": "relu",
            "hidden_sizes": [64, 32],
            "n_layers": 2,
        }

        sac = SACModel(
            obs_size=6,
            action_size=4,
            head_kwargs=head_kwargs,
            feature_extractor_kwargs=feature_extractor_kwargs,
        )

        # Test that it works
        dummy_state = torch.rand((5, 6))
        action, z, mean, log_std = sac(dummy_state)

        assert action.shape == (5, 4), "Should work with custom kwargs"
        assert mean.shape == (5, 4), "Mean should work with custom kwargs"

    def test_value_function_module(self):
        """Test the value function module wrapper."""
        sac = SACModel(obs_size=4, action_size=2)
        dummy_state = torch.rand((8, 4))

        # Test that value function module works
        values_module = sac.value_function_module(dummy_state)
        values_direct = sac.forward_value(dummy_state)

        assert torch.allclose(values_module, values_direct), (
            "Value function module should produce same output as forward_value"
        )
        assert values_module.shape == (
            8,
            1,
        ), "Value function module should have correct shape"

    def test_forward_stochastic(self):
        """Test forward pass with stochastic policy."""
        sac = SACModel(obs_size=6, action_size=4, action_low=-2.0, action_high=3.0)
        sac = SACModel(obs_size=6, action_size=4, action_low=-2.0, action_high=3.0)
        dummy_state = torch.rand((10, 6))

        action, z, mean, log_std = sac(dummy_state, deterministic=False)

        # Check shapes
        assert action.shape == (10, 4), "Action should have shape (10, 4)"
        assert z.shape == (10, 4), "Raw action (z) should have shape (10, 4)"
        assert mean.shape == (10, 4), "Mean should have shape (10, 4)"
        assert log_std.shape == (10, 4), "Log_std should have shape (10, 4)"

        # Check that all outputs are finite
        assert torch.all(torch.isfinite(action)), "Actions should be finite"
        assert torch.all(torch.isfinite(z)), "Raw actions should be finite"
        assert torch.all(torch.isfinite(mean)), "Means should be finite"
        assert torch.all(torch.isfinite(log_std)), "Log_stds should be finite"

        # Check action bounds - should be in [action_low, action_high] range
        assert torch.all(action >= -2.0) and torch.all(action <= 3.0), (
            "Actions should be in [-2.0, 3.0] range"
        )

        # Check log_std clamping
        assert torch.all(log_std >= sac.log_std_min), "Log_std should be >= log_std_min"
        assert torch.all(log_std <= sac.log_std_max), "Log_std should be <= log_std_max"

        # Check relationship: raw_action = tanh(z), then scaled to [action_low, action_high]
        raw_action = torch.tanh(z)
        expected_action = raw_action * sac.action_scale + sac.action_bias
        assert torch.allclose(action, expected_action, atol=1e-6), (
            "Action should equal scaled tanh(z)"
        )

    def test_forward_deterministic(self):
        """Test forward pass with deterministic policy."""
        sac = SACModel(obs_size=5, action_size=2)
        dummy_state = torch.rand((8, 5))

        action, z, mean, log_std = sac(dummy_state, deterministic=True)

        # Check shapes
        assert action.shape == (8, 2), "Action should have shape (8, 2)"
        assert z.shape == (8, 2), "Raw action (z) should have shape (8, 2)"
        assert mean.shape == (8, 2), "Mean should have shape (8, 2)"
        assert log_std.shape == (8, 2), "Log_std should have shape (8, 2)"

        # In deterministic mode, z should equal mean
        assert torch.allclose(z, mean), "In deterministic mode, z should equal mean"

        # Action should be scaled tanh(mean)
        raw_action = torch.tanh(mean)
        expected_action = raw_action * sac.action_scale + sac.action_bias
        assert torch.allclose(action, expected_action), (
            "Action should equal scaled tanh(mean) in deterministic mode"
        )

    def test_stochastic_vs_deterministic(self):
        """Test that stochastic and deterministic modes produce different results."""
        sac = SACModel(obs_size=4, action_size=2)
        dummy_state = torch.rand((5, 4))

        # Get stochastic output
        action_stoch, z_stoch, mean_stoch, log_std_stoch = sac(
            dummy_state, deterministic=False
        )

        # Get deterministic output
        action_det, z_det, mean_det, log_std_det = sac(dummy_state, deterministic=True)

        # Mean and log_std should be the same
        assert torch.allclose(mean_stoch, mean_det), "Means should be identical"
        assert torch.allclose(log_std_stoch, log_std_det), (
            "Log_stds should be identical"
        )

        # In deterministic mode, z should equal mean
        assert torch.allclose(z_det, mean_det), "Deterministic z should equal mean"

        # Stochastic z should likely be different from mean (due to noise)
        # Note: There's a tiny chance they could be the same, but extremely unlikely
        assert not torch.allclose(z_stoch, mean_stoch), (
            "Stochastic z should be different from mean"
        )

    def test_policy_log_prob(self):
        """Test policy log probability calculation."""
        sac = SACModel(obs_size=4, action_size=2)
        dummy_state = torch.rand((6, 4))

        action, z, mean, log_std = sac(dummy_state, deterministic=False)
        log_prob = sac.policy_log_prob(z, mean, log_std)

        # Check shape
        assert log_prob.shape == (6, 1), "Log prob should have shape (6, 1)"

        # Check that log probabilities are finite
        assert torch.all(torch.isfinite(log_prob)), "Log probs should be finite"

        # Note: Log probabilities can be positive in some cases for transformed distributions
        # The key constraint is that they should be reasonable values
        # For SAC with tanh transformation, log probs can be positive due to the Jacobian correction
        assert torch.all(log_prob > -50.0), "Log probs should not be extremely negative"
        assert torch.all(log_prob < 50.0), "Log probs should not be extremely positive"

        # Test with deterministic actions (z = mean)
        log_prob_det = sac.policy_log_prob(mean, mean, log_std)
        assert torch.all(torch.isfinite(log_prob_det)), (
            "Deterministic log probs should be finite"
        )

    def test_q_networks(self):
        """Test Q-network forward passes."""
        sac = SACModel(obs_size=4, action_size=2)
        dummy_state = torch.rand((7, 4))
        dummy_action = torch.rand((7, 2)) * 2 - 1  # Actions in [-1, 1] range

        # Concatenate state and action for Q-networks
        state_action = torch.cat([dummy_state, dummy_action], dim=-1)

        q1_value = sac.forward_q1(state_action)
        q2_value = sac.forward_q2(state_action)

        # Check shapes
        assert q1_value.shape == (7, 1), "Q1 values should have shape (7, 1)"
        assert q2_value.shape == (7, 1), "Q2 values should have shape (7, 1)"

        # Check that values are finite
        assert torch.all(torch.isfinite(q1_value)), "Q1 values should be finite"
        assert torch.all(torch.isfinite(q2_value)), "Q2 values should be finite"

    def test_target_networks_initialization(self):
        """Test that target networks are initialized with same weights as live networks."""
        sac = SACModel(obs_size=3, action_size=2)

        # Check that target feature extractors have same weights as live ones
        for p1, p_target1 in zip(
            sac.q_feature_extractor1.parameters(),
            sac.target_q_feature_extractor1.parameters(),
        ):
            assert torch.allclose(p1, p_target1), (
                "Target Q1 feature extractor should have same initial weights"
            )

        for p2, p_target2 in zip(
            sac.q_feature_extractor2.parameters(),
            sac.target_q_feature_extractor2.parameters(),
        ):
            assert torch.allclose(p2, p_target2), (
                "Target Q2 feature extractor should have same initial weights"
            )

        # Check that target heads have same weights as live heads
        for p1, p_target1 in zip(
            sac.q_head1.parameters(), sac.target_q_head1.parameters()
        ):
            assert torch.allclose(p1, p_target1), (
                "Target Q1 head should have same initial weights as Q1 head"
            )

        for p2, p_target2 in zip(
            sac.q_head2.parameters(), sac.target_q_head2.parameters()
        ):
            assert torch.allclose(p2, p_target2), (
                "Target Q2 head should have same initial weights as Q2 head"
            )

    def test_twin_q_networks_independence(self):
        """Test that twin Q-networks are independent."""
        sac = SACModel(obs_size=4, action_size=2)

        # Check that Q-networks have different objects (due to separate creation)
        assert sac.q_feature_extractor1 is not sac.q_feature_extractor2, (
            "Q feature extractors should be separate objects"
        )
        assert sac.q_head1 is not sac.q_head2, "Q heads should be separate objects"
        assert sac.q_net1 is not sac.q_net2, "Q-networks should be separate objects"
        assert sac.target_q_net1 is not sac.target_q_net2, (
            "Target Q-networks should be separate objects"
        )

    def test_log_std_bounds_enforcement(self):
        """Test that log_std bounds are properly enforced."""
        log_std_min = -5.0
        log_std_max = 0.5
        sac = SACModel(
            obs_size=3, action_size=2, log_std_min=log_std_min, log_std_max=log_std_max
        )

        dummy_state = torch.rand((10, 3))
        _, _, _, log_std = sac(dummy_state)

        assert torch.all(log_std >= log_std_min), (
            "Log_std should be >= custom log_std_min"
        )
        assert torch.all(log_std <= log_std_max), (
            "Log_std should be <= custom log_std_max"
        )

    def test_gradient_flow(self):
        """Test that gradients flow properly through networks."""
        sac = SACModel(obs_size=4, action_size=2)
        dummy_state = torch.rand((3, 4))
        dummy_action = torch.rand((3, 2)) * 2 - 1  # Actions in [-1, 1]
        state_action = torch.cat([dummy_state, dummy_action], dim=-1)

        # Test policy network gradients
        action, z, mean, log_std = sac(dummy_state)
        policy_loss = action.mean()  # Dummy loss
        policy_loss.backward(retain_graph=True)

        # Check that policy components have gradients
        policy_feat_has_grad = any(
            p.grad is not None for p in sac.policy_feature_extractor.parameters()
        )
        policy_head_has_grad = any(
            p.grad is not None for p in sac.policy_head.parameters()
        )
        assert policy_feat_has_grad or policy_head_has_grad, (
            "Policy feature extractor or head should have gradients"
        )

        # Test Q-network gradients
        sac.zero_grad()
        q1_value = sac.forward_q1(state_action)
        q_loss = q1_value.mean()  # Dummy loss
        q_loss.backward()

        # Check that Q1 components have gradients
        q1_feat_has_grad = any(
            p.grad is not None for p in sac.q_feature_extractor1.parameters()
        )
        q1_head_has_grad = any(p.grad is not None for p in sac.q_head1.parameters())
        assert q1_feat_has_grad or q1_head_has_grad, (
            "Q1 feature extractor or head should have gradients"
        )

        # Check that target networks don't have gradients
        target_q1_feat_has_grad = any(
            p.grad is not None for p in sac.target_q_feature_extractor1.parameters()
        )
        target_q1_head_has_grad = any(
            p.grad is not None for p in sac.target_q_head1.parameters()
        )
        assert not target_q1_feat_has_grad, (
            "Target Q1 feature extractor should not have gradients"
        )
        assert not target_q1_head_has_grad, "Target Q1 head should not have gradients"

    def test_numerical_stability(self):
        """Test numerical stability of log probability calculation."""
        sac = SACModel(obs_size=2, action_size=1)

        # Test with extreme values
        dummy_state = torch.tensor([[10.0, -10.0], [0.0, 0.0]])
        action, z, mean, log_std = sac(dummy_state, deterministic=False)

        # Test log probability calculation doesn't produce NaN or inf
        log_prob = sac.policy_log_prob(z, mean, log_std)
        assert torch.all(torch.isfinite(log_prob)), (
            "Log probabilities should be finite even with extreme inputs"
        )

        # Test with actions close to boundary values (-1, 1)
        boundary_z = torch.tensor(
            [[5.0], [-5.0], [0.0]]
        )  # These will be close to ±1 after tanh
        boundary_mean = torch.zeros_like(boundary_z)
        boundary_log_std = torch.zeros_like(boundary_z)

        boundary_log_prob = sac.policy_log_prob(
            boundary_z, boundary_mean, boundary_log_std
        )
        assert torch.all(torch.isfinite(boundary_log_prob)), (
            "Log probabilities should be finite for boundary actions"
        )

def test_action_scaling(self):
        """Test that action scaling works correctly."""
        # Test with custom action bounds
        action_low = -2.5
        action_high = 1.5
        sac = SACModel(
            obs_size=3, action_size=2, action_low=action_low, action_high=action_high
        )

        dummy_state = torch.rand((5, 3))
        action, z, mean, log_std = sac(dummy_state)

        # Actions should be within the specified bounds
        assert torch.all(action >= action_low), f"Actions should be >= {action_low}"
        assert torch.all(action <= action_high), f"Actions should be <= {action_high}"

        # Check the scaling math
        raw_action = torch.tanh(z)
        expected_scale = (action_high - action_low) / 2.0
        expected_bias = (action_high + action_low) / 2.0
        expected_action = raw_action * expected_scale + expected_bias

        assert torch.allclose(action, expected_action, atol=1e-6), (
            "Action scaling should match expected formula"
        )
        assert torch.allclose(sac.action_scale, torch.tensor(expected_scale)), (
            "Action scale should be computed correctly"
        )
        assert torch.allclose(sac.action_bias, torch.tensor(expected_bias)), (
            "Action bias should be computed correctly"
        )
