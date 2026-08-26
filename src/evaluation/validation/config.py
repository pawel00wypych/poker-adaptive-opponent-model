from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrityRequirements:
    min_seeds_per_matchup: int = 2
    min_evaluation_replicates_per_matchup: int = 2
    expected_model_seeds: tuple[int, ...] | None = None
    expected_games_per_matchup: int | None = None
    expected_evaluation_replicates: int | None = None
    require_manifest: bool = False
    enforce_frozen_final_protocol: bool = False


@dataclass(frozen=True)
class DiagnosticThresholds:
    min_adaptive_delta_vs_rule_based_bb: float = 0.0
    max_oracle_underperformance_bb: float = 1.0
    min_tight_win_rate: float = 95.0
    min_tight_mean_profit_bb: float = 15.0
    min_classifier_accuracy: float = 80.0
    min_classifier_coverage: float = 80.0
    max_std_across_seeds_bb: float = 5.0
    extreme_bb_per_100_threshold: float = 300.0
    low_mean_hands_played_threshold: float = 5.0
    always_raise_adaptive_warning_gap_bb: float = 3.0
    high_always_raise_mean_profit_bb: float = 18.0
    high_always_raise_win_rate: float = 95.0
    min_head_to_head_mean_profit_bb: float = 0.0
    max_adaptive_underperformance_vs_general_bb: float = 1.0
    always_raise_stress_loss_bb: float = -15.0
    always_raise_stress_bust_rate: float = 80.0
    min_generalization_positive_variants: int = 3
    min_generalization_adaptive_beats_general_variants: int = 3
    min_generalization_adaptive_beats_rule_based_variants: int = 3
    max_generalization_oracle_gap_bb: float = 3.0
    generalization_extreme_aggressive_min_profit_bb: float = -5.0
    generalization_extreme_aggressive_max_bust_rate: float = 85.0
    max_std_across_evaluation_replicates_bb: float = 5.0
    max_baseline_mirror_abs_profit_bb: float = 1.0
    max_baseline_pair_sum_abs_profit_bb: float = 2.0
    max_cross_play_pair_sum_abs_profit_bb: float = 2.0


@dataclass(frozen=True)
class ValidationThresholds:
    """Compatibility configuration with typed integrity/diagnostic views."""

    min_adaptive_delta_vs_rule_based_bb: float = 0.0
    max_oracle_underperformance_bb: float = 1.0
    min_tight_win_rate: float = 95.0
    min_tight_mean_profit_bb: float = 15.0
    min_classifier_accuracy: float = 80.0
    min_classifier_coverage: float = 80.0
    max_std_across_seeds_bb: float = 5.0
    extreme_bb_per_100_threshold: float = 300.0
    low_mean_hands_played_threshold: float = 5.0
    always_raise_adaptive_warning_gap_bb: float = 3.0
    high_always_raise_mean_profit_bb: float = 18.0
    high_always_raise_win_rate: float = 95.0
    min_head_to_head_mean_profit_bb: float = 0.0
    max_adaptive_underperformance_vs_general_bb: float = 1.0
    always_raise_stress_loss_bb: float = -15.0
    always_raise_stress_bust_rate: float = 80.0
    min_generalization_positive_variants: int = 3
    min_generalization_adaptive_beats_general_variants: int = 3
    min_generalization_adaptive_beats_rule_based_variants: int = 3
    max_generalization_oracle_gap_bb: float = 3.0
    generalization_extreme_aggressive_min_profit_bb: float = -5.0
    generalization_extreme_aggressive_max_bust_rate: float = 85.0
    min_seeds_per_matchup: int = 2
    min_evaluation_replicates_per_matchup: int = 2
    max_std_across_evaluation_replicates_bb: float = 5.0
    max_baseline_mirror_abs_profit_bb: float = 1.0
    max_baseline_pair_sum_abs_profit_bb: float = 2.0
    max_cross_play_pair_sum_abs_profit_bb: float = 2.0
    expected_model_seeds: tuple[int, ...] | None = None
    expected_games_per_matchup: int | None = None
    expected_evaluation_replicates: int | None = None
    require_manifest: bool = False
    enforce_frozen_final_protocol: bool = False

    @property
    def integrity_requirements(self) -> IntegrityRequirements:
        from src.experiment_protocol import FINAL_EXPERIMENT_CONFIG

        frozen = self.enforce_frozen_final_protocol
        return IntegrityRequirements(
            min_seeds_per_matchup=(
                len(FINAL_EXPERIMENT_CONFIG.training.seeds)
                if frozen
                else self.min_seeds_per_matchup
            ),
            min_evaluation_replicates_per_matchup=(
                self.min_evaluation_replicates_per_matchup
            ),
            expected_model_seeds=(
                FINAL_EXPERIMENT_CONFIG.training.seeds
                if frozen
                else self.expected_model_seeds
            ),
            expected_games_per_matchup=(
                FINAL_EXPERIMENT_CONFIG.evaluation.games_per_matchup
                if frozen
                else self.expected_games_per_matchup
            ),
            expected_evaluation_replicates=(
                FINAL_EXPERIMENT_CONFIG.evaluation.baseline_evaluation_replicates
                if frozen
                else self.expected_evaluation_replicates
            ),
            require_manifest=self.require_manifest or frozen,
            enforce_frozen_final_protocol=frozen,
        )

    @property
    def diagnostic_thresholds(self) -> DiagnosticThresholds:
        integrity_fields = {
            "min_seeds_per_matchup",
            "min_evaluation_replicates_per_matchup",
            "expected_model_seeds",
            "expected_games_per_matchup",
            "expected_evaluation_replicates",
            "require_manifest",
            "enforce_frozen_final_protocol",
        }
        return DiagnosticThresholds(
            **{
                name: getattr(self, name)
                for name in DiagnosticThresholds.__dataclass_fields__
                if name not in integrity_fields
            }
        )
