/// Trust scoring with Bayesian updates, temporal decay, and revocation.
///
/// Each agent gets a score in `[0, max_trust]`. Positive observations push
/// the score up, negative observations pull it down. Scores decay over time.
/// Revoked agents are pinned to `-1.0`.
///
/// # Security
///
/// All public methods guard against NaN / Infinity poisoning. If a
/// `TrustConfig` field or an externally-supplied `f64` is non-finite the
/// operation either falls back to safe defaults or is a no-op.

#[derive(Clone, Debug)]
pub struct TrustConfig {
    pub positive_weight: f64,
    pub negative_weight: f64,
    pub max_trust: f64,
    pub decay_per_hour: f64,
    pub none_threshold: f64,
    pub trusted_threshold: f64,
}

impl Default for TrustConfig {
    fn default() -> Self {
        Self {
            positive_weight: 0.1,
            negative_weight: 0.3,
            max_trust: 0.95,
            decay_per_hour: 0.01,
            none_threshold: 0.2,
            trusted_threshold: 0.6,
        }
    }
}

impl TrustConfig {
    /// Build a `TrustConfig` with explicit values, clamping every field to
    /// a safe range and replacing NaN / Infinity with the corresponding
    /// default.
    pub fn new(
        positive_weight: f64,
        negative_weight: f64,
        max_trust: f64,
        decay_per_hour: f64,
        none_threshold: f64,
        trusted_threshold: f64,
    ) -> Self {
        let d = Self::default();
        Self {
            positive_weight: sane(positive_weight, 0.0, 1.0, d.positive_weight),
            negative_weight: sane(negative_weight, 0.0, 1.0, d.negative_weight),
            max_trust: sane(max_trust, 0.01, 1.0, d.max_trust),
            decay_per_hour: sane(decay_per_hour, 0.0, 1.0, d.decay_per_hour),
            none_threshold: sane(none_threshold, 0.0, 1.0, d.none_threshold),
            trusted_threshold: sane(trusted_threshold, 0.0, 1.0, d.trusted_threshold),
        }
    }

    /// Returns `true` when every field is finite and within a reasonable range.
    pub fn is_valid(&self) -> bool {
        self.positive_weight.is_finite() && self.positive_weight >= 0.0
            && self.negative_weight.is_finite() && self.negative_weight >= 0.0
            && self.max_trust.is_finite() && self.max_trust > 0.0 && self.max_trust <= 1.0
            && self.decay_per_hour.is_finite() && self.decay_per_hour >= 0.0
            && self.none_threshold.is_finite()
            && self.trusted_threshold.is_finite()
    }
}

/// Clamp `value` to `[lo, hi]`. If `value` is NaN or Inf, return `fallback`.
#[inline]
fn sane(value: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if !value.is_finite() {
        return fallback;
    }
    value.clamp(lo, hi)
}

#[derive(Clone, Debug)]
pub struct TrustEntry {
    pub agent_id: u16,
    pub score: f64,
    pub positive: u32,
    pub negative: u32,
    pub observations: u32,
    pub revoked: bool,
    pub created: u64,
    pub last_seen: u64,
    pub max_trust: f64,
}

pub struct TrustTable {
    entries: Vec<TrustEntry>,
}

impl TrustTable {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Returns the current trust score for an agent, or `-1.0` if unknown.
    pub fn score(&self, id: u16) -> f64 {
        self.entries.iter().find(|e| e.agent_id == id).map_or(-1.0, |e| e.score)
    }

    /// Record a positive or negative observation and apply a Bayesian-style update.
    ///
    /// If `cfg` contains NaN / Inf values the operation is a no-op to prevent
    /// trust-poisoning.
    pub fn observe(&mut self, id: u16, positive: bool, cfg: &TrustConfig, now: u64) {
        if !cfg.is_valid() {
            return;
        }
        let entry = self.entries.iter_mut().find(|e| e.agent_id == id);
        match entry {
            Some(e) => {
                if e.revoked {
                    return;
                }
                let delta = if positive { cfg.positive_weight } else { -cfg.negative_weight };
                e.score = (e.score + delta).clamp(0.0, cfg.max_trust);
                if positive {
                    e.positive += 1;
                } else {
                    e.negative += 1;
                }
                e.observations += 1;
                e.last_seen = now;
                e.max_trust = cfg.max_trust;
            }
            None => {
                let initial = if positive { cfg.positive_weight } else { -cfg.negative_weight };
                let score = initial.clamp(0.0, cfg.max_trust);
                self.entries.push(TrustEntry {
                    agent_id: id,
                    score,
                    positive: if positive { 1 } else { 0 },
                    negative: if positive { 0 } else { 1 },
                    observations: 1,
                    revoked: false,
                    created: now,
                    last_seen: now,
                    max_trust: cfg.max_trust,
                });
            }
        }
    }

    /// Revoke an agent, pinning their score to `-1.0`.
    pub fn revoke(&mut self, id: u16) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.agent_id == id) {
            e.revoked = true;
            e.score = -1.0;
        }
    }

    /// Apply time-based decay to all non-revoked entries.
    ///
    /// If `hours` is NaN, negative, or infinite the call is a no-op.
    /// If `cfg` is invalid the call is also a no-op.
    pub fn decay(&mut self, cfg: &TrustConfig, hours: f64) {
        if !cfg.is_valid() || !hours.is_finite() || hours < 0.0 {
            return;
        }
        let base = (1.0 - cfg.decay_per_hour).clamp(0.0, 1.0);
        let factor = base.powf(hours);
        for e in &mut self.entries {
            if !e.revoked {
                e.score *= factor;
                // Guard against any residual NaN / Infinity from fp math
                if !e.score.is_finite() {
                    e.score = 0.0;
                }
                e.score = e.score.clamp(0.0, e.max_trust);
            }
        }
    }

    /// Check if an agent's score meets the trusted threshold.
    ///
    /// Returns `false` if the config is invalid (safe default).
    pub fn is_trusted(&self, id: u16, cfg: &TrustConfig) -> bool {
        if !cfg.is_valid() {
            return false;
        }
        let s = self.score(id);
        s.is_finite() && s >= cfg.trusted_threshold
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the `n` most-trusted entries sorted descending by score.
    pub fn most_trusted(&self, n: usize) -> Vec<&TrustEntry> {
        let mut sorted: Vec<&TrustEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Returns the `n` least-trusted entries sorted ascending by score.
    pub fn least_trusted(&self, n: usize) -> Vec<&TrustEntry> {
        let mut sorted: Vec<&TrustEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    pub fn count_trusted(&self, cfg: &TrustConfig) -> usize {
        if !cfg.is_valid() {
            return 0;
        }
        self.entries
            .iter()
            .filter(|e| !e.revoked && e.score.is_finite() && e.score >= cfg.trusted_threshold)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TrustConfig {
        TrustConfig::default()
    }

    #[test]
    fn default_config_values() {
        let c = cfg();
        assert!((c.positive_weight - 0.1).abs() < 1e-9);
        assert!((c.negative_weight - 0.3).abs() < 1e-9);
        assert!((c.max_trust - 0.95).abs() < 1e-9);
        assert!((c.decay_per_hour - 0.01).abs() < 1e-9);
        assert!((c.none_threshold - 0.2).abs() < 1e-9);
        assert!((c.trusted_threshold - 0.6).abs() < 1e-9);
    }

    #[test]
    fn new_table_is_empty() {
        let t = TrustTable::new();
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn unknown_agent_scores_neg_one() {
        let t = TrustTable::new();
        assert_eq!(t.score(42), -1.0);
    }

    #[test]
    fn unknown_agent_not_trusted() {
        let t = TrustTable::new();
        assert!(!t.is_trusted(42, &cfg()));
    }

    #[test]
    fn positive_observation_creates_entry() {
        let mut t = TrustTable::new();
        t.observe(1, true, &cfg(), 100);
        assert_eq!(t.count(), 1);
        assert!((t.score(1) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn negative_observation_clamps_to_zero() {
        let mut t = TrustTable::new();
        t.observe(1, false, &cfg(), 100);
        assert!((t.score(1) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn multiple_positive_observations_accumulate() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        assert!((t.score(1) - c.max_trust).abs() < 1e-9); // 10*0.1 capped at max_trust
    }

    #[test]
    fn score_capped_at_max_trust() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..100 {
            t.observe(1, true, &c, 100);
        }
        assert!(t.score(1) <= c.max_trust + 1e-9);
    }

    #[test]
    fn revoke_sets_neg_one() {
        let mut t = TrustTable::new();
        t.observe(1, true, &cfg(), 100);
        t.revoke(1);
        assert_eq!(t.score(1), -1.0);
        assert!(!t.is_trusted(1, &cfg()));
    }

    #[test]
    fn observe_after_revoke_noop() {
        let mut t = TrustTable::new();
        t.observe(1, true, &cfg(), 100);
        t.revoke(1);
        t.observe(1, true, &cfg(), 200);
        assert_eq!(t.score(1), -1.0);
    }

    #[test]
    fn decay_reduces_score() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100); // score = 0.1
        t.decay(&c, 10.0);
        // 0.1 * 0.99^10 ≈ 0.0904
        assert!(t.score(1) < 0.1);
        assert!(t.score(1) > 0.0);
    }

    #[test]
    fn decay_does_not_affect_revoked() {
        let mut t = TrustTable::new();
        t.observe(1, true, &cfg(), 100);
        t.revoke(1);
        t.decay(&cfg(), 100.0);
        assert_eq!(t.score(1), -1.0);
    }

    #[test]
    fn is_trusted_works() {
        let mut t = TrustTable::new();
        let c = cfg();
        // Need 6+ positive observations to reach 0.6
        for _ in 0..6 {
            t.observe(1, true, &c, 100);
        }
        assert!(t.is_trusted(1, &c));
    }

    #[test]
    fn count_trusted() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..6 {
            t.observe(1, true, &c, 100);
        }
        t.observe(2, true, &c, 100); // only 1 obs → 0.1, not trusted
        assert_eq!(t.count_trusted(&c), 1);
    }

    #[test]
    fn most_trusted_sorts_desc() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        for _ in 0..3 {
            t.observe(2, true, &c, 100);
        }
        let top = t.most_trusted(2);
        assert!(top[0].score >= top[1].score);
    }

    #[test]
    fn least_trusted_sorts_asc() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        for _ in 0..3 {
            t.observe(2, true, &c, 100);
        }
        let bottom = t.least_trusted(2);
        assert!(bottom[0].score <= bottom[1].score);
    }

    #[test]
    fn most_trusted_truncates_to_n() {
        let mut t = TrustTable::new();
        let c = cfg();
        for id in 0..5u16 {
            t.observe(id, true, &c, 100);
        }
        assert_eq!(t.most_trusted(2).len(), 2);
    }

    #[test]
    fn observation_counts() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        t.observe(1, true, &c, 200);
        t.observe(1, false, &c, 300);
        let e = t.entries.iter().find(|e| e.agent_id == 1).unwrap();
        assert_eq!(e.positive, 2);
        assert_eq!(e.negative, 1);
        assert_eq!(e.observations, 3);
    }

    #[test]
    fn last_seen_updated() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        t.observe(1, true, &c, 999);
        let e = t.entries.iter().find(|e| e.agent_id == 1).unwrap();
        assert_eq!(e.last_seen, 999);
    }

    #[test]
    fn negative_then_positive_recover() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, false, &c, 100); // score = 0.0
        t.observe(1, true, &c, 200);  // score = 0.1
        assert!((t.score(1) - 0.1).abs() < 1e-9);
    }

    // ─── NaN / Infinity poisoning tests ─────────────────────────────────

    #[test]
    fn observe_with_nan_positive_weight_is_noop() {
        let mut t = TrustTable::new();
        let bad = TrustConfig {
            positive_weight: f64::NAN,
            ..cfg()
        };
        t.observe(1, true, &bad, 100);
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn observe_with_nan_negative_weight_is_noop() {
        let mut t = TrustTable::new();
        let bad = TrustConfig {
            negative_weight: f64::NAN,
            ..cfg()
        };
        t.observe(1, false, &bad, 100);
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn observe_with_nan_max_trust_is_noop() {
        let mut t = TrustTable::new();
        let bad = TrustConfig {
            max_trust: f64::NAN,
            ..cfg()
        };
        t.observe(1, true, &bad, 100);
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn observe_with_inf_positive_weight_is_noop() {
        let mut t = TrustTable::new();
        let bad = TrustConfig {
            positive_weight: f64::INFINITY,
            ..cfg()
        };
        t.observe(1, true, &bad, 100);
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn observe_with_neg_inf_weight_is_noop() {
        let mut t = TrustTable::new();
        let bad = TrustConfig {
            positive_weight: f64::NEG_INFINITY,
            ..cfg()
        };
        t.observe(1, true, &bad, 100);
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn observe_with_all_nan_config_is_noop() {
        let mut t = TrustTable::new();
        let bad = TrustConfig {
            positive_weight: f64::NAN,
            negative_weight: f64::NAN,
            max_trust: f64::NAN,
            decay_per_hour: f64::NAN,
            none_threshold: f64::NAN,
            trusted_threshold: f64::NAN,
        };
        t.observe(1, true, &bad, 100);
        t.observe(2, false, &bad, 200);
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn observe_existing_entry_with_nan_config_is_noop() {
        let mut t = TrustTable::new();
        let good = cfg();
        t.observe(1, true, &good, 100);
        let score_before = t.score(1);
        let bad = TrustConfig {
            positive_weight: f64::NAN,
            ..good
        };
        t.observe(1, true, &bad, 200);
        assert_eq!(t.score(1), score_before);
    }

    #[test]
    fn decay_with_nan_hours_is_noop() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let score_before = t.score(1);
        t.decay(&c, f64::NAN);
        assert!((t.score(1) - score_before).abs() < 1e-12);
    }

    #[test]
    fn decay_with_inf_hours_is_noop() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let score_before = t.score(1);
        t.decay(&c, f64::INFINITY);
        assert!((t.score(1) - score_before).abs() < 1e-12);
    }

    #[test]
    fn decay_with_neg_inf_hours_is_noop() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let score_before = t.score(1);
        t.decay(&c, f64::NEG_INFINITY);
        assert!((t.score(1) - score_before).abs() < 1e-12);
    }

    #[test]
    fn decay_with_negative_hours_is_noop() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let score_before = t.score(1);
        t.decay(&c, -5.0);
        assert!((t.score(1) - score_before).abs() < 1e-12);
    }

    #[test]
    fn decay_with_nan_decay_per_hour_is_noop() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let score_before = t.score(1);
        let bad = TrustConfig {
            decay_per_hour: f64::NAN,
            ..c
        };
        t.decay(&bad, 10.0);
        assert!((t.score(1) - score_before).abs() < 1e-12);
    }

    #[test]
    fn decay_with_inf_decay_per_hour_is_noop() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let score_before = t.score(1);
        let bad = TrustConfig {
            decay_per_hour: f64::INFINITY,
            ..c
        };
        t.decay(&bad, 10.0);
        assert!((t.score(1) - score_before).abs() < 1e-12);
    }

    #[test]
    fn is_trusted_with_nan_threshold_returns_false() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        let bad = TrustConfig {
            trusted_threshold: f64::NAN,
            ..c
        };
        assert!(!t.is_trusted(1, &bad));
    }

    #[test]
    fn is_trusted_with_inf_threshold_returns_false() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        let bad = TrustConfig {
            trusted_threshold: f64::INFINITY,
            ..c
        };
        assert!(!t.is_trusted(1, &bad));
    }

    #[test]
    fn count_trusted_with_nan_config_returns_zero() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        let bad = TrustConfig {
            trusted_threshold: f64::NAN,
            ..c
        };
        assert_eq!(t.count_trusted(&bad), 0);
    }

    // ─── TrustConfig::new() sanitization tests ─────────────────────────

    #[test]
    fn config_new_clamps_negative_weights() {
        let c = TrustConfig::new(-5.0, -5.0, -1.0, -1.0, -1.0, -1.0);
        // Negative values are clamped to the lower bound of their range, not the default
        assert!((c.positive_weight - 0.0).abs() < 1e-9);
        assert!((c.negative_weight - 0.0).abs() < 1e-9);
        assert!((c.max_trust - 0.01).abs() < 1e-9); // lower bound for max_trust
        assert!((c.decay_per_hour - 0.0).abs() < 1e-9);
        assert!((c.none_threshold - 0.0).abs() < 1e-9);
        assert!((c.trusted_threshold - 0.0).abs() < 1e-9);
    }

    #[test]
    fn config_new_replaces_nan_with_defaults() {
        let c = TrustConfig::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        let d = TrustConfig::default();
        assert!((c.positive_weight - d.positive_weight).abs() < 1e-9);
        assert!((c.negative_weight - d.negative_weight).abs() < 1e-9);
        assert!((c.max_trust - d.max_trust).abs() < 1e-9);
        assert!((c.decay_per_hour - d.decay_per_hour).abs() < 1e-9);
        assert!((c.none_threshold - d.none_threshold).abs() < 1e-9);
        assert!((c.trusted_threshold - d.trusted_threshold).abs() < 1e-9);
    }

    #[test]
    fn config_new_replaces_inf_with_defaults() {
        let c = TrustConfig::new(f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let d = TrustConfig::default();
        assert!((c.positive_weight - d.positive_weight).abs() < 1e-9);
        assert!((c.negative_weight - d.negative_weight).abs() < 1e-9);
    }

    #[test]
    fn config_new_clamps_over_range() {
        let c = TrustConfig::new(2.0, 2.0, 2.0, 2.0, 2.0, 2.0);
        assert!((c.positive_weight - 1.0).abs() < 1e-9);
        assert!((c.negative_weight - 1.0).abs() < 1e-9);
        assert!((c.max_trust - 1.0).abs() < 1e-9);
        assert!((c.decay_per_hour - 1.0).abs() < 1e-9);
        assert!((c.none_threshold - 1.0).abs() < 1e-9);
        assert!((c.trusted_threshold - 1.0).abs() < 1e-9);
    }

    #[test]
    fn config_new_passes_through_valid_values() {
        let c = TrustConfig::new(0.15, 0.25, 0.90, 0.02, 0.3, 0.55);
        assert!((c.positive_weight - 0.15).abs() < 1e-9);
        assert!((c.negative_weight - 0.25).abs() < 1e-9);
        assert!((c.max_trust - 0.90).abs() < 1e-9);
        assert!((c.decay_per_hour - 0.02).abs() < 1e-9);
        assert!((c.none_threshold - 0.3).abs() < 1e-9);
        assert!((c.trusted_threshold - 0.55).abs() < 1e-9);
    }

    // ─── TrustConfig::is_valid() tests ─────────────────────────────────

    #[test]
    fn default_config_is_valid() {
        assert!(cfg().is_valid());
    }

    #[test]
    fn nan_positive_weight_is_invalid() {
        let mut c = cfg();
        c.positive_weight = f64::NAN;
        assert!(!c.is_valid());
    }

    #[test]
    fn nan_any_field_is_invalid() {
        let fields: &[fn(&mut TrustConfig)] = &[
            |c| c.positive_weight = f64::NAN,
            |c| c.negative_weight = f64::NAN,
            |c| c.max_trust = f64::NAN,
            |c| c.decay_per_hour = f64::NAN,
            |c| c.none_threshold = f64::NAN,
            |c| c.trusted_threshold = f64::NAN,
        ];
        for setter in fields {
            let mut c = cfg();
            setter(&mut c);
            assert!(!c.is_valid());
        }
    }

    #[test]
    fn negative_weight_is_invalid() {
        let mut c = cfg();
        c.positive_weight = -0.5;
        assert!(!c.is_valid());
    }

    #[test]
    fn negative_negative_weight_is_invalid() {
        let mut c = cfg();
        c.negative_weight = -0.1;
        assert!(!c.is_valid());
    }

    #[test]
    fn max_trust_zero_is_invalid() {
        let mut c = cfg();
        c.max_trust = 0.0;
        assert!(!c.is_valid());
    }

    #[test]
    fn max_trust_above_one_is_invalid() {
        let mut c = cfg();
        c.max_trust = 1.5;
        assert!(!c.is_valid());
    }

    // ─── sane() helper tests ────────────────────────────────────────────

    #[test]
    fn sane_normal_value() {
        assert!((sane(0.5, 0.0, 1.0, 0.1) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sane_clamps_low() {
        assert!((sane(-0.5, 0.0, 1.0, 0.1) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn sane_clamps_high() {
        assert!((sane(1.5, 0.0, 1.0, 0.1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sane_nan_returns_fallback() {
        assert!((sane(f64::NAN, 0.0, 1.0, 0.42) - 0.42).abs() < 1e-12);
    }

    #[test]
    fn sane_inf_returns_fallback() {
        assert!((sane(f64::INFINITY, 0.0, 1.0, 0.42) - 0.42).abs() < 1e-12);
    }

    #[test]
    fn sane_neg_inf_returns_fallback() {
        assert!((sane(f64::NEG_INFINITY, 0.0, 1.0, 0.42) - 0.42).abs() < 1e-12);
    }

    // ─── Additional functional tests ────────────────────────────────────

    #[test]
    fn revoke_nonexistent_agent_is_noop() {
        let mut t = TrustTable::new();
        t.revoke(999); // should not panic
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn decay_zero_hours_no_change() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let score_before = t.score(1);
        t.decay(&c, 0.0);
        assert!((t.score(1) - score_before).abs() < 1e-12);
    }

    #[test]
    fn decay_large_hours_drains_score() {
        let mut t = TrustTable::new();
        let c = cfg();
        // Build up a high score
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        assert!((t.score(1) - c.max_trust).abs() < 1e-9);
        // Decay for 1000 hours
        t.decay(&c, 1000.0);
        // 0.95 * 0.99^1000 ≈ effectively 0
        assert!(t.score(1) < 0.01);
        assert!(t.score(1) >= 0.0);
    }

    #[test]
    fn decay_with_zero_decay_rate_no_change() {
        let mut t = TrustTable::new();
        let c = TrustConfig {
            decay_per_hour: 0.0,
            ..cfg()
        };
        t.observe(1, true, &c, 100);
        let score_before = t.score(1);
        t.decay(&c, 1000.0);
        assert!((t.score(1) - score_before).abs() < 1e-12);
    }

    #[test]
    fn observe_multiple_agents() {
        let mut t = TrustTable::new();
        let c = cfg();
        for id in 0..10u16 {
            t.observe(id, true, &c, 100);
        }
        assert_eq!(t.count(), 10);
        for id in 0..10u16 {
            assert!(t.score(id) >= 0.0);
        }
    }

    #[test]
    fn score_never_exceeds_max_trust() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..1000 {
            t.observe(1, true, &c, 100);
        }
        assert!(t.score(1) <= c.max_trust + 1e-9);
        assert!(t.score(1) >= 0.0);
    }

    #[test]
    fn score_never_below_zero_after_negative() {
        let mut t = TrustTable::new();
        let c = cfg();
        // First positive to create entry
        t.observe(1, true, &c, 100);
        // Many negatives
        for _ in 0..100 {
            t.observe(1, false, &c, 200);
        }
        assert!(t.score(1) >= 0.0);
        assert!(t.score(1) <= c.max_trust + 1e-9);
    }

    #[test]
    fn decay_clamps_score_after_operation() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        t.observe(1, true, &c, 100);
        t.observe(1, true, &c, 100);
        // After 3 observations: score = 0.3
        t.decay(&c, 1000.0);
        let score = t.score(1);
        assert!(score.is_finite());
        assert!(score >= 0.0);
        assert!(score <= c.max_trust + 1e-9);
    }

    #[test]
    fn mixed_observations_trust_calculation() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        t.observe(1, true, &c, 200);
        t.observe(1, true, &c, 300);
        t.observe(1, false, &c, 400);
        t.observe(1, false, &c, 500);
        let e = t.entries.iter().find(|e| e.agent_id == 1).unwrap();
        assert_eq!(e.positive, 3);
        assert_eq!(e.negative, 2);
        assert_eq!(e.observations, 5);
        // Step by step: 0.1 -> 0.2 -> 0.3 -> 0.0 -> 0.0
        assert!((e.score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn most_trusted_empty_table() {
        let t = TrustTable::new();
        assert!(t.most_trusted(5).is_empty());
    }

    #[test]
    fn least_trusted_empty_table() {
        let t = TrustTable::new();
        assert!(t.least_trusted(5).is_empty());
    }

    #[test]
    fn count_trusted_empty_table() {
        let t = TrustTable::new();
        assert_eq!(t.count_trusted(&cfg()), 0);
    }

    #[test]
    fn created_timestamp_set() {
        let mut t = TrustTable::new();
        t.observe(1, true, &cfg(), 42);
        let e = t.entries.iter().find(|e| e.agent_id == 1).unwrap();
        assert_eq!(e.created, 42);
    }

    #[test]
    fn revoked_agent_not_counted_as_trusted() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        assert!(t.is_trusted(1, &c));
        assert_eq!(t.count_trusted(&c), 1);
        t.revoke(1);
        assert!(!t.is_trusted(1, &c));
        assert_eq!(t.count_trusted(&c), 0);
    }

    #[test]
    fn multiple_revoke_calls_idempotent() {
        let mut t = TrustTable::new();
        t.observe(1, true, &cfg(), 100);
        t.revoke(1);
        t.revoke(1);
        t.revoke(1);
        assert_eq!(t.score(1), -1.0);
    }
}
