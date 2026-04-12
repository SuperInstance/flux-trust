/// Trust scoring with Bayesian updates, temporal decay, and revocation.
///
/// Each agent gets a score in `[0, max_trust]`. Positive observations push
/// the score up, negative observations pull it down. Scores decay over time.
/// Revoked agents are pinned to `-1.0`.

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
    pub fn observe(&mut self, id: u16, positive: bool, cfg: &TrustConfig, now: u64) {
        let entry = self.entries.iter_mut().find(|e| e.agent_id == id);
        match entry {
            Some(e) => {
                if e.revoked {
                    return;
                }
                let delta = if positive { cfg.positive_weight } else { cfg.negative_weight };
                e.score = (e.score + delta).min(cfg.max_trust);
                e.score = e.score.max(0.0);
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
    pub fn decay(&mut self, cfg: &TrustConfig, hours: f64) {
        let factor = (1.0 - cfg.decay_per_hour).powf(hours);
        for e in &mut self.entries {
            if !e.revoked {
                e.score *= factor;
            }
        }
    }

    /// Check if an agent's score meets the trusted threshold.
    pub fn is_trusted(&self, id: u16, cfg: &TrustConfig) -> bool {
        self.score(id) >= cfg.trusted_threshold
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
        self.entries.iter().filter(|e| !e.revoked && e.score >= cfg.trusted_threshold).count()
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
}
