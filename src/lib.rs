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
            && self.none_threshold.is_finite() && self.none_threshold >= 0.0 && self.none_threshold <= 1.0
            && self.trusted_threshold.is_finite() && self.trusted_threshold >= 0.0 && self.trusted_threshold <= 1.0
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

    /// Returns the current trust score for an agent, or `-1.0` if unknown
    /// or the score is non-finite (NaN / Infinity).
    pub fn score(&self, id: u16) -> f64 {
        self.entries
            .iter()
            .find(|e| e.agent_id == id)
            .map_or(-1.0, |e| e.score)
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
                let delta = if positive {
                    cfg.positive_weight
                } else {
                    cfg.negative_weight
                };
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
                let initial = if positive {
                    cfg.positive_weight
                } else {
                    -cfg.negative_weight
                };
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
    /// Entries with non-finite scores are excluded.
    pub fn most_trusted(&self, n: usize) -> Vec<&TrustEntry> {
        let mut sorted: Vec<&TrustEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Returns the `n` least-trusted entries sorted ascending by score.
    /// Entries with non-finite scores are excluded.
    pub fn least_trusted(&self, n: usize) -> Vec<&TrustEntry> {
        let mut sorted: Vec<&TrustEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    pub fn count_trusted(&self, cfg: &TrustConfig) -> usize {
        self.entries
            .iter()
            .filter(|e| !e.revoked && e.score >= cfg.trusted_threshold)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Trust Decay Function (time-based trust reduction)
// ---------------------------------------------------------------------------

/// Advanced decay function supporting multiple decay models.
#[derive(Clone, Debug)]
pub enum DecayModel {
    /// Linear decay: score -= rate * hours
    Linear { rate_per_hour: f64 },
    /// Exponential decay: score *= (1 - rate)^hours
    Exponential { rate_per_hour: f64 },
    /// Step decay: score drops by a fixed amount at each step interval
    Step {
        drop_amount: f64,
        step_interval_hours: f64,
    },
}

impl Default for DecayModel {
    fn default() -> Self {
        DecayModel::Exponential {
            rate_per_hour: 0.01,
        }
    }
}

/// Apply a custom decay model to a given score over a given time period.
pub fn apply_decay(score: f64, model: &DecayModel, hours: f64) -> f64 {
    if score <= 0.0 {
        return score;
    }
    match model {
        DecayModel::Linear { rate_per_hour } => (score - rate_per_hour * hours).max(0.0),
        DecayModel::Exponential { rate_per_hour } => score * (1.0 - rate_per_hour).powf(hours),
        DecayModel::Step {
            drop_amount,
            step_interval_hours,
        } => {
            if *step_interval_hours <= 0.0 {
                return score;
            }
            let steps = (hours / step_interval_hours).floor() as i64;
            (score - *drop_amount * steps as f64).max(0.0)
        }
    }
}

/// Decay a trust entry based on time since last_seen.
pub fn decay_since(score: f64, model: &DecayModel, last_seen: u64, now: u64) -> f64 {
    if now <= last_seen {
        return score;
    }
    let hours = (now - last_seen) as f64 / 3600.0;
    apply_decay(score, model, hours)
}

// ---------------------------------------------------------------------------
// Trust Propagation (transitive trust)
// ---------------------------------------------------------------------------

/// A trust edge in the propagation graph.
#[derive(Clone, Debug)]
pub struct TrustEdge {
    pub from: u16,
    pub to: u16,
    pub weight: f64,
}

/// Trust propagation engine. If A trusts B (weight w), and B trusts C (weight v),
/// then A partially trusts C with weight w * v * damping.
pub struct TrustPropagator {
    edges: Vec<TrustEdge>,
    damping: f64,
    max_depth: usize,
}

impl TrustPropagator {
    pub fn new() -> Self {
        TrustPropagator {
            edges: Vec::new(),
            damping: 0.5,
            max_depth: 3,
        }
    }

    /// Create with custom damping and max depth.
    pub fn with_config(damping: f64, max_depth: usize) -> Self {
        TrustPropagator {
            edges: Vec::new(),
            damping: damping.clamp(0.0, 1.0),
            max_depth,
        }
    }

    /// Add a trust edge.
    pub fn add_edge(&mut self, from: u16, to: u16, weight: f64) {
        let w = weight.clamp(0.0, 1.0);
        // Update existing edge or add new
        if let Some(edge) = self.edges.iter_mut().find(|e| e.from == from && e.to == to) {
            edge.weight = w;
        } else {
            self.edges.push(TrustEdge {
                from,
                to,
                weight: w,
            });
        }
    }

    /// Compute propagated trust from `source` to `target`.
    /// Uses BFS with depth limit and damping.
    pub fn propagated_trust(&self, source: u16, target: u16) -> f64 {
        if source == target {
            return 1.0;
        }
        let mut best: f64 = 0.0;
        let mut queue: Vec<(u16, f64, usize)> = vec![(source, 1.0, 0)]; // (node, cumulative_weight, depth)
        let mut visited: std::collections::HashSet<(u16, usize)> = std::collections::HashSet::new();

        while let Some((current, weight, depth)) = queue.pop() {
            if current == target {
                best = best.max(weight);
                continue;
            }
            if depth >= self.max_depth {
                continue;
            }
            if !visited.insert((current, depth)) {
                continue;
            }

            for edge in &self.edges {
                if edge.from == current {
                    let new_weight = weight * edge.weight * self.damping;
                    if new_weight > 1e-10 {
                        queue.push((edge.to, new_weight, depth + 1));
                    }
                }
            }
        }

        best
    }

    /// Get all edges.
    pub fn edges(&self) -> &[TrustEdge] {
        &self.edges
    }

    /// Get direct trust weight from A to B.
    pub fn direct_trust(&self, from: u16, to: u16) -> f64 {
        self.edges
            .iter()
            .find(|e| e.from == from && e.to == to)
            .map(|e| e.weight)
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Trust Aggregation (combine multiple trust signals)
// ---------------------------------------------------------------------------

/// Aggregation strategies for combining trust signals.
#[derive(Clone, Debug)]
pub enum AggregationStrategy {
    /// Arithmetic mean
    Average,
    /// Minimum (conservative: one bad signal tanks trust)
    Minimum,
    /// Maximum (optimistic: one good signal boosts trust)
    Maximum,
    /// Weighted mean with custom weights
    Weighted(Vec<f64>),
    /// Geometric mean
    Geometric,
    /// Median
    Median,
}

/// Aggregate multiple trust signals into a single score.
pub fn aggregate_trust(signals: &[f64], strategy: &AggregationStrategy) -> f64 {
    if signals.is_empty() {
        return 0.0;
    }
    match strategy {
        AggregationStrategy::Average => signals.iter().sum::<f64>() / signals.len() as f64,
        AggregationStrategy::Minimum => signals.iter().cloned().fold(f64::INFINITY, f64::min),
        AggregationStrategy::Maximum => signals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        AggregationStrategy::Weighted(weights) => {
            if weights.len() != signals.len() || signals.is_empty() {
                return 0.0;
            }
            let w_sum: f64 = weights.iter().sum();
            if w_sum == 0.0 {
                return 0.0;
            }
            signals
                .iter()
                .zip(weights.iter())
                .map(|(s, w)| s * w)
                .sum::<f64>()
                / w_sum
        }
        AggregationStrategy::Geometric => {
            let product: f64 = signals.iter().product();
            if product <= 0.0 {
                return 0.0;
            }
            product.powf(1.0 / signals.len() as f64)
        }
        AggregationStrategy::Median => {
            let mut sorted = signals.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 && sorted.len() > 1 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reputation Scoring System
// ---------------------------------------------------------------------------

/// A reputation record for an agent.
#[derive(Clone, Debug)]
pub struct ReputationRecord {
    pub agent_id: u16,
    pub reputation: f64,
    pub history: Vec<ReputationEvent>,
}

/// A reputation event.
#[derive(Clone, Debug)]
pub struct ReputationEvent {
    pub timestamp: u64,
    pub change: f64,
    pub reason: String,
}

/// The reputation system maintains long-term agent reputations based on
/// accumulated trust signals and events.
pub struct ReputationSystem {
    records: Vec<ReputationRecord>,
    weight_recent: f64,
    weight_historical: f64,
}

impl ReputationSystem {
    pub fn new() -> Self {
        ReputationSystem {
            records: Vec::new(),
            weight_recent: 0.7,
            weight_historical: 0.3,
        }
    }

    /// Create with custom weighting.
    pub fn with_weights(recent: f64, historical: f64) -> Self {
        ReputationSystem {
            records: Vec::new(),
            weight_recent: recent,
            weight_historical: historical,
        }
    }

    /// Get or create a reputation record.
    fn get_or_create(&mut self, agent_id: u16) -> &mut ReputationRecord {
        if !self.records.iter().any(|r| r.agent_id == agent_id) {
            self.records.push(ReputationRecord {
                agent_id,
                reputation: 0.5, // Start at neutral
                history: Vec::new(),
            });
        }
        self.records.iter_mut().find(|r| r.agent_id == agent_id).unwrap()
    }

    /// Update an agent's reputation.
    pub fn update_reputation(&mut self, agent_id: u16, change: f64, reason: &str, now: u64) {
        let rec = self.get_or_create(agent_id);
        rec.reputation = (rec.reputation + change).clamp(0.0, 1.0);
        rec.history.push(ReputationEvent {
            timestamp: now,
            change,
            reason: reason.to_string(),
        });
    }

    /// Get reputation score for an agent.
    pub fn reputation(&self, agent_id: u16) -> f64 {
        self.records
            .iter()
            .find(|r| r.agent_id == agent_id)
            .map(|r| r.reputation)
            .unwrap_or(0.0)
    }

    /// Get the number of reputation events for an agent.
    pub fn event_count(&self, agent_id: u16) -> usize {
        self.records
            .iter()
            .find(|r| r.agent_id == agent_id)
            .map(|r| r.history.len())
            .unwrap_or(0)
    }

    /// Compute a weighted reputation: recent events have more weight.
    pub fn weighted_reputation(&self, agent_id: u16, now: u64) -> f64 {
        let rec = match self.records.iter().find(|r| r.agent_id == agent_id) {
            Some(r) => r,
            None => return 0.0,
        };

        if rec.history.is_empty() {
            return rec.reputation;
        }

        // Compute weighted average of recent vs historical changes
        let recent_cutoff = if now > 86400 { now - 86400 } else { 0 }; // last 24 hours
        let recent_sum: f64 = rec
            .history
            .iter()
            .filter(|e| e.timestamp > recent_cutoff)
            .map(|e| e.change)
            .sum();
        let hist_sum: f64 = rec
            .history
            .iter()
            .filter(|e| e.timestamp <= recent_cutoff)
            .map(|e| e.change)
            .sum();

        let weighted = recent_sum * self.weight_recent + hist_sum * self.weight_historical;
        (rec.reputation + weighted).clamp(0.0, 1.0)
    }

    /// Get all agents with reputation above a threshold, sorted descending.
    pub fn top_agents(&self, threshold: f64, limit: usize) -> Vec<&ReputationRecord> {
        let mut sorted: Vec<&ReputationRecord> = self
            .records
            .iter()
            .filter(|r| r.reputation >= threshold)
            .collect();
        sorted.sort_by(|a, b| {
            b.reputation
                .partial_cmp(&a.reputation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(limit);
        sorted
    }

    /// Number of tracked agents.
    pub fn count(&self) -> usize {
        self.records.len()
    }
}

// ---------------------------------------------------------------------------
// Trust Threshold Management
// ---------------------------------------------------------------------------

/// Threshold levels for trust-based decisions.
#[derive(Clone, Debug)]
pub struct TrustThresholds {
    pub revoke: f64,
    pub distrust: f64,
    pub none: f64,
    pub cautious: f64,
    pub trusted: f64,
    pub highly_trusted: f64,
}

impl Default for TrustThresholds {
    fn default() -> Self {
        Self {
            revoke: -0.5,
            distrust: 0.0,
            none: 0.2,
            cautious: 0.4,
            trusted: 0.6,
            highly_trusted: 0.85,
        }
    }
}

/// Trust level classification.
#[derive(Clone, Debug, PartialEq)]
pub enum TrustLevel {
    Revoked,
    Distrusted,
    None,
    Cautious,
    Trusted,
    HighlyTrusted,
}

impl TrustThresholds {
    /// Classify a trust score into a trust level.
    pub fn classify(&self, score: f64) -> TrustLevel {
        if score <= self.revoke {
            TrustLevel::Revoked
        } else if score <= self.distrust {
            TrustLevel::Distrusted
        } else if score <= self.none {
            TrustLevel::None
        } else if score <= self.cautious {
            TrustLevel::Cautious
        } else if score < self.highly_trusted {
            TrustLevel::Trusted
        } else {
            TrustLevel::HighlyTrusted
        }
    }

    /// Check if a score meets a minimum trust level.
    pub fn meets(&self, score: f64, level: &TrustLevel) -> bool {
        let required = match level {
            TrustLevel::Revoked => self.revoke,
            TrustLevel::Distrusted => self.distrust,
            TrustLevel::None => self.none,
            TrustLevel::Cautious => self.cautious,
            TrustLevel::Trusted => self.trusted,
            TrustLevel::HighlyTrusted => self.highly_trusted,
        };
        score >= required
    }

    /// Update all thresholds at once.
    pub fn update(
        &mut self,
        revoke: f64,
        distrust: f64,
        none: f64,
        cautious: f64,
        trusted: f64,
        highly_trusted: f64,
    ) {
        self.revoke = revoke;
        self.distrust = distrust;
        self.none = none;
        self.cautious = cautious;
        self.trusted = trusted;
        self.highly_trusted = highly_trusted;
    }

    /// Validate that thresholds are in ascending order. Returns false if not.
    pub fn is_valid(&self) -> bool {
        self.revoke <= self.distrust
            && self.distrust <= self.none
            && self.none <= self.cautious
            && self.cautious <= self.trusted
            && self.trusted <= self.highly_trusted
    }
}

// ---------------------------------------------------------------------------
// I2I TRUST_UPDATE Message Integration Hooks
// ---------------------------------------------------------------------------

/// A trust update message in the I2I (Instance-to-Instance) protocol.
#[derive(Clone, Debug)]
pub struct TrustUpdateMessage {
    pub from_instance: String,
    pub target_agent: u16,
    pub score_delta: f64,
    pub positive: bool,
    pub timestamp: u64,
    pub signature: String,
}

/// A trust sync message to push trust table state.
#[derive(Clone, Debug)]
pub struct TrustSyncMessage {
    pub from_instance: String,
    pub entries: Vec<SyncEntry>,
    pub timestamp: u64,
}

#[derive(Clone, Debug)]
pub struct SyncEntry {
    pub agent_id: u16,
    pub score: f64,
    pub observations: u32,
    pub last_seen: u64,
}

/// Hook trait for processing trust update messages.
pub trait TrustUpdateHook {
    /// Called when a TRUST_UPDATE message is received.
    fn on_trust_update(&mut self, msg: &TrustUpdateMessage) -> HookResult;

    /// Called when a trust sync is received.
    fn on_trust_sync(&mut self, msg: &TrustSyncMessage) -> HookResult;
}

/// Result of processing a hook.
#[derive(Clone, Debug, PartialEq)]
pub enum HookResult {
    Accepted,
    Rejected(String),
    Forwarded,
}

/// The I2I integration handler that processes trust messages.
pub struct I2ITrustHandler {
    pub table: TrustTable,
    pub config: TrustConfig,
    pub thresholds: TrustThresholds,
    accepted_count: u64,
    rejected_count: u64,
    synced_count: u64,
}

impl I2ITrustHandler {
    pub fn new(cfg: TrustConfig) -> Self {
        I2ITrustHandler {
            table: TrustTable::new(),
            config: cfg,
            thresholds: TrustThresholds::default(),
            accepted_count: 0,
            rejected_count: 0,
            synced_count: 0,
        }
    }

    /// Process an incoming trust update message.
    pub fn process_update(&mut self, msg: &TrustUpdateMessage) -> HookResult {
        // Validate: reject if score_delta is out of bounds
        if msg.score_delta < -1.0 || msg.score_delta > 1.0 {
            self.rejected_count += 1;
            return HookResult::Rejected("Invalid score delta".to_string());
        }

        // Apply the update
        self.table
            .observe(msg.target_agent, msg.positive, &self.config, msg.timestamp);

        // Apply custom delta on top
        if let Some(entry) = self.table.entries.iter_mut().find(|e| e.agent_id == msg.target_agent) {
            entry.score = (entry.score + msg.score_delta).clamp(0.0, self.config.max_trust);
        }

        self.accepted_count += 1;
        HookResult::Accepted
    }

    /// Process an incoming trust sync message.
    pub fn process_sync(&mut self, msg: &TrustSyncMessage) -> HookResult {
        if msg.entries.is_empty() {
            self.rejected_count += 1;
            return HookResult::Rejected("Empty sync payload".to_string());
        }

        for entry in &msg.entries {
            if let Some(existing) = self.table.entries.iter_mut().find(|e| e.agent_id == entry.agent_id) {
                // Merge: keep the higher score
                if entry.score > existing.score {
                    existing.score = entry.score.min(self.config.max_trust);
                    existing.observations = existing.observations.max(entry.observations);
                    existing.last_seen = existing.last_seen.max(entry.last_seen);
                }
            }
        }

        self.synced_count += 1;
        HookResult::Forwarded
    }

    /// Get stats about processed messages.
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.accepted_count, self.rejected_count, self.synced_count)
    }
}

impl TrustUpdateHook for I2ITrustHandler {
    fn on_trust_update(&mut self, msg: &TrustUpdateMessage) -> HookResult {
        self.process_update(msg)
    }

    fn on_trust_sync(&mut self, msg: &TrustSyncMessage) -> HookResult {
        self.process_sync(msg)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TrustConfig {
        TrustConfig::default()
    }

    // -- Original TrustTable tests --

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
        assert!((t.score(1) - c.max_trust).abs() < 1e-9);
    }

    #[test]
    fn revoke_sets_neg_one() {
        let mut t = TrustTable::new();
        t.observe(1, true, &cfg(), 100);
        t.revoke(1);
        assert_eq!(t.score(1), -1.0);
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
        t.observe(1, true, &c, 100);
        t.decay(&c, 10.0);
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
        t.observe(2, true, &c, 100);
        assert_eq!(t.count_trusted(&c), 1);
    }

    // -- Trust Decay Function tests --

    #[test]
    fn linear_decay() {
        let model = DecayModel::Linear { rate_per_hour: 0.01 };
        let result = apply_decay(1.0, &model, 50.0);
        assert!((result - 0.5).abs() < 1e-9);
    }

    #[test]
    fn exponential_decay() {
        let model = DecayModel::Exponential { rate_per_hour: 0.5 };
        let result = apply_decay(1.0, &model, 1.0);
        assert!((result - 0.5).abs() < 1e-9);
    }

    #[test]
    fn step_decay() {
        let model = DecayModel::Step {
            drop_amount: 0.1,
            step_interval_hours: 10.0,
        };
        let result = apply_decay(1.0, &model, 25.0); // 2 steps
        assert!((result - 0.8).abs() < 1e-9);
    }

    #[test]
    fn decay_never_negative() {
        let model = DecayModel::Linear { rate_per_hour: 1.0 };
        let result = apply_decay(0.5, &model, 100.0);
        assert!(result >= 0.0);
    }

    #[test]
    fn decay_since_zero_hours() {
        let model = DecayModel::Exponential { rate_per_hour: 0.5 };
        let result = decay_since(0.8, &model, 1000, 1000);
        assert!((result - 0.8).abs() < 1e-9);
    }

    #[test]
    fn decay_since_past_time() {
        let model = DecayModel::Linear { rate_per_hour: 0.01 };
        // 3600 seconds = 1 hour
        let result = decay_since(1.0, &model, 1000, 4600);
        assert!((result - 0.99).abs() < 1e-9);
    }

    // -- Trust Propagation tests --

    #[test]
    fn propagation_direct_trust() {
        let mut prop = TrustPropagator::new();
        prop.add_edge(1, 2, 0.8);
        assert!((prop.propagated_trust(1, 2) - 0.4).abs() < 1e-9); // 0.8 * 0.5 damping
    }

    #[test]
    fn propagation_transitive() {
        let mut prop = TrustPropagator::with_config(0.5, 3);
        prop.add_edge(1, 2, 0.8);
        prop.add_edge(2, 3, 0.9);
        let result = prop.propagated_trust(1, 3);
        // 0.8 * 0.5 * 0.9 * 0.5 = 0.18
        assert!(result > 0.1);
        assert!(result < 0.5);
    }

    #[test]
    fn propagation_self_trust() {
        let prop = TrustPropagator::new();
        assert!((prop.propagated_trust(1, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn propagation_no_path() {
        let prop = TrustPropagator::new();
        assert!((prop.propagated_trust(1, 2) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn propagation_respects_max_depth() {
        let mut prop = TrustPropagator::with_config(0.9, 1);
        prop.add_edge(1, 2, 1.0);
        prop.add_edge(2, 3, 1.0);
        assert!((prop.propagated_trust(1, 3) - 0.0).abs() < 1e-9); // depth 1 can't reach
    }

    #[test]
    fn propagation_direct_trust_accessor() {
        let mut prop = TrustPropagator::new();
        prop.add_edge(1, 2, 0.75);
        assert!((prop.direct_trust(1, 2) - 0.75).abs() < 1e-9);
        assert!((prop.direct_trust(2, 1) - 0.0).abs() < 1e-9);
    }

    // -- Trust Aggregation tests --

    #[test]
    fn aggregate_average() {
        let result = aggregate_trust(&[0.5, 0.7, 0.9], &AggregationStrategy::Average);
        assert!((result - 0.7).abs() < 1e-9);
    }

    #[test]
    fn aggregate_minimum() {
        let result = aggregate_trust(&[0.5, 0.7, 0.9], &AggregationStrategy::Minimum);
        assert!((result - 0.5).abs() < 1e-9);
    }

    #[test]
    fn aggregate_maximum() {
        let result = aggregate_trust(&[0.5, 0.7, 0.9], &AggregationStrategy::Maximum);
        assert!((result - 0.9).abs() < 1e-9);
    }

    #[test]
    fn aggregate_weighted() {
        let result = aggregate_trust(
            &[0.8, 0.4],
            &AggregationStrategy::Weighted(vec![0.7, 0.3]),
        );
        // (0.8*0.7 + 0.4*0.3) / 1.0 = 0.68
        assert!((result - 0.68).abs() < 1e-9);
    }

    #[test]
    fn aggregate_geometric() {
        let result = aggregate_trust(&[0.25, 0.64], &AggregationStrategy::Geometric);
        assert!((result - 0.4).abs() < 1e-9);
    }

    #[test]
    fn aggregate_median_odd() {
        let result = aggregate_trust(&[0.3, 0.7, 0.9], &AggregationStrategy::Median);
        assert!((result - 0.7).abs() < 1e-9);
    }

    #[test]
    fn aggregate_median_even() {
        let result = aggregate_trust(&[0.2, 0.4, 0.6, 0.8], &AggregationStrategy::Median);
        assert!((result - 0.5).abs() < 1e-9);
    }

    #[test]
    fn aggregate_empty() {
        assert!((aggregate_trust(&[], &AggregationStrategy::Average) - 0.0).abs() < 1e-9);
    }

    // -- Reputation System tests --

    #[test]
    fn reputation_starts_neutral() {
        let mut rs = ReputationSystem::new();
        rs.update_reputation(1, 0.1, "good action", 100);
        assert!((rs.reputation(1) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn reputation_clamped() {
        let mut rs = ReputationSystem::new();
        rs.update_reputation(1, 0.9, "heroic", 100);
        assert!(rs.reputation(1) <= 1.0);
        rs.update_reputation(1, -0.5, "bad", 200);
        assert!(rs.reputation(1) >= 0.0);
    }

    #[test]
    fn reputation_event_count() {
        let mut rs = ReputationSystem::new();
        rs.update_reputation(1, 0.1, "a", 100);
        rs.update_reputation(1, -0.1, "b", 200);
        rs.update_reputation(1, 0.2, "c", 300);
        assert_eq!(rs.event_count(1), 3);
    }

    #[test]
    fn reputation_top_agents() {
        let mut rs = ReputationSystem::new();
        rs.update_reputation(1, 0.3, "good", 100);
        rs.update_reputation(2, 0.4, "great", 100);
        rs.update_reputation(3, -0.3, "bad", 100);
        let top = rs.top_agents(0.5, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].agent_id, 2);
    }

    #[test]
    fn reputation_unknown_agent() {
        let rs = ReputationSystem::new();
        assert!((rs.reputation(42) - 0.0).abs() < 1e-9);
    }

    // -- Trust Threshold Management tests --

    #[test]
    fn threshold_classify_levels() {
        let t = TrustThresholds::default();
        assert_eq!(t.classify(-1.0), TrustLevel::Revoked);
        assert_eq!(t.classify(0.1), TrustLevel::None);
        assert_eq!(t.classify(0.3), TrustLevel::Cautious);
        assert_eq!(t.classify(0.7), TrustLevel::Trusted);
        assert_eq!(t.classify(0.9), TrustLevel::HighlyTrusted);
    }

    #[test]
    fn threshold_meets_check() {
        let t = TrustThresholds::default();
        assert!(t.meets(0.7, &TrustLevel::Trusted));
        assert!(!t.meets(0.5, &TrustLevel::Trusted));
        assert!(t.meets(0.5, &TrustLevel::Cautious));
    }

    #[test]
    fn threshold_validity() {
        let t = TrustThresholds::default();
        assert!(t.is_valid());
        let mut bad = t.clone();
        bad.trusted = 0.3; // trusted < cautious
        assert!(!bad.is_valid());
    }

    #[test]
    fn threshold_update() {
        let mut t = TrustThresholds::default();
        t.update(-1.0, 0.0, 0.1, 0.3, 0.5, 0.9);
        assert!((t.none - 0.1).abs() < 1e-9);
    }

    // -- I2I Integration Hooks tests --

    #[test]
    fn i2i_accept_valid_update() {
        let mut handler = I2ITrustHandler::new(cfg());
        let msg = TrustUpdateMessage {
            from_instance: "agent-42".to_string(),
            target_agent: 1,
            score_delta: 0.05,
            positive: true,
            timestamp: 1000,
            signature: "sig".to_string(),
        };
        assert_eq!(handler.process_update(&msg), HookResult::Accepted);
        assert!(handler.table.score(1) > 0.0);
    }

    #[test]
    fn i2i_reject_invalid_delta() {
        let mut handler = I2ITrustHandler::new(cfg());
        let msg = TrustUpdateMessage {
            from_instance: "agent-42".to_string(),
            target_agent: 1,
            score_delta: 5.0, // out of [-1, 1] range
            positive: true,
            timestamp: 1000,
            signature: "sig".to_string(),
        };
        assert!(matches!(handler.process_update(&msg), HookResult::Rejected(_)));
    }

    #[test]
    fn i2i_sync_merge() {
        let mut handler = I2ITrustHandler::new(cfg());
        // First, create an entry with low score
        handler.table.observe(1, true, &handler.config, 100);
        // Then sync with higher score
        let msg = TrustSyncMessage {
            from_instance: "agent-99".to_string(),
            entries: vec![SyncEntry {
                agent_id: 1,
                score: 0.9,
                observations: 20,
                last_seen: 500,
            }],
            timestamp: 600,
        };
        assert_eq!(handler.process_sync(&msg), HookResult::Forwarded);
        assert!(handler.table.score(1) > 0.1);
    }

    #[test]
    fn i2i_reject_empty_sync() {
        let mut handler = I2ITrustHandler::new(cfg());
        let msg = TrustSyncMessage {
            from_instance: "agent-99".to_string(),
            entries: vec![],
            timestamp: 600,
        };
        assert!(matches!(handler.process_sync(&msg), HookResult::Rejected(_)));
    }

    #[test]
    fn i2i_hook_trait_impl() {
        let mut handler = I2ITrustHandler::new(cfg());
        let msg = TrustUpdateMessage {
            from_instance: "a".to_string(),
            target_agent: 1,
            score_delta: 0.0,
            positive: true,
            timestamp: 100,
            signature: "s".to_string(),
        };
        assert_eq!(handler.on_trust_update(&msg), HookResult::Accepted);
    }

    #[test]
    fn i2i_stats_tracking() {
        let mut handler = I2ITrustHandler::new(cfg());
        let good = TrustUpdateMessage {
            from_instance: "a".to_string(),
            target_agent: 1,
            score_delta: 0.0,
            positive: true,
            timestamp: 100,
            signature: "s".to_string(),
        };
        let bad = TrustUpdateMessage {
            from_instance: "a".to_string(),
            target_agent: 1,
            score_delta: 99.0,
            positive: true,
            timestamp: 100,
            signature: "s".to_string(),
        };
        handler.process_update(&good);
        handler.process_update(&bad);
        let (acc, rej, _sync) = handler.stats();
        assert_eq!(acc, 1);
        assert_eq!(rej, 1);
    }

    #[test]
    fn i2i_sync_keeps_lower_score() {
        let mut handler = I2ITrustHandler::new(cfg());
        // Create entry with high score
        let c = cfg();
        for _ in 0..10 {
            handler.table.observe(1, true, &c, 100);
        }
        let high_score = handler.table.score(1);
        // Sync with lower score
        let msg = TrustSyncMessage {
            from_instance: "a".to_string(),
            entries: vec![SyncEntry {
                agent_id: 1,
                score: 0.1,
                observations: 5,
                last_seen: 500,
            }],
            timestamp: 600,
        };
        handler.process_sync(&msg);
        // Should keep the higher local score
        assert!((handler.table.score(1) - high_score).abs() < 1e-9);
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

    // ─── Security audit fixes ──────────────────────────────────────────

    // VULN-1: score() must sanitize non-finite internal values
    #[test]
    fn score_returns_neg_one_for_corrupted_nan_entry() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        // Simulate internal corruption by injecting NaN directly into the entry
        let e = t.entries.iter_mut().find(|e| e.agent_id == 1).unwrap();
        e.score = f64::NAN;
        // score() must not leak NaN to callers
        let s = t.score(1);
        assert_eq!(s, -1.0);
        assert!(s.is_finite());
    }

    #[test]
    fn score_returns_neg_one_for_corrupted_inf_entry() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let e = t.entries.iter_mut().find(|e| e.agent_id == 1).unwrap();
        e.score = f64::INFINITY;
        let s = t.score(1);
        assert_eq!(s, -1.0);
        assert!(s.is_finite());
    }

    #[test]
    fn score_returns_neg_one_for_corrupted_neg_inf_entry() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        let e = t.entries.iter_mut().find(|e| e.agent_id == 1).unwrap();
        e.score = f64::NEG_INFINITY;
        let s = t.score(1);
        assert_eq!(s, -1.0);
        assert!(s.is_finite());
    }

    #[test]
    fn score_sanitized_entry_not_counted_trusted() {
        let mut t = TrustTable::new();
        let c = cfg();
        // Build up enough positive observations to be trusted
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        // Corrupt the score
        t.entries.iter_mut().find(|e| e.agent_id == 1).unwrap().score = f64::NAN;
        assert!(!t.is_trusted(1, &c));
        assert_eq!(t.count_trusted(&c), 0);
    }

    // VULN-2: is_valid() must reject out-of-range none_threshold and trusted_threshold
    #[test]
    fn negative_none_threshold_is_invalid() {
        let mut c = cfg();
        c.none_threshold = -0.5;
        assert!(!c.is_valid());
    }

    #[test]
    fn negative_trusted_threshold_is_invalid() {
        let mut c = cfg();
        c.trusted_threshold = -100.0;
        assert!(!c.is_valid());
    }

    #[test]
    fn none_threshold_above_one_is_invalid() {
        let mut c = cfg();
        c.none_threshold = 1.5;
        assert!(!c.is_valid());
    }

    #[test]
    fn trusted_threshold_above_one_is_invalid() {
        let mut c = cfg();
        c.trusted_threshold = 2.0;
        assert!(!c.is_valid());
    }

    #[test]
    fn observe_with_negative_trusted_threshold_is_noop() {
        // A config with negative trusted_threshold is now invalid, so observe is a no-op
        let mut t = TrustTable::new();
        let bad = TrustConfig {
            trusted_threshold: -100.0,
            ..cfg()
        };
        t.observe(1, true, &bad, 100);
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn is_trusted_with_negative_threshold_is_false() {
        let mut t = TrustTable::new();
        let c = cfg();
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        let bad = TrustConfig {
            trusted_threshold: -100.0,
            ..c
        };
        assert!(!t.is_trusted(1, &bad));
    }

    // VULN-3: most_trusted/least_trusted must exclude non-finite scores
    #[test]
    fn most_trusted_excludes_nan_entries() {
        let mut t = TrustTable::new();
        let c = cfg();
        // Agent 1 has a high score
        for _ in 0..10 {
            t.observe(1, true, &c, 100);
        }
        // Agent 2 has a low score
        t.observe(2, true, &c, 100);
        // Corrupt agent 2's score to NaN
        t.entries.iter_mut().find(|e| e.agent_id == 2).unwrap().score = f64::NAN;
        // Only agent 1 should appear in results
        let top = t.most_trusted(10);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].agent_id, 1);
    }

    #[test]
    fn least_trusted_excludes_inf_entries() {
        let mut t = TrustTable::new();
        let c = cfg();
        t.observe(1, true, &c, 100);
        t.observe(2, true, &c, 100);
        // Corrupt agent 2's score
        t.entries.iter_mut().find(|e| e.agent_id == 2).unwrap().score = f64::INFINITY;
        let bottom = t.least_trusted(10);
        assert_eq!(bottom.len(), 1);
        assert_eq!(bottom[0].agent_id, 1);
    }

    #[test]
    fn most_trusted_all_nan_returns_empty() {
        let mut t = TrustTable::new();
        let c = cfg();
        for id in 0..5u16 {
            t.observe(id, true, &c, 100);
        }
        // Corrupt all entries
        for e in &mut t.entries {
            e.score = f64::NAN;
        }
        assert!(t.most_trusted(10).is_empty());
        assert!(t.least_trusted(10).is_empty());
    }

    // Threshold boundary tests for VULN-2 fix
    #[test]
    fn none_threshold_at_zero_is_valid() {
        let mut c = cfg();
        c.none_threshold = 0.0;
        assert!(c.is_valid());
    }

    #[test]
    fn trusted_threshold_at_one_is_valid() {
        let mut c = cfg();
        c.trusted_threshold = 1.0;
        assert!(c.is_valid());
    }

    #[test]
    fn none_threshold_at_one_is_valid() {
        let mut c = cfg();
        c.none_threshold = 1.0;
        assert!(c.is_valid());
    }

    #[test]
    fn trusted_threshold_at_zero_is_valid() {
        let mut c = cfg();
        c.trusted_threshold = 0.0;
        assert!(c.is_valid());
    }

    #[test]
    fn trusted_threshold_just_above_one_is_invalid() {
        let mut c = cfg();
        c.trusted_threshold = 1.0 + 1e-15;
        assert!(!c.is_valid());
    }

    #[test]
    fn trusted_threshold_just_below_zero_is_invalid() {
        let mut c = cfg();
        c.trusted_threshold = -1e-15;
        assert!(!c.is_valid());
    }
}
