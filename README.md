# flux-trust

> Bayesian trust scoring with temporal decay, revocation, and NaN/Infinity hardening for the FLUX fleet.

## What This Is

`flux-trust` is a Rust crate implementing a **trust table** — each agent gets a score in `[0, max_trust]` that's pushed up by positive observations and pulled down by negative ones. Scores decay over time, and agents can be permanently revoked.

## Role in the FLUX Ecosystem

Trust is the foundation of fleet coordination. `flux-trust` determines who gets access to what:

- **`flux-social`** uses trust scores to weight relationship influence
- **`flux-evolve`** feeds trust as a fitness signal for behavioral evolution
- **`flux-simulator`** models trust dynamics across multi-agent simulations
- **`flux-necropolis`** records peak trust scores on tombstones for posthumous learning
- **`flux-grimoire`** teaches trust-building strategies as part of the curriculum

## Key Features

| Feature | Description |
|---------|-------------|
| **Bayesian Updates** | Positive/negative observations shift scores with configurable weights |
| **Temporal Decay** | `decay(hours)` applies exponential time-based score reduction |
| **Revocation** | `revoke(id)` permanently pins score to -1.0; no future observations accepted |
| **Trusted Threshold** | `is_trusted()` checks against configurable threshold |
| **Rankings** | `most_trusted(n)` / `least_trusted(n)` for access control |
| **NaN/Infinity Hardening** | All operations guard against floating-point poisoning attacks |

## Quick Start

```rust
use flux_trust::{TrustTable, TrustConfig};

let cfg = TrustConfig::default(); // or TrustConfig::new(...)
let mut table = TrustTable::new();

// Record observations
table.observe(1, true, &cfg, 100);  // positive
table.observe(1, true, &cfg, 200);  // positive
table.observe(1, false, &cfg, 300); // negative

println!("Agent 1 trust: {:.2}", table.score(1));     // 0.1
println!("Is trusted: {}", table.is_trusted(1, &cfg)); // false (needs 0.6)

// Build up trust
for _ in 0..6 { table.observe(1, true, &cfg, 400); }
println!("Now trusted: {}", table.is_trusted(1, &cfg)); // true

// Decay over time
table.decay(&cfg, 100.0); // 100 hours pass
println!("After decay: {:.4}", table.score(1));

// Revoke a malicious agent
table.revoke(42);
assert_eq!(table.score(42), -1.0);
```

## Building & Testing

```bash
cargo build
cargo test
```

## Related Fleet Repos

- [`flux-social`](https://github.com/SuperInstance/flux-social) — Social graph with relationship types
- [`flux-evolve`](https://github.com/SuperInstance/flux-evolve) — Evolutionary optimization of trust behaviors
- [`flux-necropolis`](https://github.com/SuperInstance/flux-necropolis) — Posthumous knowledge harvesting
- [`flux-simulator`](https://github.com/SuperInstance/flux-simulator) — Multi-agent fleet simulation
- [`flux-grimoire`](https://github.com/SuperInstance/flux-grimoire) — Trust curriculum for agent onboarding

## License

Part of the [SuperInstance](https://github.com/SuperInstance) FLUX fleet.
