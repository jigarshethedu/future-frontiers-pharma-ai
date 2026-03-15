"""
smpc_demo.py
==============
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Demonstrates Secure Multi-Party Computation (SMPC) for pharma AI using
    additive secret sharing — the simplest SMPC primitive underlying production
    systems like the SCALLOP Consortium (35,000 patients, 11 countries, 2,733
    genetic associations identified).

    Shows how multiple pharma companies can jointly compute a model training
    gradient or a summary statistic over pooled sensitive data without any
    party ever seeing another party's raw data.

Dependencies:
    - Python 3.10+
    - numpy >= 1.26.0

GitHub: chapter05/smpc_demo.py
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SecretShare:
    """One party's additive share of a secret value."""
    party_id: str
    share: np.ndarray   # Each element is one share of the original value


def additive_secret_share(secret: np.ndarray, n_parties: int,
                            seed: int = 42) -> list[SecretShare]:
    """
    Splits a secret array into n_parties additive shares over a large prime field.
    Party i holds share_i. Reconstruction: sum(share_i) = secret (mod prime).
    No single party or any (n-1) colluding parties can reconstruct the secret.

    In production (PySyft, MP-SPDZ), secret sharing runs over finite fields
    with cryptographic security proofs. This demo uses real arithmetic for clarity.
    """
    rng = np.random.default_rng(seed)
    shares = []
    # First n-1 parties get random shares
    cumulative = np.zeros_like(secret, dtype=float)
    for i in range(n_parties - 1):
        share_i = rng.uniform(-1e6, 1e6, size=secret.shape)
        shares.append(SecretShare(party_id=f"Party_{i+1}", share=share_i))
        cumulative += share_i
    # Last party's share is the remainder (ensures sum = secret)
    last_share = secret - cumulative
    shares.append(SecretShare(party_id=f"Party_{n_parties}", share=last_share))
    return shares


def smpc_sum(shares: list[SecretShare]) -> np.ndarray:
    """
    Trusted aggregator sums shares to reconstruct the secret.
    In production SMPC, this aggregation runs using cryptographic protocols
    so even the aggregator never sees individual shares.
    """
    result = np.zeros_like(shares[0].share)
    for share in shares:
        result += share.share
    return result


def simulate_federated_gradient_smpc(site_gradients: list[np.ndarray],
                                      n_parties: int = 3) -> np.ndarray:
    """
    Demonstrates SMPC-protected gradient aggregation across pharma sites.
    Each site secret-shares its gradient; the aggregator sees only the sum,
    not any individual site's gradient.

    This is the core mechanism behind the SCALLOP Consortium's privacy guarantee —
    no single consortium member ever observes another member's patient-derived gradient.
    """
    print(f"\nSMPC gradient aggregation: {len(site_gradients)} sites, "
          f"{n_parties}-party sharing per gradient")

    # Each site shares its gradient with n_parties shares
    all_party_sums = [np.zeros_like(site_gradients[0]) for _ in range(n_parties)]

    for site_idx, gradient in enumerate(site_gradients):
        shares = additive_secret_share(gradient, n_parties, seed=site_idx * 17 + 42)
        print(f"  Site {site_idx+1} gradient shared (raw gradient NEVER transmitted).")
        for party_idx, share in enumerate(shares):
            all_party_sums[party_idx] += share.share

    # Aggregator reconstructs by summing all party sums
    final_aggregate = smpc_sum([
        SecretShare(f"Party_{i+1}", s) for i, s in enumerate(all_party_sums)
    ])
    return final_aggregate


if __name__ == "__main__":
    print("=" * 70)
    print("SMPC DEMO — SECURE GRADIENT AGGREGATION FOR PHARMA FL")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("Future Frontiers by Jigar Sheth")
    print("=" * 70)

    # Simulate 4 pharma company sites each computing local gradients
    rng = np.random.default_rng(99)
    n_features = 8
    site_gradients = [
        rng.normal(0, 0.1, size=n_features) for _ in range(4)
    ]
    true_aggregate = sum(site_gradients)

    print("\nTrue aggregate gradient (computed centrally for verification):")
    print(f"  {true_aggregate.round(6)}")

    smpc_aggregate = simulate_federated_gradient_smpc(site_gradients, n_parties=3)

    print(f"\nSMPC-reconstructed aggregate gradient:")
    print(f"  {smpc_aggregate.round(6)}")

    max_error = np.abs(true_aggregate - smpc_aggregate).max()
    print(f"\nMax reconstruction error: {max_error:.2e} (should be ~0)")

    print("\nSecret sharing demo (one gradient element):")
    secret = np.array([3.14159])
    shares = additive_secret_share(secret, n_parties=3)
    for s in shares:
        print(f"  {s.party_id} holds share: {s.share[0]:.6f} (meaningless alone)")
    reconstructed = smpc_sum(shares)
    print(f"  Reconstructed:            {reconstructed[0]:.6f}")
    print(f"  Original:                 {secret[0]:.6f}")
    print(f"  Error:                    {abs(reconstructed[0] - secret[0]):.2e}")
    print("\nModule complete. SCALLOP Consortium reference: Guo et al. 2023.")
