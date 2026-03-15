"""
consent_ledger_stub.py
========================
Chapter 5: Building a Privacy-First AI Strategy
Book: Future Frontiers: Harnessing AI, Safeguarding Privacy, and Shaping Ethics
      in Pharma and Healthcare
Author: Jigar Sheth

Purpose:
    Implements the Dynamic Consent Architecture [ORIGINAL FRAMEWORK] as a
    consent ledger stub. Supports per-use-case granular consent, cryptographic
    audit trail, and withdrawal with verifiable proof — the three properties
    required for GDPR Art. 9(2)(a) compliance in an AI training context.

    This is a stub implementation without blockchain infrastructure.
    For smart-contract integration (Web3), see chapter08/blockchain_consent_ledger.py.

Dependencies:
    - Python 3.10+
    - hashlib (stdlib)
    - json (stdlib)

GitHub: chapter05/consent_ledger_stub.py
"""

from __future__ import annotations
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ConsentRecord:
    """One consent grant or withdrawal for a specific patient and use case."""
    record_id: str
    patient_pseudonym: str       # Hash of patient ID — never store raw ID
    use_case_id: str             # e.g., "AI_DILI_TRAINING", "PV_SIGNAL_DETECTION"
    use_case_description: str
    action: str                  # "GRANT" or "WITHDRAW"
    timestamp: str
    expiry_date: Optional[str]   # ISO date or None (no expiry)
    version: str                 # Consent form version signed
    previous_record_hash: str    # Hash of previous record (chain)
    record_hash: str = ""        # Computed on creation


@dataclass
class ConsentLedger:
    """
    In-memory consent ledger with append-only semantics and cryptographic chaining.
    Each record hashes its content + the previous record's hash, forming a chain
    that makes retroactive modification detectable.
    """
    ledger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    records: list[ConsentRecord] = field(default_factory=list)
    use_case_registry: dict[str, str] = field(default_factory=dict)

    def register_use_case(self, use_case_id: str, description: str) -> None:
        """Register a permitted AI use case before patients can consent to it."""
        self.use_case_registry[use_case_id] = description
        print(f"[LEDGER] Use case registered: {use_case_id}")

    def _hash_patient_id(self, patient_id: str) -> str:
        """One-way hash of patient identifier for pseudonymization."""
        return hashlib.sha256(f"pharma-ai-pseudonym-salt-{patient_id}".encode()).hexdigest()[:16]

    def _compute_record_hash(self, record: ConsentRecord) -> str:
        """Cryptographic hash of record content for tamper detection."""
        payload = json.dumps({
            "record_id": record.record_id,
            "patient_pseudonym": record.patient_pseudonym,
            "use_case_id": record.use_case_id,
            "action": record.action,
            "timestamp": record.timestamp,
            "version": record.version,
            "previous_record_hash": record.previous_record_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_last_hash(self) -> str:
        """Returns the hash of the most recent record (genesis hash if empty)."""
        if not self.records:
            return hashlib.sha256(b"genesis").hexdigest()
        return self.records[-1].record_hash

    def grant_consent(self, patient_id: str, use_case_id: str,
                      version: str = "v1.0", expiry_date: Optional[str] = None) -> ConsentRecord:
        """Records a patient's consent grant for a specific use case."""
        if use_case_id not in self.use_case_registry:
            raise ValueError(f"Use case '{use_case_id}' not registered in ledger.")

        record = ConsentRecord(
            record_id=str(uuid.uuid4()),
            patient_pseudonym=self._hash_patient_id(patient_id),
            use_case_id=use_case_id,
            use_case_description=self.use_case_registry[use_case_id],
            action="GRANT",
            timestamp=datetime.utcnow().isoformat() + "Z",
            expiry_date=expiry_date,
            version=version,
            previous_record_hash=self._get_last_hash(),
        )
        record.record_hash = self._compute_record_hash(record)
        self.records.append(record)
        return record

    def withdraw_consent(self, patient_id: str, use_case_id: str,
                         version: str = "v1.0") -> ConsentRecord:
        """Records a patient's consent withdrawal. Never deletes prior records."""
        record = ConsentRecord(
            record_id=str(uuid.uuid4()),
            patient_pseudonym=self._hash_patient_id(patient_id),
            use_case_id=use_case_id,
            use_case_description=self.use_case_registry.get(use_case_id, ""),
            action="WITHDRAW",
            timestamp=datetime.utcnow().isoformat() + "Z",
            expiry_date=None,
            version=version,
            previous_record_hash=self._get_last_hash(),
        )
        record.record_hash = self._compute_record_hash(record)
        self.records.append(record)
        return record

    def is_consented(self, patient_id: str, use_case_id: str) -> bool:
        """Returns True if the patient's most recent action for this use case is GRANT."""
        pseudonym = self._hash_patient_id(patient_id)
        relevant = [r for r in self.records
                    if r.patient_pseudonym == pseudonym and r.use_case_id == use_case_id]
        if not relevant:
            return False
        return relevant[-1].action == "GRANT"

    def verify_chain_integrity(self) -> bool:
        """
        Verifies the entire ledger chain has not been tampered with.
        Any mismatch indicates a record was modified after creation.
        """
        prev_hash = hashlib.sha256(b"genesis").hexdigest()
        for record in self.records:
            if record.previous_record_hash != prev_hash:
                print(f"[INTEGRITY FAIL] Record {record.record_id} has invalid previous hash.")
                return False
            expected_hash = self._compute_record_hash(record)
            if record.record_hash != expected_hash:
                print(f"[INTEGRITY FAIL] Record {record.record_id} hash mismatch — tampered.")
                return False
            prev_hash = record.record_hash
        return True

    def export_audit_trail(self) -> list[dict]:
        """Returns a list of all records as dicts for regulatory audit submission."""
        return [
            {
                "record_id": r.record_id,
                "patient_pseudonym": r.patient_pseudonym,
                "use_case_id": r.use_case_id,
                "action": r.action,
                "timestamp": r.timestamp,
                "version": r.version,
                "record_hash": r.record_hash,
            }
            for r in self.records
        ]


if __name__ == "__main__":
    print("=" * 70)
    print("DYNAMIC CONSENT LEDGER STUB")
    print("Chapter 5: Building a Privacy-First AI Strategy")
    print("Future Frontiers by Jigar Sheth")
    print("=" * 70)

    ledger = ConsentLedger()
    ledger.register_use_case("AI_DILI_TRAINING",
                              "Training an AI model to predict drug-induced liver injury risk")
    ledger.register_use_case("PV_SIGNAL_DETECTION",
                              "Using de-identified AE reports to detect safety signals")
    ledger.register_use_case("GENOMIC_RESEARCH",
                              "Genomic association study for drug metabolism variants")

    # Simulate three synthetic patients (no real identifiers)
    patients = ["SynPatient_001", "SynPatient_002", "SynPatient_003"]

    print("\nGranting consents...")
    for pid in patients:
        ledger.grant_consent(pid, "AI_DILI_TRAINING", expiry_date="2026-12-31")
        ledger.grant_consent(pid, "PV_SIGNAL_DETECTION")
    ledger.grant_consent(patients[0], "GENOMIC_RESEARCH")

    print("\nSynPatient_001 withdraws GENOMIC_RESEARCH consent...")
    ledger.withdraw_consent(patients[0], "GENOMIC_RESEARCH")

    print("\nConsent Status Check:")
    for pid in patients:
        for uc in ["AI_DILI_TRAINING", "PV_SIGNAL_DETECTION", "GENOMIC_RESEARCH"]:
            status = "✅ CONSENTED" if ledger.is_consented(pid, uc) else "❌ NOT CONSENTED"
            print(f"  {pid} | {uc:<30}: {status}")

    print(f"\nLedger contains {len(ledger.records)} records.")
    integrity = ledger.verify_chain_integrity()
    print(f"Chain integrity check: {'✅ PASSED' if integrity else '❌ FAILED'}")

    # Demonstrate tamper detection
    print("\nSimulating tamper attempt on record 0...")
    ledger.records[0].action = "WITHDRAW"  # Tamper!
    integrity_after_tamper = ledger.verify_chain_integrity()
    print(f"Chain integrity after tamper: {'✅ PASSED' if integrity_after_tamper else '❌ FAILED — tamper detected'}")
