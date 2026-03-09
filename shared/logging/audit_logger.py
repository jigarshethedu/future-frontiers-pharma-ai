"""
Structured audit logging for pharma AI regulatory inspection readiness.
Required for FDA SaMD, EU AI Act Art. 13, and GVP audit trails.
See: Ch10 (governance), Ch13 (security), Ch19 (monitoring).
"""
import json, datetime
from pathlib import Path

class AuditLogger:
    def __init__(self, log_dir="audit_logs", chapter=""):
        self.log_dir    = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.chapter    = chapter
        self.session_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def log(self, event_type, details):
        entry = {
            "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "chapter":    self.chapter,
            "event_type": event_type,
            "details":    details,
        }
        with open(self.log_dir / f"audit_{self.session_id}.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def log_model_training(self, model_name, n_samples,
                           privacy_protected, epsilon=None, metrics=None):
        return self.log("MODEL_TRAINING", {
            "model":             model_name,
            "n_samples":         n_samples,
            "privacy_protected": privacy_protected,
            "epsilon":           epsilon,
            "metrics":           metrics or {},
        })

    def log_data_access(self, accessor, data_type, purpose,
                        consent_verified, n_records=None):
        return self.log("DATA_ACCESS", {
            "accessor":         accessor,
            "data_type":        data_type,
            "purpose":          purpose,
            "consent_verified": consent_verified,
            "n_records":        n_records,
        })

    def log_model_decision(self, model_name, input_hash,
                           output, confidence, human_override=False):
        return self.log("MODEL_DECISION", {
            "model":          model_name,
            "input_hash":     input_hash,
            "output":         output,
            "confidence":     round(confidence, 4),
            "human_override": human_override,
        })
