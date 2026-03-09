#!/usr/bin/env bash
# =============================================================================
# commit_chapter.sh — Future Frontiers by Jigar Sheth
# =============================================================================
#
# USAGE:
#   bash commit_chapter.sh <chapter_number> [staging_folder]
#
# EXAMPLES:
#   bash commit_chapter.sh 5
#   bash commit_chapter.sh 5 ~/Downloads/chapter5_files
#   bash commit_chapter.sh 5 ~/Desktop/staging
#
# WHAT IT DOES (in order):
#   1. Validates you are inside the repo root
#   2. Reads expected files for the chapter from the built-in registry
#   3. Copies files from staging folder → correct repo paths
#   4. Runs pytest on the chapter's tests
#   5. Runs all figure scripts to generate PNGs
#   6. Stages all new/changed files
#   7. Commits with a standardized message
#   8. Pushes to origin/main
#   9. Prints a summary report
#
# STAGING FOLDER LAYOUT (what Claude delivers, what you save):
#   staging/
#     ├── <module_name>.py          → chapter[NN]/<module_name>.py
#     ├── test_<module_name>.py     → chapter[NN]/tests/test_<module_name>.py
#     ├── fig[NN]_<name>.py         → chapter[NN]/figures/fig[NN]_<name>.py
#     └── Sheth_Future_Frontiers_<N>__<Title>.docx → docs/chapters/
#
# SETUP (one-time, run once after cloning the repo):
#   chmod +x commit_chapter.sh
#   pip install -r requirements.txt
#
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
header()  { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"; \
            echo -e "${BOLD}${BLUE}  $*${NC}"; \
            echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}"; }

# ── Argument parsing ─────────────────────────────────────────────────────────
[[ $# -lt 1 ]] && error "Usage: bash commit_chapter.sh <chapter_number> [staging_folder]"

CHAPTER_NUM="$1"
STAGING_DIR="${2:-}"

# Zero-pad chapter number for folder names (chapter01, chapter05, chapter12…)
CHAPTER_PAD=$(printf "%02d" "$CHAPTER_NUM")
CHAPTER_DIR="chapter${CHAPTER_PAD}"

# ── Validate repo root ────────────────────────────────────────────────────────
[[ ! -f "requirements.txt" ]] && \
  error "Run this script from the repo root (future-frontiers-pharma-ai/). Not found: requirements.txt"
[[ ! -d ".git" ]] && \
  error "No .git directory found. Run from inside the cloned repo."
[[ ! -d "$CHAPTER_DIR" ]] && \
  error "Chapter folder '$CHAPTER_DIR' does not exist. Run setup_repo.sh first."

header "Future Frontiers — Chapter ${CHAPTER_NUM} Commit"
info "Chapter folder : $CHAPTER_DIR"
info "Repo root      : $(pwd)"
info "Git branch     : $(git branch --show-current)"
echo ""

# ── Module registry ───────────────────────────────────────────────────────────
# Complete list of expected modules per chapter, derived from project instructions.
# Format: "module_name" (without .py extension)
# Tests are automatically expected as tests/test_<module_name>.py

declare -A CHAPTER_MODULES
CHAPTER_MODULES[5]="privacy_by_design_checklist differential_privacy_demo federated_learning_stub homomorphic_encryption_demo smpc_demo privacy_impact_assessment synthetic_data_generator data_minimization_analyzer consent_ledger_stub privacy_budget_optimizer"
CHAPTER_MODULES[6]="data_lineage_tracker data_sharing_agreement_checker metadata_standards_validator cloud_governance_audit data_quality_scorer"
CHAPTER_MODULES[7]="flower_fl_clinical substrafl_drug_discovery vertical_fl_demo fl_convergence_monitor fedprox_implementation fl_fairness_evaluator"
CHAPTER_MODULES[8]="blockchain_consent_ledger zkp_proof_generator gan_synthetic_ehr diffusion_synthetic_clinical tee_secure_inference quantum_safe_key_exchange"
CHAPTER_MODULES[9]="compliance_scorecard regulatory_submission_checker cross_border_transfer_decision_tree aia_risk_classifier privacy_kpi_dashboard"
CHAPTER_MODULES[10]="ai_register_template ethics_committee_charter_gen vendor_ai_assessment ai_incident_response model_retirement_checklist"
CHAPTER_MODULES[11]="fairlearn_bias_audit aif360_pipeline subgroup_performance_monitor model_card_generator dataset_inclusivity_scorer disparate_impact_analyzer"
CHAPTER_MODULES[12]="pv_signal_detector faers_nlp_pipeline synthetic_adverse_event_generator pv_privacy_architecture"
CHAPTER_MODULES[13]="adversarial_attack_demo model_poisoning_detector ai_threat_model_generator secure_mlops_checker sbom_generator"
CHAPTER_MODULES[14]="success_metrics_dashboard roi_case_study_calculator"
CHAPTER_MODULES[15]="failure_postmortem_framework bias_regression_tester model_drift_detector"
CHAPTER_MODULES[16]="stakeholder_workshop_template explainability_dashboard trust_score_calculator"
CHAPTER_MODULES[17]="esg_ai_scorecard privacy_premium_calculator ai_due_diligence_checker"
CHAPTER_MODULES[18]="frl_adaptive_treatment post_quantum_migration_planner agent_privacy_risk_model foundation_model_audit"
CHAPTER_MODULES[19]="model_drift_monitor privacy_budget_tracker ai_audit_trail_generator retraining_governance_gate performance_kpi_registry"
CHAPTER_MODULES[20]="patient_engagement_simulator fhir_consent_integrator cultural_consent_adapter health_literacy_assessor"
CHAPTER_MODULES[21]="ai_carbon_calculator model_efficiency_optimizer green_fl_communicator scope3_ai_emissions"

# ── Chapter titles (for commit messages) ─────────────────────────────────────
declare -A CHAPTER_TITLES
CHAPTER_TITLES[5]="Building a Privacy-First AI Strategy"
CHAPTER_TITLES[6]="Data Governance for Pharma AI"
CHAPTER_TITLES[7]="Federated Learning: The Comprehensive Implementation Guide"
CHAPTER_TITLES[8]="Technology to the Rescue: Advanced PETs and Emerging Tools"
CHAPTER_TITLES[9]="Framework for Compliance"
CHAPTER_TITLES[10]="Proactive Governance in Pharma AI"
CHAPTER_TITLES[11]="Developing Ethical and Fair AI"
CHAPTER_TITLES[12]="AI in Pharmacovigilance"
CHAPTER_TITLES[13]="Cybersecurity and AI Risk"
CHAPTER_TITLES[14]="Success Stories: What Good Looks Like"
CHAPTER_TITLES[15]="Lessons from Failures: What Went Wrong and Why"
CHAPTER_TITLES[16]="Stakeholder Collaboration Framework"
CHAPTER_TITLES[17]="The Business of Ethics"
CHAPTER_TITLES[18]="Emerging Trends: What Is Coming and When"
CHAPTER_TITLES[19]="Continuous Monitoring and Auditing"
CHAPTER_TITLES[20]="Patient-Centric AI: Putting People Back at the Center"
CHAPTER_TITLES[21]="Sustainable AI in Pharma"

CHAPTER_TITLE="${CHAPTER_TITLES[$CHAPTER_NUM]:-Chapter $CHAPTER_NUM}"

# ── Step 1: Copy files from staging → repo ───────────────────────────────────
header "Step 1: Copy files from staging folder"

if [[ -n "$STAGING_DIR" ]]; then
  [[ ! -d "$STAGING_DIR" ]] && error "Staging folder not found: $STAGING_DIR"
  info "Staging folder: $STAGING_DIR"
  echo ""

  # Ensure subdirectories exist
  mkdir -p "${CHAPTER_DIR}/tests" "${CHAPTER_DIR}/figures" "docs/chapters"

  COPIED=0
  SKIPPED=0

  for f in "$STAGING_DIR"/*; do
    fname=$(basename "$f")

    if [[ "$fname" == test_*.py ]]; then
      dest="${CHAPTER_DIR}/tests/${fname}"
      cp "$f" "$dest"
      success "Copied test : $dest"
      ((COPIED++))

    elif [[ "$fname" == fig${CHAPTER_PAD}_*.py || "$fname" == fig${CHAPTER_NUM}_*.py ]]; then
      dest="${CHAPTER_DIR}/figures/${fname}"
      cp "$f" "$dest"
      success "Copied figure: $dest"
      ((COPIED++))

    elif [[ "$fname" == *.py ]]; then
      dest="${CHAPTER_DIR}/${fname}"
      cp "$f" "$dest"
      success "Copied module: $dest"
      ((COPIED++))

    elif [[ "$fname" == *.docx ]]; then
      dest="docs/chapters/${fname}"
      cp "$f" "$dest"
      success "Copied docx  : $dest"
      ((COPIED++))

    else
      warn "Skipped (unknown type): $fname"
      ((SKIPPED++))
    fi
  done

  echo ""
  info "Copied: $COPIED files | Skipped: $SKIPPED files"
else
  warn "No staging folder provided. Skipping file copy."
  warn "Assuming files are already in place in $CHAPTER_DIR/"
fi

# ── Step 2: Validate expected modules are present ────────────────────────────
header "Step 2: Validate expected modules"

EXPECTED_MODULES="${CHAPTER_MODULES[$CHAPTER_NUM]:-}"
MISSING_MODULES=()
MISSING_TESTS=()

if [[ -z "$EXPECTED_MODULES" ]]; then
  warn "No module registry entry for Chapter $CHAPTER_NUM. Skipping validation."
else
  for mod in $EXPECTED_MODULES; do
    mod_path="${CHAPTER_DIR}/${mod}.py"
    test_path="${CHAPTER_DIR}/tests/test_${mod}.py"

    if [[ -f "$mod_path" ]]; then
      success "Module  : $mod_path"
    else
      warn "MISSING : $mod_path"
      MISSING_MODULES+=("$mod_path")
    fi

    if [[ -f "$test_path" ]]; then
      success "Test    : $test_path"
    else
      warn "MISSING : $test_path"
      MISSING_TESTS+=("$test_path")
    fi
  done
fi

echo ""
if [[ ${#MISSING_MODULES[@]} -gt 0 || ${#MISSING_TESTS[@]} -gt 0 ]]; then
  warn "${#MISSING_MODULES[@]} module(s) and ${#MISSING_TESTS[@]} test file(s) missing."
  warn "Commit will proceed but GitHub Actions CI will flag these."
  echo ""
  read -r -p "  Continue anyway? [y/N] " CONTINUE
  [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]] && error "Aborted by user."
else
  success "All expected modules and test files present."
fi

# ── Step 3: Run pytest ────────────────────────────────────────────────────────
header "Step 3: Run pytest — Chapter ${CHAPTER_NUM}"

TEST_DIR="${CHAPTER_DIR}/tests"
PYTEST_PASS=true

if [[ ! -d "$TEST_DIR" || -z "$(ls -A "$TEST_DIR" 2>/dev/null)" ]]; then
  warn "No tests found in $TEST_DIR. Skipping pytest."
else
  echo ""
  if python3 -m pytest "$TEST_DIR" -v --tb=short 2>&1; then
    success "All tests passed."
    PYTEST_PASS=true
  else
    echo ""
    warn "One or more tests FAILED."
    PYTEST_PASS=false
    echo ""
    read -r -p "  Tests failed. Commit anyway? [y/N] " CONTINUE
    [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]] && error "Aborted — fix tests before committing."
  fi
fi

# ── Step 4: Generate figures ──────────────────────────────────────────────────
header "Step 4: Generate figures"

FIGURES_DIR="${CHAPTER_DIR}/figures"
FIG_COUNT=0
FIG_ERRORS=()

if [[ ! -d "$FIGURES_DIR" || -z "$(ls -A "$FIGURES_DIR"/*.py 2>/dev/null)" ]]; then
  warn "No figure scripts found in $FIGURES_DIR. Skipping."
else
  # Run each figure script from inside the figures directory
  # so plt.savefig() writes the PNG alongside the script
  pushd "$FIGURES_DIR" > /dev/null
  for fig_script in fig*.py; do
    [[ ! -f "$fig_script" ]] && continue
    fig_name="${fig_script%.py}"
    echo -n "  Running $fig_script ... "
    if python3 "$fig_script" > /dev/null 2>&1; then
      echo -e "${GREEN}OK${NC}"
      ((FIG_COUNT++))
    else
      echo -e "${RED}FAILED${NC}"
      FIG_ERRORS+=("$fig_script")
    fi
  done
  popd > /dev/null

  echo ""
  if [[ ${#FIG_ERRORS[@]} -gt 0 ]]; then
    warn "${#FIG_ERRORS[@]} figure script(s) failed: ${FIG_ERRORS[*]}"
    warn "PNGs for failed figures will not be committed."
  else
    success "Generated $FIG_COUNT figure(s)."
  fi
fi

# ── Step 5: Git stage ─────────────────────────────────────────────────────────
header "Step 5: Stage changes"

git add "${CHAPTER_DIR}/" 2>/dev/null || true
git add "docs/chapters/" 2>/dev/null || true
git add "CHANGELOG.md" 2>/dev/null || true

STAGED=$(git diff --cached --name-only | wc -l | tr -d ' ')
info "$STAGED file(s) staged."
echo ""
git diff --cached --name-only | sed 's/^/    + /'
echo ""

if [[ "$STAGED" -eq 0 ]]; then
  warn "Nothing to commit — all files already up to date."
  exit 0
fi

# ── Step 6: Update CHANGELOG ──────────────────────────────────────────────────
header "Step 6: Update CHANGELOG"

CHANGELOG_ENTRY="- Chapter ${CHAPTER_NUM}: ${CHAPTER_TITLE} — modules, tests, figures, docx added"
DATE_TODAY=$(date +"%Y-%m-%d")

# Insert after the [Unreleased] line
if grep -q "\[Unreleased\]" CHANGELOG.md; then
  sed -i "s/## \[Unreleased\]/## [Unreleased]\n${CHANGELOG_ENTRY}/" CHANGELOG.md
  git add CHANGELOG.md
  success "CHANGELOG.md updated."
else
  warn "Could not find [Unreleased] section in CHANGELOG.md — skipping update."
fi

# ── Step 7: Commit ────────────────────────────────────────────────────────────
header "Step 7: Commit"

TEST_STATUS="tests: PASS"
[[ "$PYTEST_PASS" == false ]] && TEST_STATUS="tests: WARN (some failures)"

COMMIT_MSG="Chapter ${CHAPTER_NUM}: ${CHAPTER_TITLE}

- Added ${STAGED} file(s) for Chapter ${CHAPTER_NUM}
- ${TEST_STATUS}
- Figures: ${FIG_COUNT} generated
- Book: Future Frontiers by Jigar Sheth
- Committed: $(date '+%Y-%m-%d %H:%M')"

git commit -m "$COMMIT_MSG"
success "Committed: $COMMIT_MSG"

# ── Step 8: Push ──────────────────────────────────────────────────────────────
header "Step 8: Push to origin/main"

git push origin main
success "Pushed to origin/main."

# ── Step 9: Summary ───────────────────────────────────────────────────────────
header "✅  Chapter ${CHAPTER_NUM} Complete"

echo ""
echo -e "  ${BOLD}Chapter ${CHAPTER_NUM}:${NC} ${CHAPTER_TITLE}"
echo -e "  ${BOLD}Files staged:${NC}   ${STAGED}"
echo -e "  ${BOLD}Tests:${NC}          $( [[ "$PYTEST_PASS" == true ]] && echo -e "${GREEN}PASSED${NC}" || echo -e "${YELLOW}WARNINGS${NC}" )"
echo -e "  ${BOLD}Figures:${NC}        ${FIG_COUNT} generated"
echo -e "  ${BOLD}Missing modules:${NC} ${#MISSING_MODULES[@]}"
echo -e "  ${BOLD}Missing tests:${NC}  ${#MISSING_TESTS[@]}"
echo ""
echo -e "  ${BOLD}GitHub Actions CI:${NC}"
echo -e "  → https://github.com/jigarsheth-pharmaai/future-frontiers-pharma-ai/actions"
echo ""

if [[ ${#MISSING_MODULES[@]} -gt 0 ]]; then
  echo -e "  ${YELLOW}⚠  Missing modules (ask Claude to generate):${NC}"
  for m in "${MISSING_MODULES[@]}"; do echo "     $m"; done
  echo ""
fi

if [[ ${#FIG_ERRORS[@]} -gt 0 ]]; then
  echo -e "  ${YELLOW}⚠  Figure scripts that failed:${NC}"
  for f in "${FIG_ERRORS[@]}"; do echo "     ${CHAPTER_DIR}/figures/$f"; done
  echo ""
fi

echo -e "  ${GREEN}${BOLD}Next step:${NC} Copy Chapter ${CHAPTER_NUM} Word doc to docs/chapters/"
echo -e "  ${GREEN}${BOLD}Then:${NC}      bash commit_chapter.sh $((CHAPTER_NUM + 1)) ~/path/to/staging"
echo ""
