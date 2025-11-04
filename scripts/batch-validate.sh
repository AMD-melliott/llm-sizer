#!/bin/bash
# Batch Validation Script
# Run calculator validation against multiple profiler outputs and generate summary

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${1:-${SCRIPT_DIR}/../results/memory-profiles}"
OUTPUT_DIR="${SCRIPT_DIR}/../results/validation-reports"
SUMMARY_FILE="${OUTPUT_DIR}/batch-summary-$(date +%Y%m%d_%H%M%S).txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "LLM Sizer Batch Validation"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results Directory: $RESULTS_DIR"
echo "Output Directory:  $OUTPUT_DIR"
echo ""

# Find all JSON profile files
profiles=($(find "$RESULTS_DIR" -name "*.json" -type f))

if [ ${#profiles[@]} -eq 0 ]; then
    echo -e "${RED}Error: No JSON profile files found in $RESULTS_DIR${NC}"
    exit 1
fi

echo "Found ${#profiles[@]} profile(s) to validate"
echo ""

# Summary variables
total_profiles=0
exact_matches=0
good_matches=0
fair_matches=0
poor_matches=0
errors=0

# Initialize summary file
echo "LLM SIZER BATCH VALIDATION SUMMARY" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "═══════════════════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Process each profile
for profile in "${profiles[@]}"; do
    total_profiles=$((total_profiles + 1))
    profile_name=$(basename "$profile")
    
    echo -e "${BLUE}[$total_profiles/${#profiles[@]}]${NC} Validating: $profile_name"
    
    # Generate output filename
    output_json="${OUTPUT_DIR}/${profile_name%.json}_validation.json"
    output_txt="${OUTPUT_DIR}/${profile_name%.json}_validation.txt"
    
    # Run validation
    if npm run validate-calculator -- "$profile" --json > "$output_json" 2>/dev/null; then
        # Success - extract match quality
        match_quality=$(jq -r '.summary.overall_match' "$output_json" 2>/dev/null || echo "unknown")
        percent_diff=$(jq -r '.summary.total_percent_diff' "$output_json" 2>/dev/null || echo "0")
        
        case "$match_quality" in
            "exact")
                exact_matches=$((exact_matches + 1))
                echo -e "  ${GREEN}✓✓ EXACT${NC} match (${percent_diff}% difference)"
                status_icon="✓✓"
                ;;
            "good")
                good_matches=$((good_matches + 1))
                echo -e "  ${GREEN}✓ GOOD${NC} match (${percent_diff}% difference)"
                status_icon="✓"
                ;;
            "fair")
                fair_matches=$((fair_matches + 1))
                echo -e "  ${YELLOW}⚠ FAIR${NC} match (${percent_diff}% difference)"
                status_icon="⚠"
                ;;
            "poor")
                poor_matches=$((poor_matches + 1))
                echo -e "  ${RED}✗ POOR${NC} match (${percent_diff}% difference)"
                status_icon="✗"
                ;;
            *)
                errors=$((errors + 1))
                echo -e "  ${RED}? UNKNOWN${NC}"
                status_icon="?"
                ;;
        esac
        
        # Also generate text report
        npm run validate-calculator -- "$profile" > "$output_txt" 2>/dev/null || true
        
        # Add to summary
        echo "$status_icon  ${profile_name} - ${match_quality^^} (${percent_diff}%)" >> "$SUMMARY_FILE"
        
    else
        # Error during validation
        errors=$((errors + 1))
        error_msg=$(npm run validate-calculator -- "$profile" 2>&1 | tail -n 1 || echo "Unknown error")
        echo -e "  ${RED}✗ ERROR:${NC} $error_msg"
        echo "✗  ${profile_name} - ERROR: $error_msg" >> "$SUMMARY_FILE"
    fi
    
    echo ""
done

# Write summary statistics
echo "" >> "$SUMMARY_FILE"
echo "═══════════════════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"
echo "SUMMARY STATISTICS" >> "$SUMMARY_FILE"
echo "───────────────────────────────────────────────────────────────────────────" >> "$SUMMARY_FILE"
echo "Total Profiles:   $total_profiles" >> "$SUMMARY_FILE"
echo "Exact Matches:    $exact_matches (✓✓)" >> "$SUMMARY_FILE"
echo "Good Matches:     $good_matches (✓)" >> "$SUMMARY_FILE"
echo "Fair Matches:     $fair_matches (⚠)" >> "$SUMMARY_FILE"
echo "Poor Matches:     $poor_matches (✗)" >> "$SUMMARY_FILE"
echo "Errors:           $errors" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

successful=$((exact_matches + good_matches + fair_matches + poor_matches))
if [ $successful -gt 0 ]; then
    success_rate=$(echo "scale=1; ($exact_matches + $good_matches) * 100 / $successful" | bc)
    echo "Success Rate:     ${success_rate}% (Exact + Good)" >> "$SUMMARY_FILE"
fi

echo "═══════════════════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"

# Print summary to console
echo "═══════════════════════════════════════════════════════════════════════════"
echo "BATCH VALIDATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Total Profiles:   $total_profiles"
echo -e "Exact Matches:    ${GREEN}$exact_matches${NC} (✓✓)"
echo -e "Good Matches:     ${GREEN}$good_matches${NC} (✓)"
echo -e "Fair Matches:     ${YELLOW}$fair_matches${NC} (⚠)"
echo -e "Poor Matches:     ${RED}$poor_matches${NC} (✗)"
echo -e "Errors:           ${RED}$errors${NC}"
echo ""

if [ $successful -gt 0 ]; then
    success_rate=$(echo "scale=1; ($exact_matches + $good_matches) * 100 / $successful" | bc)
    echo -e "Success Rate:     ${GREEN}${success_rate}%${NC} (Exact + Good matches)"
    echo ""
fi

echo "Summary saved to: $SUMMARY_FILE"
echo "Individual reports saved to: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════════════════"

# Exit with error if too many poor matches or errors
if [ $((poor_matches + errors)) -gt $((total_profiles / 2)) ]; then
    echo ""
    echo -e "${RED}Warning: More than 50% of profiles had poor matches or errors${NC}"
    exit 1
fi

exit 0
