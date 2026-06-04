#!/bin/bash
# ============================================================================
# run_SIB_lme.sh
# ============================================================================

mkdir -p /nlp/scr/$USER/SIB/logs /nlp/scr/$USER/SIB/lme_results

ESTIMATORS=(
  "1:Cloze"
  "2:GPT-2"
  "3:GPT-2_XL"
  "4:GPT-J"
  "5:GPT-Neo"
  "6:GPT-NeoX"
  "7:LLaMA-2"
  "8:OLMo-2"
)

for entry in "${ESTIMATORS[@]}"; do
  idx="${entry%%:*}"
  name="${entry##*:}"
  echo "Submitting job $idx ($name)..."

  nlprun \
    -q john \
    -r 32G \
    -c 4 \
    -p standard \
    -o "/nlp/scr/$USER/SIB/logs/SIB_lme_${idx}_${name}.out" \
    "source /nlp/scr/$USER/miniconda3/etc/profile.d/conda.sh && conda activate r_env && Rscript /nlp/scr/$USER/SIB/SIB_lme_array.R $idx"

done

echo ""
echo "All 8 jobs submitted. Monitor with:  squeue -u $USER"
echo "When all finish, run:"
echo "  source /nlp/scr/$USER/miniconda3/etc/profile.d/conda.sh && conda activate r_env"
echo "  cd /nlp/scr/$USER/SIB && Rscript SIB_combine_and_plot.R"
