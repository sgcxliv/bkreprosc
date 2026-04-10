#!/usr/bin/env python
"""
make_lrt_jobs.py
================
Generates SLURM PBS job scripts for likelihood ratio tests (LRT).

LRT is ONLY valid for NESTED model pairs, i.e. where Model A's predictor
set is a strict subset of Model B's.  Non-nested comparisons (e.g. cloze
vs. gpt2) must use permutation tests — see make_test_jobs.py.

Each job calls: python simple_lrt.py <experiment> <modelA>_v_<modelB>

NESTED PAIRS GENERATED
-----------------------
  1. nosurp vs. <model>          — null vs. surprisal
  2. nosurp vs. <model>prob      — null vs. probability
  3. <model> vs. <model>prob     — surprisal vs. probability (base vs. PROB)
  4. <model> vs. <model>prob-<model>   — surp vs. surp+prob (base vs. combined)
  5. <model>prob vs. <model>prob-<model> — prob vs. surp+prob (PROB vs. combined)

NOT included (non-nested — use permutation tests):
  - cloze vs. gpt2, etc.
  - cloze vs. gpt2region, etc.
  - base vs. region variants of different model families
"""

import os
import stat

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

experiments = ['spr']

models = [
    'cloze',    'clozeregion',
    'gpt2',     'gpt2region',
    'gpt2xl',   'gpt2xlregion',
    'gptj',     'gptjregion',
    'gptneo',   'gptneoregion',
    'gptneox',  'gptneoxregion',
    'olmo',     'olmoregion',
    'llama2',   'llama2region',
]

# Results path
if os.path.exists('bk21_results_path.txt'):
    with open('bk21_results_path.txt') as f:
        results_path = next(l.strip() for l in f if l.strip())
else:
    results_path = 'results/bk21'

PBS_TEMPLATE = """\
#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --output="{job_name}-%N-%j.out"
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --time=2:00:00
#SBATCH --mem=16gb
#SBATCH --ntasks=1

source ~/.bashrc
conda activate cs24

cd /afs/cs.stanford.edu/u/sgcxliv
python simple_lrt.py {experiment} {comparison}
"""


def write_executable(path, content):
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD NESTED COMPARISON LIST
# ─────────────────────────────────────────────────────────────────────────────

nested_comparisons = []

for model in models:
    # 1. Null vs. surprisal
    nested_comparisons.append(('nosurp', model))
    # 2. Null vs. probability
    nested_comparisons.append(('nosurp', '%sprob' % model))
    # 3. Surprisal vs. probability  (base vs. PROB)
    nested_comparisons.append((model, '%sprob' % model))
    # 4. Surprisal vs. surprisal+probability  (base vs. combined)
    nested_comparisons.append((model, '%sprob-%s' % (model, model)))
    # 5. Probability vs. surprisal+probability  (PROB vs. combined)
    nested_comparisons.append(('%sprob' % model, '%sprob-%s' % (model, model)))

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE JOB FILES
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("LRT JOBS GENERATOR")
print("=" * 70)
print("\nNested comparisons: %d" % len(nested_comparisons))

os.makedirs('lrt_scripts', exist_ok=True)
count = 0
for experiment in experiments:
    for (model_a, model_b) in nested_comparisons:
        comparison = '%s_v_%s' % (model_a, model_b)
        job_name   = 'lrt_%s_%s' % (experiment, comparison)
        pbs_path   = 'lrt_scripts/%s.pbs' % job_name
        write_executable(pbs_path, PBS_TEMPLATE.format(
            job_name=job_name,
            experiment=experiment,
            comparison=comparison
        ))
        count += 1

print("Generated %d job files in lrt_scripts/" % count)
print("\nTo submit:")
print("  cd lrt_scripts")
print("  for job in *.pbs; do sbatch $job; done")
print("  cd ..")
print("\nNOTE: LRT jobs depend on LME model fits completing first.")
print("      Submit these AFTER all lme_*.pbs jobs from make_bk21_jobs.py finish.")
