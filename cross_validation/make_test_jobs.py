#!/usr/bin/env python
"""
make_test_jobs.py
=================
Generates SLURM PBS job scripts for permutation tests (PT).

Permutation tests compare HELD-OUT log-likelihoods from LME models
(fitted by make_bk21_jobs.py).  Unlike LRT, permutation tests do NOT
assume nestedness and do NOT rely on asymptotic chi-squared distributions.
They are valid for both nested and non-nested comparisons, but we use them
primarily for the NON-NESTED comparisons where LRT is not applicable.

Permutation tests are also generated for within-model surp vs. prob
comparisons as a cross-check against the LRT results.

COMPARISON CATEGORIES
----------------------
  1.  Null vs. model surp/prob     (redundant check; LRT is primary here)
  2.  Surp vs. prob (within model, non-region)
  3.  Both vs. single (surp+prob vs. surp or prob alone)
  4a. Cloze vs. other non-region models  (bakeoff)
  4b. Cloze-Region vs. other region models  (bakeoff)
  5.  Cloze surp vs. model region surp (cross-type)
  6.  Within-model surp vs. region surp (e.g. gpt2 vs. gpt2region)
  7.  Cloze prob vs. model region prob
  8.  Cloze prob vs. model prob (non-region)
  9.  Cloze-region prob vs. model region prob
  10a. ClozeProb vs. model word surprisal (non-region)
  10b. ClozeProb vs. model region surprisal

Each job calls: python test.py <experiment> <comparison>
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

non_region = [m for m in models if 'region' not in m]
region     = [m for m in models if 'region' in m]

region_map = {
    'cloze':   'clozeregion',
    'gpt2':    'gpt2region',
    'gpt2xl':  'gpt2xlregion',
    'gptj':    'gptjregion',
    'gptneo':  'gptneoregion',
    'gptneox': 'gptneoxregion',
    'olmo':    'olmoregion',
    'llama2':  'llama2region',
}

PBS_HEADER = """\
#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --output="{job_name}-%N-%j.out"
#SBATCH --account=nlp
#SBATCH --partition=sphinx
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --ntasks=1

source ~/.bashrc
conda activate cs24

cd /afs/cs.stanford.edu/u/sgcxliv
python test.py {experiment} {comparison}
"""


def write_executable(path, content):
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD COMPARISON LISTS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PERMUTATION TEST JOBS GENERATOR")
print("=" * 70)

# 1. Null vs. model (surp and prob separately)
cat1 = []
for m in models:
    cat1.append('nosurp_v_%s' % m)        # null vs. surp
    cat1.append('nosurp_v_%sprob' % m)    # null vs. prob
print("\n1. Null comparisons: %d" % len(cat1))

# 2. Surp vs. prob (within model, non-region only)
cat2 = []
for m in non_region:
    cat2.append('%s_v_%sprob' % (m, m))
print("2. Surp vs. prob (non-region): %d" % len(cat2))

# 3. Both vs. single (for ALL models including region)
cat3 = []
for m in models:
    cat3.append('%s_v_%sprob-%s' % (m, m, m))       # surp vs. surp+prob
    cat3.append('%sprob_v_%sprob-%s' % (m, m, m))   # prob vs. surp+prob
print("3. Both vs. single: %d" % len(cat3))

# 4a. Cloze vs. other non-region models (bakeoff, surp)
cat4a = []
for m in non_region:
    if m != 'cloze':
        cat4a.append('cloze_v_%s' % m)
print("4a. Cloze vs. non-region (bakeoff): %d" % len(cat4a))

# 4b. Cloze-region vs. other region models (bakeoff, surp)
cat4b = []
for m in region:
    if m != 'clozeregion':
        cat4b.append('clozeregion_v_%s' % m)
print("4b. Cloze-region vs. region (bakeoff): %d" % len(cat4b))

# 5. Cloze surp vs. model region surp
cat5 = []
for m in region:
    cat5.append('cloze_v_%s' % m)
print("5. Cloze surp vs. region surp: %d" % len(cat5))

# 6. Within-model surp vs. region surp (e.g. gpt2 vs. gpt2region)
cat6 = []
for base, reg in region_map.items():
    if base in non_region and reg in region:
        cat6.append('%s_v_%s' % (base, reg))
print("6. Within-model surp vs. region surp: %d" % len(cat6))

# 7. Cloze prob vs. model region prob
cat7 = []
for m in region:
    cat7.append('clozeprob_v_%sprob' % m)
print("7. Cloze prob vs. region prob: %d" % len(cat7))

# 8. Cloze prob vs. model prob (non-region, excl. cloze itself)
cat8 = []
for m in non_region:
    if m != 'cloze':
        cat8.append('clozeprob_v_%sprob' % m)
print("8. Cloze prob vs. model prob (non-region): %d" % len(cat8))

# 9. Cloze-region prob vs. model region prob
cat9 = []
for m in region:
    if m != 'clozeregion':
        cat9.append('clozeregionprob_v_%sprob' % m)
print("9. Cloze-region prob vs. model region prob: %d" % len(cat9))

# 10a. ClozeProb vs. model word surprisal (non-region, excl. cloze itself)
cat10a = []
for m in non_region:
    if m != 'cloze':
        cat10a.append('clozeprob_v_%s' % m)
print("10a. ClozeProb vs. model word surprisal (non-region): %d" % len(cat10a))

# 10b. ClozeProb vs. model region surprisal
cat10b = []
for m in region:
    cat10b.append('clozeprob_v_%s' % m)
print("10b. ClozeProb vs. model region surprisal: %d" % len(cat10b))

# Combine, deduplicate, preserve order
seen = set()
all_comparisons = []
for c in (cat1 + cat2 + cat3 + cat4a + cat4b + cat5 + cat6 + cat7 + cat8 + cat9 + cat10a + cat10b):
    if c not in seen:
        seen.add(c)
        all_comparisons.append(c)

print("\nTotal unique comparisons: %d" % len(all_comparisons))

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE JOB FILES
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs('test_scripts', exist_ok=True)
count = 0
for experiment in experiments:
    for comparison in all_comparisons:
        job_name = 'pt_%s_%s' % (experiment, comparison)
        pbs_path = 'test_scripts/%s.pbs' % job_name
        write_executable(pbs_path, PBS_HEADER.format(
            job_name=job_name,
            experiment=experiment,
            comparison=comparison
        ))
        count += 1

print("\nGenerated %d job files in test_scripts/" % count)
print("\nTo submit:")
print("  cd test_scripts")
print("  for job in *.pbs; do sbatch $job; done")
print("  cd ..")
print("\nNOTE: Permutation test jobs depend on LME model fits completing first.")
print("      Submit these AFTER all lme_*.pbs jobs from make_bk21_jobs.py finish.")
