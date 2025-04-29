import sys

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=1:00:00
#SBATCH --mem=1gb
#SBATCH --ntasks=1
"""

wrapper = '\n%s'

experiments = [
  'spr',
]

models = [
    'cloze',
    'trigram',
    'gpt2new',
    'gpt2regionnew',
    'gpt2xlnew',
    'gpt2xlregionnew',
    'gptjnew',
    'gptjregionnew',
    'gptneonew',
    'gptneoregionnew',
    'gptneoxnew',
    'gptneoxregionnew',
    'olmonew',
    'olmoregionnew',
    'llama2new',
    'llama2regionnew',
]

functions = [
    'prob',
    'surp',
]

comparisons_nosurp = []
for model in models:
    for fn in functions:
        s = 'nosurp_v_%s%s' % (model, fn)
        comparisons_nosurp.append(s)

comparisons_fn = []
for i, fn1 in enumerate(functions):
    for model in models:
        if i < len(functions) - 1:
            for fn2 in functions[i+1:]:
                s = '%s%s_v_%s%s' % (model, fn1, model, fn2)
                comparisons_fn.append(s)

comparisons_clozeprob = []
for model in models:
    if model != 'cloze':
        s = 'clozeprob_v_%s' % model
        comparisons_clozeprob.append(s)
        s = 'clozeprob_v_clozeprob-%s' % model
        comparisons_clozeprob.append(s)

comparisons_diamond = []
for model in models:
    for i in range(2):
        if i == 0:
            s = '%s_v_%sprob-%s' % (model, model, model)
        else:
            s = '%sprob_v_%sprob-%s' % (model, model, model)
        comparisons_diamond.append(s)

comparisons_direct = []
for model in models:
    s = '%s_v_%sprob' % (model, model)
    comparisons_direct.append(s)
    
comparisons_null_surp = []
for model in models:
    s = 'nosurp_v_%s' % model
    comparisons_null_surp.append(s)
    
comparisons = comparisons_nosurp + comparisons_fn + comparisons_clozeprob + comparisons_diamond + comparisons_direct + comparisons_null_surp

for experiment in experiments:
    for comparison in comparisons:
        job_name = 'bk21_%s_%s_test' % (experiment, comparison)
        job_str = 'python -m cdr.bk21.test %s %s' % (experiment, comparison)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
