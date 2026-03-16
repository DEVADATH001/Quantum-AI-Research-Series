"""Author: DEVADATH H K

Project: Quantum Chemistry VQE"""

import sys
import os
import yaml
import json
sys.path.insert(0, os.path.abspath('.'))
from src.pes_generator import PESGenerator

config_str = """
general: {random_seed: 7, allow_synthetic_fallback: true}
molecules:
  H2:
    distances: {start: 0.9, end: 0.9, step: 0.1}
    charge: 0
    spin: 0
    basis: 'sto3g'
vqe:
  ansatz: [{name: 'UCCSD'}]
  optimizer: {name: 'SLSQP', maxiter: 200}
runtime: {backend: 'local'}
analysis: {chemical_accuracy_mhartree: 1.6}
"""
config = yaml.safe_load(config_str)
gen = PESGenerator(config)
res = gen.run('H2')
history = res['histories']['UCCSD']['0.900']
print(json.dumps(history, indent=2))
