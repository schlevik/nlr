from dataclasses import field
from transformers import HfArgumentParser

import re
import subprocess
import json
import numpy

from nlr.generate_rel_triple import *


@dataclass()
class ScriptArguments:
    """
    Class for command-line arguments.
    Contains all variable parameters used in generation of problem sets.
    """
    na: int = field(default = 200) # number of atoms
    nf: int = field(default = 200) # number of formulae
    pt: float = field(default = 0.5) # probability of triple
    pu: float = field(default = 0.8) # probability of universal
    pn: float = field(default = 0.5) # probability of negated predicate
    e: bool = field(default = True) # true for random generation,
                                    # false for removal of trivial absurdities
    ch: int = field(default = 0) # length of chain blowup in hard generation
    s: int = field(default = 500) # how many problems to generate
    sd: int = field(default = random.randint(0, 9223372036854775807)) # seed


# Get command line arguments
parser = HfArgumentParser((ScriptArguments,))
args, *_ = parser.parse_args_into_dataclasses()

# Seed the random number generator
random.seed(args.sd)

# Create name tag for problem set directory,
# containing information about all parameters
tag = "Tri" + ("" if args.e else ("hard" + str(args.ch))) + "na" + str(args.na)
tag += "nf" + str(args.nf) + "pt" + str(args.pt) + "pu" + str(args.pu)
tag += "pn" + str(args.pn) + "s" + str(args.s) + "sd" + str(args.sd)
subprocess.run(["mkdir", tag]) # Create the directory

count = 0 # Variable to count number of satisfiable problems
proof_lens = [] # List of proof lengths of unsatisfiable problems
json_out = [] # List of problems in JSON format
for i in range(args.s): # Generate s problems.
    # Store filepath for current problem
    out = tag + "/problem" + "_" + str(i + 1) + ".tptp"

    # Generate problem
    problem = Tproblem(args.na, args.nf, [], [], "", args.e, args.ch)
    problem.generate(args.pt, args.pu, args.pn)

    # Write problem to file
    with open(out, "w") as f:
        f.write(problem.toTPTP())

    # Run Vampire on problem to check satisfiability
    proof = subprocess.run(["./vampire", out],
                           capture_output = True,
                           text = True)

    # Extract result
    regex = "(?<=Termination reason: )\w+(?=\\n)"
    result = re.search(regex, proof.stdout).group()

    # Tag problem as satisfiable/unsatisfiable
    if result == "Refutation": # If problem is unsatisfiable
        problem.consistent += "in"

        # record number of formulas required for refutation
        problem.ref_subset = proof.stdout.count("[input]")

        # record the length of the proof
        # (subtract 14 lines of constant non-proof Vampire output)
        problem.proof_len = proof.stdout.count("\n") - 14
        proof_lens.append(problem.proof_len)
    else: # If problem is satisfiable
        count += 1 # uptick counter
    problem.consistent += "consistent"

    # Append tag(s) to file
    with open(out, "a") as f:
        f.write("% " + result + "\n")
        if result == "Refutation":
            f.write("% Number of lines in proof: "
                    + str(problem.proof_len) + "\n")

    # Store problem's JSON representation
    json_out.append(json.dumps(problem.toJSON()))

# Output all problems to JSON file
with open(tag + "/" + tag + ".json", "w") as f:
    f.write("\n".join(json_out) + "\n")

# Calculate percentage of satisfiable problems in set
perc = count / args.s * 100

print(perc)

# Create statistics file
# with satisfiability percentage and proof length information
with open(tag + "/stats.txt", "w") as f:
    f.write("Satisfiability percentage: " + str(perc) + "%\n\n")
    if perc != 100:
        f.write("Proof lengths of unsat problems: \n" + str(proof_lens) + "\n"
                + "Average proof length: "
                + str(sum(proof_lens) / len(proof_lens)) + "\n"
                + "Standard deviation: " + str(numpy.std(proof_lens)) + "\n")
