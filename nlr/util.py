import contextlib
import json
import os
from dataclasses import asdict
from enum import Enum
from json import JSONEncoder
from typing import Any

import sys
from _jsonnet import evaluate_file
from transformers import HfArgumentParser


def get_args(*arg_types):
    print(arg_types)
    parser = HfArgumentParser(arg_types)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".jsonnet"):
        conf = json.loads(evaluate_file(sys.argv[1]))
        args = parser.parse_dict(conf)
    else:
        args = parser.parse_args_into_dataclasses()
    return args


class Encoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value


def dump_args(*args, file_name):
    d = {str(type(arg).__name__): asdict(arg) for arg in args}
    with open(file_name, 'w+')as f:
        json.dump(d, f, cls=Encoder, indent=4)
