import os, sys
import traceback

# Context manager that copies stdout and any exceptions to a log file
class TeeLog(object):
    """
    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """
    def __init__(self, filename = None, mode = "wt"):
        self.files = []
        if filename is not None: self.append(filename, mode)
        self.stdout = sys.stdout
    # __init__

    def append(self, filename, mode = "wt"):
        try:
            file = open(filename, mode) if filename is not None and filename != "" else None
        except:
            file = None
        # try
        if file is not None: self.files.append(file)
    # init

    def __enter__(self):
        sys.stdout = self
        return self
    # __enter__

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            for file in self.files: file.write(traceback.format_exc())
        for file in self.files: file.close()
    # __exit__

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()
        self.stdout.write(data)
    # write

    def flush(self):
        for file in self.files: file.flush()
        self.stdout.flush()
    # flush
# TeeLog

def seed_everything (seed, tf2_set_seed = True, torch_set_seed = True, verbose = True):
    import random, os
    import numpy as np

    if verbose: print(f'+ Seed Everything: {seed} - tf2_set_seed = {tf2_set_seed} - torch_set_seed = {torch_set_seed}')

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if torch_set_seed == True:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if tf2_set_seed == True:
        import tensorflow as tf
        tf.random.set_seed(seed)
# seed_everything

def parse_params(node, keys = None, scope = None):
    if type(node) in [dict]:
        eval_keys = keys if keys is not None else list(node.keys())
        for k in eval_keys:
            node[k] = parse_params(node[k], scope = {**scope, **{v:node[v] for v in node if v != k}})
    elif type(node) in [list, tuple, set]:
        info = []
        for subnode in node: info.append(parse_params(subnode, scope=scope))
        node = info
    elif type(node) in [str]:
        if node.startswith("eval(") and node.endswith(")"):
            val_node = eval(node[5:-1], scope)
        else:
            val_node = eval("f'%s'" % node, scope)
            if str.isnumeric(val_node): val_node = eval(val_node)
        node = parse_params(val_node, scope=scope) if type(node) in [dict, list, set, tuple] else val_node
    return node
    pass

def run_command(cmd, init_globals = None, update_globals = None, type_cmd = "module", logs_file = None, verbose = True):
    """
    type_cmd: module, path
    """
    import runpy, shlex
    bak_argv = sys.argv
    bak_stdout = sys.stdout
    sys.argv = shlex.split(cmd)
    cmd_rets = None
    try:
        if verbose>=2: print(sys.argv)
        with TeeLog(f'{logs_file}'):
            if type_cmd == "module":
                cmd_rets = runpy.run_module(sys.argv[0], run_name = "__main__", init_globals = init_globals)
            elif type_cmd == "path":
                cmd_rets = runpy.run_path(sys.argv[0], run_name="__main__", init_globals=init_globals)
        # with
        if update_globals is not None: update_globals.update(**cmd_rets)
    except SystemExit as ex:
        if verbose>=1: print(f"Program Exit: {ex}")
    # try
    sys.argv = bak_argv
    sys.stdout = bak_stdout
    return cmd_rets
    pass # run_command