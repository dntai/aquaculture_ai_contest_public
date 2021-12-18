from __future__ import absolute_import, division, print_function
from IPython import display, get_ipython
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except:
    pass

import sys, os
try:
    script_file = os.path.normpath(os.path.abspath(__file__))
    script_dir  = os.path.normpath(os.path.abspath(os.path.dirname(__file__))) # python
except:
    script_dir = os.path.normpath(os.path.abspath(".")) # jupyter-notebook
    script_file= f'{script_dir}/cli_main.py'
root_dir    = os.path.normpath(os.path.abspath(script_dir + "/.."))
if root_dir in sys.path: sys.path.remove(root_dir)
sys.path.insert(1, root_dir)

# libraries
from aquaculture.common import *

# skip warning
warnings.filterwarnings("ignore")

script_date = datetime.now()
script_sdate = f'{script_date:%y%m%d_%H%M%S}_{random.randint(0, 100):02}'
data_dir = os.path.normpath(os.path.abspath(f'{root_dir}/data'))
exps_dir = os.path.normpath(os.path.abspath(f'{data_dir}/exps'))

def options_common (func):
    """
    https://stackoverflow.com/questions/5409450/can-i-combine-two-decorators-into-a-single-one-in-python
    """
    options = [
        click.option("--logs-file", type=str, default="{exps_dir}/logs/logs.txt"),
        click.option("--app-cfg", type=str, default=""),
        click.option("--evalf", type=str, multiple=True, default=['logs_file']),
        click.option("--add-evalf", type=str, multiple=True, default=[]),
        click.option("--debug", type=int, default=0)
    ]
    for option in options: func = option(func)
    return func

def process_options(ctx, params):
    # init params
    ctx.obj["params"].update(**params)
    params = ctx.obj["params"]

    for k in ['script_dir', 'script_date', 'script_sdate', 'root_dir', 'source_dir', 'data_dir', 'exps_dir']:
        if params.get(k) is None and globals().get(k) is not None: params[k] = globals()[k]

    params['evalf'] = [*list(params['evalf']), *list(params['add_evalf'])]
    parse_params(params, params['evalf'], {**globals(), **locals(), **params})

    logs_dir = os.path.dirname(params["logs_file"])
    if logs_dir != "" and os.path.exists(logs_dir) == False: os.makedirs(logs_dir)
    ctx.obj["tee_log"].append(params["logs_file"])

    print("-" * 50)
    print(f'[Command {ctx.command.name}] - {script_sdate} - {__file__}')
    print("-" * 50)
    print('params: ')
    for k in params: print(f'+ {k}: {pformat(params[k])}')
    print("-" * 50)

    if params["debug"] == 1: raise SystemExit(f'Exit Debug Options')
    globals().update(**locals())
    return params

@click.group(invoke_without_command=True)
@options_common
@click.pass_context
def main(ctx, **params):
    ctx.ensure_object(dict)
    pass

def process_main(**kwargs):
    globals().update(**kwargs.get("global_scope", {}))
    try:
        ctx_obj = {"tee_log": None, "params": {}, 'global_scope': globals()}
        with TeeLog() as tee_log:
            ctx_obj["tee_log"] = tee_log
            main(obj=ctx_obj)
    except SystemExit as ex:
        with TeeLog(ctx_obj["params"].get("logs_file"), "at") as tee_log:
            print(f'\n[Program Exit]\n+ Exit Code: {ex}')
    pass

@main.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), name='test')
@options_common
@click.pass_context
def main_test(ctx, **params):
    params = process_options(ctx, params)
    print("Test Method!")
    globals().update(**locals())
    pass

@main.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), name='app2')
@options_common
@click.pass_context
@click.option("--app-type", type=str, default="dash")
def main_app2(ctx, **params):
    params = process_options(ctx, params)

    print('Import [app_v2]')
    from aquaculture.app_v2 import main as main_app

    print(f'Run [{params["app_type"]}]')
    if params["app_type"] == "web":
        dash_web(main_app.app, ctx.obj["global_scope"])
    elif params["app_type"] == "dash":
        dash_main(main_app.app, ctx.obj["global_scope"])
    elif params["app_type"] == "app":
        ctx.obj["global_scope"]["window_title"] = main_app.app_title
        ctx.obj["global_scope"]["icon_path"] = f'{main_app.assets_dir}/favicon.ico'
        dash_qt(main_app.app, ctx.obj["global_scope"])
    globals().update(**locals())
    pass

@main.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), name='app')
@options_common
@click.pass_context
@click.option("--app-type", type=str, default="dash")
def main_app(ctx, **params):
    params = process_options(ctx, params)

    print('Import [app_v2]')
    from aquaculture.apps import main
    main_app = main.MainApp()

    print(f'Run [{params["app_type"]}]')
    if params["app_type"] == "web":
        dash_web(main_app.app, ctx.obj["global_scope"])
    elif params["app_type"] == "dash":
        dash_main(main_app.app, ctx.obj["global_scope"])
    elif params["app_type"] == "app":
        # ctx.obj["global_scope"]["window_title"] = main_app.app_title
        # ctx.obj["global_scope"]["icon_path"] = f'{main_app.assets_dir}/favicon.ico'
        dash_qt(main_app.app, ctx.obj["global_scope"])
    globals().update(**locals())
    pass

if __name__ == "__main__":
    process_main(global_scope = globals())
    pass # __name__