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
@click.option("--logs_file", type=str, default="{root_dir}/data/exps/logs/app2.txt")
@click.option("--add-evalf", type=str, multiple=True, default=["logs_file"])
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
@click.option("--logs_file", type=str, default="{root_dir}/data/exps/logs/app.txt")
@click.option("--add-evalf", type=str, multiple=True, default=["logs_file"])
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

@main.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), name='index')
@click.option("--dataset-dir", type=str, default="{root_dir}/data/a2i_data")
@click.option("--preprocess-dir", type=str, default="{root_dir}/data/preprocessed")
@click.option("--has-save", type=bool, default=False)
@options_common
@click.pass_context
@click.option("--logs_file", type=str, default="{root_dir}/data/exps/logs/index.txt")
@click.option("--add-evalf", type=str, multiple=True, default=["logs_file", "dataset_dir", "preprocess_dir"])
def main_index(ctx, **params):
    params = process_options(ctx, params)
    print('Export Data Information')
    print(f'+ Dataset dir: {params["dataset_dir"]}')
    print(f'+ Preprocessed dir: {params["preprocess_dir"]}')
    print(f'+ Has Save: {params["has_save"]}')
    print('-' * 10)

    prep_root = params["preprocess_dir"]
    data_root = params["dataset_dir"]
    has_save  = params["has_save"]
    if os.path.exists(prep_root) == False: os.makedirs(prep_root)

    # find and index all csv, image files
    print("\n[Find and index all csv, image files]")
    csv_files = [os.path.relpath(sfile, start=data_root) for sfile in glob.glob(f'{data_root}/csv/*/*.csv')]
    csv_files.sort()
    cell_files = [os.path.relpath(sfile, start=data_root) for sfile in glob.glob(f'{data_root}/*/*/*/*.*')]
    cell_files.sort()
    cell_files = np.array(cell_files)
    csv_files = np.array(csv_files)

    if has_save:
        print(f'Save {prep_root}/csv_files.txt')
        np.savetxt(f'{prep_root}/csv_files.txt', csv_files, fmt="%s")
        print(f'Save {prep_root}/cell_files.txt')
        np.savetxt(f'{prep_root}/cell_files.txt', cell_files, fmt="%s")

    # build pandas data
    print("\n[Process cell files]")
    df_cell = []
    for file in tqdm.tqdm(cell_files, "Process cell"):
        cell_info = {}
        file_info = file.split('/')
        file_date = file_info[1].replace('월', '/').replace('일', '')
        file_place = file_info[2]
        file_base = os.path.splitext(os.path.basename(file))[0]

        cell_info['fcode'] = file_base
        cell_info['fdate'] = file_date
        cell_info['fplace'] = file_place
        cell_info['fpath'] = file
        df_cell.append(cell_info)
        # pass for
    df_cell = pd.DataFrame(df_cell)
    if has_save:
        print(f'Save {prep_root}/cell_info.xlsx')
        df_cell.to_excel(f'{prep_root}/cell_info.xlsx', index=False)

    print("\n[Images under Microscopy]")
    n_cell_codes = len(np.unique(df_cell['fcode']))
    print(f"+ Cell codes: {n_cell_codes} / {len(df_cell)}")
    print(f"+ Cell dates: {len(np.unique(df_cell['fdate']))} - {np.unique(df_cell['fdate'])}")
    print(f"+ Cell place: {len(np.unique(df_cell['fplace']))} - {np.unique(df_cell['fplace'])}")
    display.display(df_cell)

    print("\n[Process csv files]")
    df_csv = pd.DataFrame()
    for file in tqdm.tqdm(csv_files, "Process csv"):
        file_info = file.split("/")
        file_date = file_info[1].replace('월', '/').replace('일', '')
        df_file = pd.read_csv(f'{data_root}/{file}')
        df_file = df_file[df_file.columns[:6]]
        df_file.insert(0, 'ccode', os.path.splitext(file_info[2])[0])
        df_file.insert(1, 'cdate', file_date)
        df_csv = pd.concat([df_csv, df_file])
        # pass
    if has_save:
        print(f'Save {prep_root}/csv_info.xlsx')
        df_csv.to_excel(f'{prep_root}/csv_info.xlsx', index=False)

    print("\n[Information from Sensors]")
    print(f"+ Sensors codes: {len(np.unique(df_csv['ID CODE']))} / {len(df_csv)}")
    print(f"+ Sensors files: {len(np.unique(df_csv['ccode']))}")
    print(f"+ Sensors dates: {len(np.unique(df_csv['cdate']))} - {np.unique(df_csv['cdate'])}")
    display.display(df_csv)

    print("\n[Check Data Integraty between Sensor Data and Microscopy Images]")
    list_code = {}
    for idx in tqdm.tqdm(range(len(df_cell))):
        info = df_cell.iloc[idx]
        id_code = info["fcode"]
        if list_code.get(id_code) is None:
            list_code[id_code] = {}
        list_code[id_code]["cell"] = 1
        # pass
    for idx in tqdm.tqdm(range(len(df_csv))):
        info = df_csv.iloc[idx]
        id_code = info["ID CODE"]
        if list_code.get(id_code) is None:
            list_code[id_code] = {}
        list_code[id_code]["csv"] = 1
        # pass

    id_codes = list(list_code.keys())
    id_codes.sort()
    df_checkcode = []
    for id_code in id_codes:
        info = {}
        info["id_code"] = id_code
        if list_code[id_code].get("csv") is not None:
            info["csv"] = 1
        else:
            info["csv"] = 0

        if list_code[id_code].get("cell") is not None:
            info["cell"] = 1
        else:
            info["cell"] = 0

        df_checkcode.append(info)
        # pass
    df_checkcode = pd.DataFrame(df_checkcode)
    if has_save:
        print(f'Save {prep_root}/checkcode_info.xlsx')
        df_checkcode.to_excel(f'{prep_root}/checkcode_info.xlsx', index=False)

    print("Data Integrity: ")
    print(f'+ Sensors, not cell : {len(df_checkcode.query("csv==1 and cell==0"))}')
    print(f'+ Not Sensors, cell : {len(df_checkcode.query("csv==0 and cell==1"))}')
    print(f'+ Both Sensors, cell: {len(df_checkcode.query("csv==1 and cell==1"))}')
    print(f'+ Total: {len(df_checkcode)}')
    display.display(df_checkcode.query("csv==1 and cell==0"))
    display.display(df_checkcode.query("csv==0 and cell==1"))
    display.display(df_checkcode)

    print("\n[Merge Data between Sensor Data and Microscopy Images]")
    df_full = df_csv.set_index('ID CODE', drop=False).join(df_cell.set_index('fcode', drop=False), how='outer').reset_index()
    display.display(df_full)

    print('-' * 10)
    print(f'Save {prep_root}/full_info.xlsx')
    df_full.to_excel(f'{prep_root}/full_info.xlsx', index=False)
    print(f'Save {prep_root}/full_info.hdf5')
    df_full.to_hdf(f'{prep_root}/full_info.hdf5', key = 'data')

    globals().update(**locals())
    pass

@main.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), name='detect-one')
@click.option("--dataset-dir", type=str, default="{root_dir}/data/a2i_data")
@click.option("--preprocess-dir", type=str, default="{root_dir}/data/preprocessed")
@click.option("--index-file", type=str, default="{preprocess_dir}/full_info.hdf5")
@click.option("--exps-dir", type=str, default="{root_dir}/data/exps")
@click.option("--id-code", type=str, default="2-1-1-2-2-1001-0120126")
@options_common
@click.pass_context
@click.option("--logs_file", type=str, default="{root_dir}/data/exps/logs/detect_one.txt")
@click.option("--add-evalf", type=str, multiple=True, default=["logs_file", "dataset_dir", "preprocess_dir", "index_file", "exps_dir"])
def main_detect_one(ctx, **params):
    params = process_options(ctx, params)
    print('Detect number of cells in a microscopy images')
    print(f'+ Dataset dir: {params["dataset_dir"]}')
    print(f'+ Preprocessed dir: {params["preprocess_dir"]}')
    print(f'+ Index file: {params["index_file"]}')
    print(f'+ Experiments dir: {params["exps_dir"]}')
    print('-' * 10)

    prep_root  = params["preprocess_dir"]
    data_root  = params["dataset_dir"]
    index_file = params["index_file"]
    exps_dir   = params["exps_dir"]

    ext_file = os.path.splitext(index_file)[1]
    if ext_file == ".xlsx":
        df_index = pd.read_excel(f'{prep_root}/full_info.xlsx')
    elif ext_file in [".hdf", ".hdf5"]:
        df_index = pd.read_hdf(f'{prep_root}/full_info.hdf5')


    id_code = params["id_code"]
    is_flag = True
    image_path = None
    while is_flag:
        image_info = df_index.query(f'fcode=="{id_code}"')
        if len(image_info)>0:
            image_path = f'{data_root}/{image_info["fpath"].values[0]}'
            is_flag = False
        else:
            id_code = input("Enter id-code in data (type 'exit' to quit): ")
            if id_code.lower() == 'exit':
                is_flag = False
            else:
                is_flag = True
        pass # is_flag

    if image_path is not None:
        blobs, blobs_image = detect_cells(image_path, verbose = 1)

        print(f"Image path: {os.path.relpath(image_path, start=data_root)}")
        print(f"Cell found: {len(blobs)}")

        save_dir = f'{exps_dir}/detect_one'
        if os.path.exists(save_dir) == False: os.makedirs(save_dir)
        save_file = f'{save_dir}/{id_code}.png'
        print(f'Save detect file with id-code = {id_code} at {save_file}!')
        cv2.imwrite(save_file, blobs_image[...,::-1])

        plt.imshow(blobs_image), plt.axis("off")
        plt.show()
        pass

    globals().update(**locals())
    pass

@main.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), name='detect-all')
@click.option("--dataset-dir", type=str, default="{root_dir}/data/a2i_data")
@click.option("--preprocess-dir", type=str, default="{root_dir}/data/preprocessed")
@click.option("--index-file", type=str, default="{preprocess_dir}/full_info.hdf5")
@click.option("--exps-dir", type=str, default="{root_dir}/data/exps")
@click.option("--export-dir", type=str, default="{root_dir}/aquaculture/assets/data")
@click.option("--has-save", type=bool, default=False)
@click.option("--ncells-detect", type=int, default=-1)
@options_common
@click.pass_context
@click.option("--logs_file", type=str, default="{root_dir}/data/exps/logs/detect_all.txt")
@click.option("--add-evalf", type=str, multiple=True, default=["logs_file", "dataset_dir", "export_dir", "preprocess_dir", "index_file", "exps_dir"])
def main_detect_all(ctx, **params):
    params = process_options(ctx, params)
    print('Detect number of cells in all microscopy images')
    print(f'+ Dataset dir: {params["dataset_dir"]}')
    print(f'+ Index file: {params["index_file"]}')
    print(f'+ Export dir: {params["export_dir"]}')
    print(f'+ Has Save: {params["has_save"]}')
    print(f'+ Experiments dir: {params["exps_dir"]}')
    print('-' * 10)

    prep_root = params["preprocess_dir"]
    data_root = params["dataset_dir"]
    has_save  = params["has_save"]
    index_file = params["index_file"]
    exps_dir   = params["exps_dir"]
    export_dir = params["export_dir"]

    if has_save:
        print('\n[Creat saving directory]')
        save_dir = f'{exps_dir}/detect_all'
        print(f"Create saving directory of all detecting images: [{save_dir}]")
        if os.path.exists(save_dir) == False: os.makedirs(save_dir)

    ext_file = os.path.splitext(index_file)[1]
    if ext_file == ".xlsx":
        df_index = pd.read_excel(f'{prep_root}/full_info.xlsx')
    elif ext_file in [".hdf", ".hdf5"]:
        df_index = pd.read_hdf(f'{prep_root}/full_info.hdf5')

    df_ncells = []
    df_index = df_index.iloc[:params["ncells_detect"]] if params["ncells_detect"] != -1 else df_index
    print("\n[Detect cells in microscopy images]")
    for idx in tqdm.tqdm(range(len(df_index)), "Detect cells"):
        info = df_index.iloc[idx]
        if pd.isna(info["fpath"]) == False and info["fpath"] != "":
            image_path = f'{data_root}/{info["fpath"]}'
            if has_save:
                blobs, blobs_image = detect_cells(image_path, verbose=1)
                save_file = f'{save_dir}/{info["ID CODE"]}.png'
                cv2.imwrite(save_file, blobs_image[..., ::-1])
            else:
                blobs, _ = detect_cells(image_path, verbose=0)
            n_cell = len(blobs)
            df_ncells.append(n_cell)
        else:
            df_ncells.append(-1)
        pass
    df_ncells = np.array(df_ncells)

    print('\n[Build final index data]')
    df_final = pd.DataFrame()
    for new_name, colname in zip(["ID","Date","Place","Temperature", "DO", "pH", "salinity", "NTU", "ncells", "Path"],
                                 ["ID CODE", "cdate", "fplace", "Temperatue", "DO", "pH", "salinity", "NTU", "", "fpath"]):
        if new_name == "ncells":
            df_final[new_name] = df_ncells
        else:
            df_final[new_name] = df_index[colname]

    print('\n[Save final index data]')
    if os.path.exists(export_dir) == False: os.makedirs(export_dir)
    print(f'Save final index file {export_dir}/final_info.csv')
    df_final.to_csv(f'{export_dir}/final_info.csv', index=False)

    print(f'Save final index file {prep_root}/final_info.xlsx')
    df_final.to_excel(f'{prep_root}/final_info.xlsx', index=False)

    print(f'Save final index file {prep_root}/final_info.hdf5')
    df_final.to_hdf(f'{prep_root}/final_info.hdf5', key='data')

    print(f'Save final index file {prep_root}/final_info.csv')
    df_final.to_csv(f'{prep_root}/final_info.csv', index=False)

    globals().update(**locals())
    pass

@main.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), name='data-analysis')
@click.option("--dataset-dir", type=str, default="{root_dir}/data/a2i_data")
@click.option("--preprocess-dir", type=str, default="{root_dir}/data/preprocessed")
@click.option("--index-file", type=str, default="{preprocess_dir}/full_info.hdf5")
@click.option("--final-file", type=str, default="{root_dir}/aquaculture/assets/data/final_info.csv") # {preprocess_dir}/final_info.hdf5 # {root_dir}/data/a2i_data/final_info.csv
@click.option("--exps-dir", type=str, default="{root_dir}/data/exps")
@click.option("--debug", type=bool, default=False)
@options_common
@click.pass_context
@click.option("--logs_file", type=str, default="{root_dir}/data/exps/logs/data_analysis.txt")
@click.option("--add-evalf", type=str, multiple=True, default=["logs_file", "dataset_dir", "preprocess_dir", "index_file", "final_file", "exps_dir"])
def main_data_analysis(ctx, **params):
    params = process_options(ctx, params)
    print('DATA ANALYSIS')
    print('-' * 10)

    prep_root = params["preprocess_dir"]
    data_root = params["dataset_dir"]
    index_file = params["index_file"]
    final_file = params["final_file"]
    exps_dir   = params["exps_dir"]

    df_final = pd.read_csv(final_file)
    df_final = df_final.query('ncells != -1 and DO == DO')
    df_final = df_final[df_final.columns[:-1]]

    list_places = np.unique(df_final['Place'])
    list_days = np.unique(df_final['Date'].values)

    print("Information: ")
    print(f'+ List Days: {list_days}')
    print(f'+ List Places: {list_places}')
    print()

    list_df_places = {}
    df_gday = df_final.groupby(['Date', 'Place']).mean().reset_index()
    df_gday["ncells"] = df_gday["ncells"].astype(np.int)
    for place in list_places:
        df_p = df_gday.query(f'Place=="{place}"').reset_index(drop = True)
        df_p["ncells"] = df_p["ncells"].astype(np.int)
        list_df_places[f"GroupDay_{place}"] = df_p
        list_df_places[f"{place}"] = df_final.query(f'Place=="{place}"').reset_index(drop = True)
    list_df_places["GroupDay_All"] = df_gday
    list_df_places["All"] = df_final

    save_dir = f'{exps_dir}/data_analsyis'
    if os.path.exists(save_dir) == False: os.makedirs(save_dir)

    pd.set_option('display.precision', 2)

    # for place in ["GroupDay_All"]:
    for place in list_df_places:
        df_data = list_df_places[place]
        df_data = df_data[df_data.columns[-6:]]
        # df_data[df_data.columns[3:]] = (df_data[df_data.columns[3:]] - np.mean(df_data[df_data.columns[3:]])) / np.std(df_data[df_data.columns[3:]])

        print(f'+ Place: {place}')
        print("  ** Statistic Information ** ")
        description = df_data.describe().T
        display.display(description)
        print()

        df_data.plot(kind='box', sharex=False, sharey=False, figsize=(12, 12), layout=(3, 3), subplots=True);
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{place}_boxplot.png')
        plt.show()

        df_data.plot(kind='density', sharex=False, sharey=False, figsize=(12, 12), layout=(3, 3), subplots=True);
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{place}_density.png')
        plt.show()

        df_data.hist(figsize=(12, 12), layout=(3, 3))
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{place}_hist.png')
        plt.show()

        print("  ** Data Correlation ** ")
        display.display(df_data.corr())
        print()

        plt.figure(figsize=(8, 8))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(df_data.corr(),
                    cmap=cmap, vmin=-1, vmax=1,
                    center=0, square=True,
                    linewidths=.5, annot=True, fmt=".2f")
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{place}_corr.png')
        plt.show()

        plt.figure(figsize=(12, 12))
        plt.matshow(df_data.corr(), vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(0, 6), rotation=45);
        plt.yticks(np.arange(0, 6), rotation=45);
        plt.gca().set_xticklabels(list(df_data.columns));
        plt.gca().set_yticklabels(list(df_data.columns));
        plt.savefig(f'{save_dir}/{place}_corr_s.png')
        plt.show()

        scatter_matrix(df_data, figsize=(12, 12))
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{place}_scatter.png')
        plt.show()

        print("  ** Data ** ")
        display.display(df_data)
        print()
        if params["debug"] == True: break

    globals().update(**locals())
    pass



if __name__ == "__main__":
    process_main(global_scope = globals())
    pass # __name__