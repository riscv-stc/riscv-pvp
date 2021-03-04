#!/usr/bin/env python3

from .isa import *
from .utils import *
import jax.numpy as jnp
import re
import yaml
import glob
import os
import io
import sys, inspect
from multiprocessing import Pool, TimeoutError
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import types
import inspect
import time
import traceback
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

from .common import import_from_directory

import_from_directory('isa', globals())
import_from_directory('utils', globals())

def export_global_variables(config):

    for k in config["processor"].keys():
        globals()[k] = config["processor"][k]

    globals()["readelf"] = config["compile"]['readelf']
    globals()["config"] = config

# find the params for matrix cases
def search_matrix(arg_names, vals, no, params, locals_dict, **kwargs):
    if no == len(vals):#when all arguments in kwargs
        params.append(kwargs)
        return

    merge_dict = { **locals_dict, **kwargs }
    # define the arguments in kwargs to help compute next param
    for key, val in merge_dict.items():
        if isinstance(val,str):
            exec(f'{key}="{val}"')
        elif isinstance(val, np.ndarray):
            val_str = val.tobytes()
            exec(f'{key}=np.reshape( np.frombuffer( val_str, dtype=jnp.{val.dtype}), {val.shape}) ')
        elif inspect.isfunction( merge_dict[key] ):
            exec(f'{key}=merge_dict["{key}"]')
        else:
            # exec(f'{key}={val}')
            exec(f'{key}=merge_dict["{key}"]')

    # just to unify the handle process
    if not isinstance(vals[no], list):
        vals[no] = [vals[no]]

    for val in vals[no]:
        try:
            vals_next = eval(val) if isinstance(val,str) else val
        except NameError:# when the string is jsut a string and can't be run
            vals_next = val
        
        # in case of the string is same as built-in function and type
        if isinstance(vals_next,types.BuiltinFunctionType) or inspect.isclass(vals_next):
            vals_next = val
        
        # just to unify the handle process
        if not isinstance(vals_next, list):
            vals_next=[vals_next]
        
        for val_el in vals_next:
            # take this argument into kwargs and continue to search next argument 
            kwargs[arg_names[no]] = val_el
            search_matrix(arg_names, vals, no+1, params, locals_dict, **kwargs)

    # get the params list from dictionary
    if no == 0:
        params_yml = []

        for param_dict in params:
            vals = []
            for arg in arg_names:
                vals.append(param_dict[arg])
            
            params_yml.append(vals)
        
        return params_yml

def cases_gen( args, filename, inst, cases_dict, templates, rule_dict, collected_case_list, collected_case_num_list, param_dict  ):
    test_dict = dict()

    for key, params in cases_dict.items():

        # the default input mode for cases argument is common, test_case @ a, b ,c:
        param_mode = 'common'

        # get the test_type and arguments from cases option
        [ test_type, *others ] = re.split(r'\s*@\s*', key)
        if len(others) == 2:
            _args = others[0]
            _defaults = others[1]
        elif len(others) == 1:
            _args = others[0]
            _defaults = ''
        else:
            _args = ''
            _defaults = ''

            # the matrix mode in cases
            if isinstance(params, dict) and 'matrix' in params:
                param_mode = 'matrix'

        if not test_type in templates:
            # if no template of this test_type, it's not a test case
            print(f"can't find the template code for {test_type} of {inst} in {filename}, Please check!")
            continue

        # use test_info to save the test information of one type of test cases
        test_info = dict()

        test_info["template"] = templates[test_type]

        if param_mode == 'common':
            # test_type @ xx,xx,xx@ xx=xx

            # must not use the matrix key with @ ...@ ..=..
            if isinstance(params, dict) and 'matrix' in params:
                print(f"@ argument list can't be used with matrix in params.-{test_type} of {inst} in {filename}")
                continue

            # separate the arguments into a list
            if _args:
                argnames = re.split(r'\s*,\s*', _args)
                for i in range(len(argnames)):
                    argnames[i] = argnames[i].strip()
            else:
                argnames = []

            test_info["args"] = argnames

            if isinstance(params, dict) and 'setup' in params:
                # use argument list and params_yml in setup to get the params
                if _args == '':
                    print(f"setup needs to be used with argument list, please check.-{test_type} of {inst} in {filename}")
                    continue
                
                # if there is the variable params_yml before, delete it first.
                if 'params_yml' in globals() or 'params_yml' in locals():
                    del params_yml

                if test_type in rule_dict['rule_params'] and 'random_times' in rule_dict['rule_params'][test_type]:
                    rtimes = rule_dict['rule_params'][test_type]['random_times']
                else:
                    rtimes = args.rtimes
                params_yml = []
                while rtimes >  0:
                    rtimes -= 1
                    locals_dict = dict()
                    exec(params["setup"], globals(), locals_dict)
                    if not 'params_yml' in locals_dict:
                        print(f"no params_yml in setup, please check.-{test_type} of {inst} in {filename}")
                        continue
                    else:
                        params_yml.extend( locals_dict['params_yml'] )
                # filtering cases which are no params_yml to arise compile error
                if 0 == len(params_yml): 
                    continue
            
            else:                
                if test_type in rule_dict['rule_params'] and 'random_times' in rule_dict['rule_params'][test_type]:
                    rtimes = rule_dict['rule_params'][test_type]['random_times']
                else:
                    rtimes = args.rtimes
                params_yml = []
                while rtimes >  0:
                    rtimes -= 1
                    params_yml.extend( params )

        elif param_mode == 'matrix':
            if test_type in rule_dict['rule_params'] and 'random_times' in rule_dict['rule_params'][test_type]:
                rtimes = rule_dict['rule_params'][test_type]['random_times']
            else:
                rtimes = args.rtimes
            params_yml = []
            # get argument names and params
            argnames = list(params['matrix'].keys())
            test_info["args"] = argnames
            vals = list(params['matrix'].values())            
            while rtimes >  0:
                rtimes -= 1

                locals_dict = {}
                # setup string is the preparatory work for matrix
                if 'setup' in params:
                    exec(params['setup'], globals(), locals_dict)

                                
                # compute the params values
                params_dict_yml = []            
                params_yml.extend( search_matrix(argnames, vals, 0, params_dict_yml, locals_dict) )

        else:
            continue

        if args.little:
            l = len(params_yml)                   
            if l > 4:
                params_yml = [ c for i, c in enumerate(params_yml) if i in [int(l/4), int(l*2/4), int(l*3/4), l-1] ]
        test_info["params"] = params_yml

        test_info['default'] = _defaults

        test_info['rule'] = rule_dict['rule']

        if test_type in rule_dict['rule_params']:
            test_info['rule_params'] = rule_dict['rule_params'][test_type]
        else:
            test_info['rule_params'] = ''

        if args.param_info:
            param_dict[inst][test_type] = dict()

        if args.level == "type":
            # collect the instruction and test_type
            collected_case_list.append(inst+'/'+test_type)
            collected_case_num_list.append(0)

        # compute params and set the case name      
        test_info["case_param"] = dict()
        if test_info["params"]:
            num = 0         
            for param in test_info["params"]:
                
                # compute params value
                param = eval(param) if isinstance(param, str) else param

                # set the case name
                case_name = f'test{num}_'
                for i in range(len(test_info["args"])):
                    if i != 0:
                        case_name += '-'
                    if isinstance(param[i], np.ndarray) or isinstance(param[i], tuple) or isinstance(param[i], list):
                        case_name += test_info["args"][i]
                    else:
                        case_name += test_info["args"][i] + "_" + str( param[i] )

                # if case_name too long, use first letter to replace arg name
                if len( case_name ) > 255:
                    case_name = f'test{num}_'
                    for i in range(len(test_info["args"])):
                        if i != 0:
                            case_name += '-'
                        if isinstance(param[i], np.ndarray) or isinstance(param[i], tuple) or isinstance(param[i], list):
                            case_name += test_info["args"][i][0]
                        else:
                            case_name += test_info["args"][i][0] + "_" + str( param[i] ) 

                    # if case_name too long after reduction, use test number only
                    if len( case_name ) > 255:
                        case_name = f'test{num}'
                
                test_info["case_param"][case_name] = param

                if args.level == "case":
                    collected_case_list.append(inst+'/'+test_type+'/'+case_name)
                    collected_case_num_list.append(1)
                else:
                    collected_case_num_list[-1] += 1
                
                num += 1
                if args.param_info:
                    param_dict[inst][test_type][case_name] = str(param)                
            
        else:
            if args.param_info:
                param_dict[inst][test_type] = None
            if args.level == "case":
                # if no param, just one case
                collected_case_list.append(inst+'/'+test_type)
                collected_case_num_list.append(1)
            else:
                collected_case_num_list[-1] += 1

        test_dict[test_type] = test_info
    
    return test_dict

# analyse spec file to collect tests
def analyse_spec( spec_file, args, collected_case_list, collected_case_num_list, param_dict ):

    # load information from the yml file 
    stream = open(spec_file, 'r')
    config = yaml.load(stream, Loader=yaml.SafeLoader)
    if config == None:
        return

    if args.exclude_groups and len(args.exclude_groups.split()) != 0:
        exclude_groups = args.exclude_groups.split(',')
    else:
        exclude_groups = []

    for inst, cfg in config.items(): #inst is the instruction needed to test, cfg is the env\head\template\cases\check  test configuration

        # don't handle options startswith _, which is a template for tested instruction
        if inst.startswith('_'):
            continue

        # exclude the test groups that match keywords from args.exclude_groups
        exclude_group = False
        for group_keyword in exclude_groups:
            if group_keyword in inst:
                exclude_group = True
                break

        if exclude_group:
            continue

        if args.param_info:
            param_dict[inst] = dict()
        
        # get the value of different options
        attrs = dict()
        attrs['name'] = inst
        attrs['inst'] = globals()[inst.capitalize()]
        attrs['env'] = cfg['env']
        attrs['header'] = cfg['head']
        if 'footer' in cfg:
            attrs['footer'] = cfg['footer']
        if 'tdata' in cfg:
            attrs['tdata'] = cfg['tdata']

        # collect the test configurations
        attrs['test'] = dict()

        # get the params and check string on the basis of args.basic and args.random
        if True == args.basic:
            if 'basic_cases' in cfg:
                cases_dict = cfg['basic_cases']
            else:
                continue
        elif True == args.random:
            if 'random_cases' in cfg:
                cases_dict = cfg['random_cases']                    
            else:
                continue
        else:
            if 'cases' in cfg:
                cases_dict = cfg['cases']                       
            else:
                continue            

        if 'rule' in cfg:
            rule_dict = dict()
            rule_dict["rule"] = cfg['rule']
            rule_dict["rule_params"] = cfg['rule_params'] if 'rule_params' in cfg else dict()
        else:
            print(f"no rule in {spec_file}.")
            continue           

        # take the test cases info into a dict and file to tell users
        if args.level == "group":
            # just collect the instruction
            collected_case_list.append(inst)
            collected_case_num_list.append(0) 

        attrs['test'] = cases_gen( args, spec_file, inst, cases_dict, cfg['templates'], rule_dict, collected_case_list, collected_case_num_list, param_dict )

        # define the test function to run tests later
        exec(f'def test_function(self, args, test_type, test_case): generate(self, args, test_type, test_case)')
        exec(f'attrs["test_function"] = test_function')

        # define a Test class to organize the test info for one inst
        globals()[f'Test_{inst}'] = type(f'Test_{inst}', (object,), attrs)


def collect_tests( args ):

    if not args.specs or len(args.specs.split()) == 0:
        # if no specs argument, default find the case from specs folder
        specs = [args.pkgroot + '/specs', 'specs']
    else:
        # otherwise find the case from args.specs
        specs = args.specs.split(',')

    print("collecting the tests...")

    # use this list to keep the case name and amount in different merge level(case, type, inst)
    collected_case_list = [] 
    collected_case_num_list = []

    # if need to know the detailed params from spec yaml files, use param_dict to keep that.
    param_dict = dict()

    # analyze yml file to find the test cases
    for spec in specs:
        if os.path.isdir(spec):
            spec = f'{spec}/**/*.spec.yml'
        # handle every .spec.yml file under spec folder or spec is a .spec.yml file
        for filename in glob.iglob(spec, recursive=True):

            # if dont't use --random, set random seed for each spec file to make sure params same.
            if not args.random:
                np.random.seed(args.seed)
            elif args.seed != 3428: # users can set seed for random cases for each spec file
                np.random.seed(args.seed)

            analyse_spec( filename, args, collected_case_list, collected_case_num_list, param_dict )

    os.makedirs("log", exist_ok=True)

    # log file to tell user what cases there are in the yaml files in this level
    with open("log/collected_case.log", 'w') as case_log:
        for no in range(len(collected_case_list)):
            case_log.write(collected_case_list[no])
            case_log.write(',')
            case_log.write(str(collected_case_num_list[no]))
            case_log.write('\n')


    # save cases' param into a params.yaml
    if args.param_info:
        with open("log/params.yml", 'w') as params_file:
            yaml.dump( param_dict, params_file, default_flow_style = False )

    return [collected_case_list, collected_case_num_list]
                
def get_retry_cases():
    print("retry last failed cases...")
    if os.access('log/generator_report.log', os.R_OK):
        with open('log/generator_report.log') as fp:
            cases = []
            lines = fp.read().splitlines()
            for line in lines:
                if line.startswith('PASS '):
                    continue
                line = line.replace('FAIL ', '')
                line = line.split( ' ', 1 )[0]
                cases.append( line )

            if len(cases) == 0:
                print('all pass, retry abort.')
                sys.exit(0)

            return cases
    else:
        print('could not retry without last run log.')
        sys.exit(-1)   


def get_arg_cases( arg_cases ):

    s = lambda l: l.strip()
    f = lambda l: l != '' and not l.startswith('#')
    if os.access( arg_cases, os.R_OK ):
        with open( arg_cases ) as fp:
            cases = list(filter(f, map(s, fp.read().splitlines())))
    elif arg_cases != '':
        cases = list(filter(f, map(s, arg_cases.split(','))))
    else:
        cases = []

    return cases

def select_case( collected_case_list, collected_case_num_list, cases, exclude_cases ):

    total_case_num = 0

    if cases and len(cases) > 0:
        selected_case_list = []
        for no in range(len(collected_case_list)):
            testcase = collected_case_list[no]
            for case in cases:
                if not case in testcase:
                    continue
                if case in exclude_cases:
                    continue
                selected_case_list.append(testcase)
                total_case_num += collected_case_num_list[no]
                break

        return [selected_case_list, total_case_num]                
    else:
        if exclude_cases and len(exclude_cases) > 0:
            selected_case_list = []
            for no in range(len(collected_case_list)):
                testcase = collected_case_list[no]
                exclude_case = False
                for case in exclude_cases:
                    if case in testcase:
                        exclude_case = True
                        break
                if not exclude_case:
                    selected_case_list.append(testcase)
                    total_case_num += collected_case_num_list[no]

            return [selected_case_list, total_case_num]
        else:
            total_case_num = sum(collected_case_num_list)
            return [collected_case_list, total_case_num]

def progress_bar_setup( total_case_num ):

    # progress bar configuration
    progress = Progress(
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "case_sum:",
        TextColumn("[bold red]{task.total}"),
        "elapsed:",
        TimeElapsedColumn(),
        "remaining:",
        TimeRemainingColumn()
    )

    progress.start()
    task_id = progress.add_task("generation", name = "generation", total=total_case_num, start=True) 
    return [progress, task_id]    

def generator_error(case):
    with result_condition:
        result_dict[case] = "python run failed."
        result_detail_dict[case] = ''
        os.makedirs(f'build/{case}', exist_ok=True)
        with open(f'build/{case}/generator.log', 'w') as f:
            f.write( result_dict[case] + '\n' + result_detail_dict[case] + '\n' )

def generator_callback(progress, task_id, completed, total):
    progress.update( task_id, completed = completed )

# call the test_function in the test class to generate the test
def generate_test( case, args ):
    # the work process need to handle the KeyboardInterrupt before the main process
    try:
        # redirect the standard output and standard error output to output, by this way, we have a clean output in command line
        stdout = sys.stdout
        stderr = sys.stderr
        output = io.StringIO()
        sys.stdout = output
        sys.stderr = output

        # get the test_instruction、 test_type and test_param by /
        test = case.split('/')
        if len(test) == 1:
            test_instruction = test[0]
            test_type = ''
            test_case = ''
        elif len(test) == 2:
            test_instruction = test[0]
            test_type = test[1]
            test_case = ''  
        else:
            test_instruction = test[0]
            test_type = test[1]
            test_case = test[2]              

        # use the test_function in Test class to run the test
        test_instruction_class = "Test_" + test_instruction
        tic = eval( f'{test_instruction_class}()' )
        tic.test_function( args, test_type, test_case )

        if os.path.exists(f'build/{case}'):
            os.system(f'cp -rf build/Makefile.subdir build/{case}/Makefile')

        sys.stdout = stdout
        sys.stderr = stderr

        return output
    
    except:
        if output in locals().keys():
            sys.stdout = stdout
            sys.stderr = stderr
        else:
            output = io.StringIO()

        result_dict[case] = 'python failed'

        error_output = io.StringIO()
        traceback.print_tb(sys.exc_info()[2], file=error_output)
        error_str = error_output.getvalue()
        error_str += "\nUnexpected error: " + str(sys.exc_info()[0]) + " " + str(sys.exc_info()[1])
        result_detail_dict[case] = error_str
        os.makedirs(f'build/{case}', exist_ok=True)
        with open(f'build/{case}/generator.log', 'w') as f:
            f.write( result_dict[case] + '\n' + result_detail_dict[case] + '\n' )

        # print(error_str)

        return output

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except TimeoutError:
        case = args[0]
        result_dict[case] = "TimeoutError"
        return io.StringIO()

def gen_report( ps, timeout, failing_info, collected_case_list, collected_case_num_list ):

    failed_num = 0
    # write the test results into log/generator_report.log
    report = open(f'log/generator_report.log', 'w')
    generator_case_log = open(f'log/generator_case.log', 'w')
    for case, p in ps:
        ok = True

        case_num = collected_case_num_list[ collected_case_list.index( case ) ]

        p_value = p.get().getvalue()
        # find case result in result_dict
        if result_dict[case] != "ok":
            reason = result_dict[case]
            ok = False    
        if p_value != '':
            with open(f'build/{case}/generator.log', 'w') as f:
                f.write( p_value )          

        if not ok:
            failed_num += 1
            if failing_info:
                # use the sleep to make that the main process can get the KeyboardInterrupt from ctrl C
                time.sleep(0.5)
                print(f'FAIL {case} - {reason}')
            report.write( f'FAIL {case} - {reason}\n' )
        else:
            report.write( f'PASS {case}\n' )
            generator_case_log.write( case + ',' + str(case_num) + '\n' )
                                                                    
    report.close()
    generator_case_log.close()

    return failed_num

def prepare_makefiles(args):
    config = args.config
    vmap = {
        'CC': config['compile']['cc'],
        'CFLAGS': f"{config['compile']['defines']} {config['compile']['cflags']}",
        'LDFLAGS': config['compile']['linkflags'],
        'OBJDUMP': config['compile']['objdump'],
        'READELF': config['compile']['readelf'],
        'SPIKE': config['spike']['cmd'], 'SPIKE_OPTS': '',
        'GEM5': config['gem5']['cmd'], 'GEM5_OPTS': '',
        'VCS': config['vcs']['cmd'], 'vcstimeout':config['vcs']['vcstimeout'],
        'fsdb': config['vcs']['fsdb'], 'tsiloadmem': config['vcs']['tsiloadmem'],
        'lsf': config['lsf']['enable'], 'LSF_CMD': config['lsf']['cmd']

    }

    rootdir_dict = {"case": '../../../..', 'type':'../../..', 'group': '../..'}
    vmap['ROOTDIR'] = rootdir_dict[args.level]
    vmap['PKGROOT'] = args.pkgroot

    tmproot = f'{args.pkgroot}/utils/make'

    os.system(f'cp -rf {tmproot}/spike.mk {tmproot}/*.py {tmproot}/Makefile build/')
    if config['gem5']['path'] and config['gem5']['path'] != '~':
        os.system(f'cp -rf {tmproot}/gem5.mk build/')
    else:
        if os.path.exists(f'build/gem5.mk'):
            os.remove('build/gem5.mk')
    if config['vcs']['path'] and config['vcs']['path'] != '~':
        os.system(f'cp -rf {tmproot}/vcs.mk build/')
    else:
        if os.path.exists(f'build/vcs.mk'):
            os.remove('build/vcs.mk')

    with open(f'{tmproot}/Makefile.subdir', 'r' ) as f:
        template = f.read()
        makefile_str = template.format_map(vmap)
        with open('build/Makefile.subdir', 'w') as f_ref:
            f_ref.write(makefile_str)

    with open(f'{tmproot}/variables.mk.in', 'r') as f:
        template = f.read()
        vars = template.format_map(vmap)
        with open('build/variables.mk', 'w') as fo:
            fo.write(vars)

def main(args):

    if args.riscv_dv == True:
        stem = os.getenv("STEM")
        i = 0
        test_flag = 0
        cur_target = args.target_dv
        cur_test = args.test_dv
        if cur_test != "":
            test_flag = 1
        if test_flag == 1:
            os.system("python " + stem + "/run.py --target " + cur_target + " -tn " + cur_test + " -o " + stem + "/out")
        else:
            os.system("python " + stem + "/run.py --target " + cur_target + " -o " + stem + "/out")
        compile_list_log = open(f'{stem}/out/compile_list.log', 'r')
        lines = compile_list_log.readlines()
        os.makedirs(f'log', exist_ok=True)
        generator_case_log = open(f'log/generator_case.log', 'w')
        for line in lines:
            line = line.strip()
            cur_test_tmp = re.search('(.+)\_\d+$', line)
            cur_test_tmp = cur_test_tmp.group(1)
            os.makedirs(f'build/risc_dv/{cur_target}/{cur_test_tmp}/{line}', exist_ok=True)
            os.system("cp -rf " + stem + "/out/asm_test/" + line + "\.* " + os.getcwd() + "/build/risc_dv/" + cur_target + "/" + cur_test_tmp + "/" + line)
            generator_case_log.write("risc_dv/" + cur_target + "/" +cur_test_tmp + "/" + line + ",1\n")
            workdir = f'build/risc_dv/{cur_target}/{cur_test_tmp}/{line}'
            check_golden = f'{workdir}/check_golden.npy'
            case_list = []
            num = 0
			# legacy
            # if no param, just one case
            #inst = test_inst.inst()

            #generate the code content
            #content = generate_code( test_info["template"], test_inst, inst, num+2 )
            #code += content["code"] + '\n'
            #data += content["data"] + '\n'
            #tdata += content["tdata"] + '\n'
            #footer += content["footer"] + '\n'
            case_dict =  { "no": 1, "inst": "riscv_dv", "name": workdir, "rule": "riscv_dv_simulation", "rule_params": "{}", "params": "{}"}
            #golden = inst.golden()
            #if isinstance( golden, np.ndarray ):
            #    case_dict["golden_data"] = copy_to_dtype( golden, np.uint8 )
            #    case_dict["golden_dtype"] = str( golden.dtype )
            #else:
            #    case_dict[ "golden" ] = golden            
            case_list.append( case_dict )            
            num += 1
            np.save(check_golden, case_list)
			#endlegacy
        generator_case_log.close()
    else:
        try:
            # get config information from config file, including isa options and compilation options mainly
            export_global_variables(args.config)

            # collect test cases in spec files
            [collected_case_list, collected_case_num_list] = collect_tests( args )

            # just collect cases
            if args.collect:
                print("please look at the contents in the log/collected_case.log")
                sys.exit(0)

            if args.retry:
                cases = get_retry_cases()
            else:
                cases = get_arg_cases(args.cases) 

            exclude_cases = []
            if args.exclude_cases:
                exclude_cases = get_arg_cases(args.exclude_cases)

            os.makedirs('build', exist_ok=True)

            prepare_makefiles(args)


            with Pool(processes=args.nproc, maxtasksperchild=1) as pool:
                
                [ selected_case_list, total_case_num ] = select_case( collected_case_list, collected_case_num_list, cases, exclude_cases )

                [progress, task_id] = progress_bar_setup( total_case_num )

                # use generate_test to generate testcase in process pool
                ps = []
                for case in selected_case_list:
                    abortable_func = partial(abortable_worker, generate_test, timeout=args.timeout)
                    res = pool.apply_async(abortable_func, [ case, args ], 
                    callback=lambda _: generator_callback( progress, task_id, tests.value, total_case_num), 
                    error_callback=lambda _: generator_error(case) )
                    ps.append((case, res))

                failed_num = gen_report( ps, args.timeout, args.failing_info, collected_case_list, collected_case_num_list )

                progress.stop()

                # print test result
                if failed_num == 0:
                    print(f'{len(ps)} files generation finish, all pass.( {tests.value} tests )')
                    sys.exit(0)
                else:
                    if args.failing_info:
                        print(f'{len(ps)} files generation finish, {failed_num} failed.( {tests.value} tests, {fails.value} failed.)')                    
                    else:
                        if args.timeout == None:
                            print(f'{len(ps)} files generation finish, {failed_num} failed.( {tests.value} tests, {fails.value} failed, please look at the log/generator_report.log for the failing information. )')
                        else:                     
                            print(f'{len(ps)} files generation finish, {failed_num} failed.( please look at the log/generator_report.log for the failing information. )')

                    sys.exit(-1)                        


        except KeyboardInterrupt:
            # handle the keyboardInterrupt, stop the progress bar and wait all of the processes in process  pool to stop, then exit
            if 'progress' in locals():
                progress.stop()
            if 'pool' in locals():
                pool.close()
                pool.join()
            print("Catch KeyboardInterrupt!")
            sys.exit(-1)


if __name__ == '__main__':
    main()

