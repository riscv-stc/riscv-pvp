#!/usr/bin/env python3

import os
import numpy as np
import jax.numpy as jnp
from multiprocessing import Pool, Manager, Condition, Value, Process,TimeoutError
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import sys
import io
import time
import re
import traceback
import subprocess
from multiprocessing.managers import BaseManager
from multiprocessing import Queue as mpQueue
from queue import Queue
from .isa import *
from .utils import *

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


def sync_variable():
    # to synchronize the runner processes with the main process
    globals()["manager"] = Manager()
    globals()["result_dict"] = manager.dict()
    globals()["result_condition"] = Condition()
    globals()["result_detail_dict"] = manager.dict()
    globals()["tests"] = Value('L', 0)
    globals()["fails"] = Value('L', 0)

# run failed cases last time
def get_retry_cases():
    print("retry last failed cases...")
    if os.access('log/runner_report.log', os.R_OK):
        with open('log/runner_report.log') as fp:
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

# get cases from arguments
def get_arg_cases(args_cases):
    s = lambda l: l.strip()
    f = lambda l: l != '' and not l.startswith('#')
    if os.access(args_cases, os.R_OK):
        with open(args_cases) as fp:
            cases = list(filter(f, map(s, fp.read().splitlines())))
    elif args_cases != '':
        cases = list(filter(f, map(s, args_cases.split(','))))
    else:
        cases = []   
    
    return cases

def get_generator_case():
    with open("log/generator_case.log") as fp:
        s = lambda l: l.strip()
        f = lambda l: l != '' and not l.startswith('#')
        generator_info_list = list(filter(f, map(s, fp.read().splitlines())))
        generator_case_list = []
        generator_num_list = []
        for no in range(len(generator_info_list)):
            [case_name, case_num] = re.split(r'\s*,\s*', generator_info_list[no])
            generator_case_list.append(case_name)
            generator_num_list.append(int(case_num))  

    return [generator_case_list, generator_num_list]  

def select_run_case( generator_case_list, generator_num_list, cases ):
    total_num = 0
    run_case_list = []

    if len(cases) > 0:
        for no in range(len(generator_case_list)):
            case_name = generator_case_list[no]
            for case in cases:
                if not case in case_name:
                    continue

                run_case_list.append(case_name)
                total_num += generator_num_list[no]
                break
    else:
        run_case_list = generator_case_list
        total_num = sum(generator_num_list)

    return [run_case_list, total_num]

def process_bar_setup( total_num ):
    # progress bar configurations
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
    task_id = progress.add_task("runner", name = "runner", total=total_num, start=True) 

    return [progress, task_id]

def runner_error(case):
    with result_condition:
        result_dict[case] = "python run failed."
        result_detail_dict[case] = ''
        with open(f'build/{case}/runner.log', 'w') as f:
            f.write( result_dict[case] + '\n' + result_detail_dict[case] + '\n' )       

def runner_callback(progress, task_id, completed, total):
    progress.update( task_id, completed = completed )

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

def gen_runner_report( ps, args, generator_case_list, generator_num_list ):

    failed_num = 0

    # save the runner result into the log file
    if args.append:
        write_mode = 'a+'
    else:
        write_mode = 'w'
    report_path = f'log/runner_report{args.worker_job_name}.log'
    report = open(report_path, write_mode)
    for case, p in ps:
        ok = True

        p_str = p.get().getvalue()            
        # find case result in result_dict
        if result_dict[case] != "ok":
            reason = result_dict[case]
            ok = False
        if p_str != '':    
            with open(f'build/{case}/runner.log', 'w') as f:
                f.write(p_str)  

        if not ok:
            failed_num += 1
            if args.failing_info:                    
                time.sleep(0.5)
                print(f'FAIL {case} - {reason}')
            
            report.write(f'FAIL {case} - {reason}\n')
        else:
            report.write(f'PASS {case}\n')

    report.close()

    return failed_num    

# the main entrance of the runner process, including run in simulators and check the data
def run_test(case, args):
    try:
        stdout = sys.stdout
        stderr = sys.stderr
        output = io.StringIO()
        sys.stdout = output
        sys.stderr = output

        check_golden = f'build/{case}/check_golden.npy'                  

        # get the cases list in the case, including test_num, name, check string, golden
        case_list = np.load( check_golden, allow_pickle=True )

        case_info_list = []
        for test_case in case_list:

            test_case_dict = dict()

            if args.riscv_dv != True:
                for key, value in test_case["params"].items():
                    if key.endswith('_data') and key.replace('data', 'dtype') in test_case["params"].keys():
                        test_case_dict[key.replace('_data','')] = copy_to_dtype( value, eval(f"jnp.{test_case['params'][key.replace('data', 'dtype')]}") )
                    elif key.endswith('_dtype') and key.replace('dtype', 'data') in test_case["params"].keys():
                        continue
                    else:
                        if value == 'bfloat16':
                            test_case_dict[ key ] = jnp.bfloat16
                        else:
                            test_case_dict[ key ] = value

            param_str = ','.join( f'{key}=test_case_dict["{key}"]' for key in test_case_dict.keys() )
            test_case_dict['inst'] = test_case["inst"]

            if args.riscv_dv != True:
                inst = eval( f'{test_case_dict["inst"]}({param_str})' )
                test_case_dict['golden'] = inst.golden()

            test_case_dict['no'] = test_case["no"]
            test_case_dict['name'] = test_case["name"]
            test_case_dict['rule'] = test_case["rule"]
            test_case_dict['rule_params'] = test_case["rule_params"]

            case_info_list.append( test_case_dict )

        param_str = '( args=args, case=case, case_info_list=case_info_list )'
        [ test_result, test_detail, failed_num ] = eval( case_info_list[0]['rule'] + param_str, globals(), locals() )           


        with result_condition:           
            result_dict[case] = test_result
            result_detail_dict[case] = test_detail
            fails.value += failed_num
            tests.value += len(case_list)
            with open(f'build/{case}/runner.log', 'w') as f:
                f.write( result_dict[case] + '\n' + result_detail_dict[case] + '\n' )     

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
        with open(f'build/{case}/runner.log', 'w') as f:
            f.write( result_dict[case] + '\n' + result_detail_dict[case] + '\n' )

        return output

def bsub_run(cmd,port,job_name,queue_hosts):
    os.system( f'bsub -Ip -J {job_name} -m "{queue_hosts}" -q normal {cmd} --worker true --worker_port {port} -wjn {job_name}' )

def hosts_bqueues(queue):
    while True:
        try: 
            result = subprocess.check_output( 'bqueues -l', shell=True, stderr=subprocess.STDOUT, encoding='utf-8' )
        except subprocess.CalledProcessError:
            continue

        if f"QUEUE: {queue}" in result:
            break

    result_list = result.split('\n')
    no = 0
    queue_flag = False
    while True:
        if result_list[no].startswith( f"QUEUE: {queue}" ):
            queue_flag = True
        if queue_flag and result_list[no].startswith( "HOSTS" ):            
            hosts_str = result_list[no]
            hosts_str = hosts_str.replace('HOSTS:  ', '')
            break
        no += 1
        if no == len(result_list):
            hosts_str = ''
            break
    return hosts_str

def host_bjobs(job_name):
    while True:
        try:
            result = subprocess.check_output( f'bjobs -a -w | grep {job_name}', shell=True, stderr=subprocess.STDOUT, encoding='utf-8' )
        except subprocess.CalledProcessError as e:
            result = e.output       # Output generated before error          
            code      = e.returncode   # Return code             
            continue

        result_list = result.split(' ')
        if result_list[4] == 'RUN':
            host = result_list[13]
            break
    
    return host

def merge_runner_report():
    files = os.listdir('log')
    with open('log/runner_report.log', 'w') as report:
        for file in files:
            if file.startswith( 'runner_report' ):
                with open(f'log/{file}', 'r') as f_read:
                    report.write(f_read.read())
                os.remove( f'log/{file}' )

def worker_runner_callback( tests, tests_done_q ):
    # transfer completed tests number to worker process
    tests_done_q.put( tests )

# create workers in different hosts
def create_workers( cmd, batch, port ):
    queue_hosts = hosts_bqueues('normal').split(' ')
    for i in range(batch):
        job_name = str(port) + '_' + str(i)
        p = Process( target = bsub_run, args=(cmd,port,job_name, ' '.join(queue_hosts)) )
        p.daemon = True # when its parent process terminates, this process terminates, too.
        p.start()
        # to know which host this job use
        host =  host_bjobs( job_name ) 
        # remove the host to make sure next job will use another host       
        queue_hosts.remove(host)

def main(args, tests_done_q=None):

    try:
        if not args.worker:
            if args.retry:
                cases = get_retry_cases()
            else:
                cases = get_arg_cases(args.cases)
            
            if not args.append: # runner process of worker needn't print this information.
                print("looking for the cases...")

            [generator_case_list, generator_num_list] = get_generator_case()

            [run_case_list, total_num] = select_run_case( generator_case_list, generator_num_list, cases )

            # server process, if args.batch == 0, don't use lsf cluster
            if args.batch == 0:

                # define some global sync variables to synchronize the runner processes with the main process
                sync_variable()

                if not args.append: # runner process of worker needn't process bar.
                    [progress, task_id] = process_bar_setup( total_num )
                
                ps = []
                with Pool(processes=args.nproc, maxtasksperchild=1) as pool:
                    # FIXME It's better to hidden --worker/append/worker_port/worker_job_name from users, but now we haven't found methods to do that.
                    if args.append:
                        # runner process of worker needn't process bar.
                        for case in run_case_list:
                            abortable_func = partial(abortable_worker, run_test, timeout=args.timeout)
                            res = pool.apply_async(abortable_func, [ case, args ], 
                            callback=lambda _: worker_runner_callback( tests.value, tests_done_q ), 
                            error_callback=lambda _: runner_error(case)  )
                            ps.append((case, res))                        

                    else:
                        global tests
                        for case in run_case_list:
                            abortable_func = partial(abortable_worker, run_test, timeout=args.timeout)
                            res = pool.apply_async(abortable_func, [ case, args ], 
                            callback=lambda _: runner_callback( progress, task_id, tests.value, total_num ), 
                            error_callback=lambda _: runner_error(case)  )
                            ps.append((case, res))              

                    
                    failed_num = gen_runner_report( ps, args, generator_case_list, generator_num_list )

                    if not args.append: # runner process of worker needn't process bar.
                        progress.stop()

                    # spike may make that user can't input in command line, use stty sane to fix that.
                    os.system("stty sane")

                    if args.append:
                        global fails
                        # transfer test results to worker process
                        tests_done_q.put( [ len(ps), failed_num, tests.value, fails.value ] )
                        return

                    else:
                        if failed_num == 0:
                            print(f'{len(ps)} files running finish, all pass.( {tests.value} tests )')
                            sys.exit(0)
                        else:
                            if args.failing_info:
                                print(f'{len(ps)} files running finish, {failed_num} failed.( {tests.value} tests, {fails.value} failed.)')                     
                            else:
                                if args.timeout == None:
                                    print(f'{len(ps)} files running finish, {failed_num} failed.( {tests.value} tests, {fails.value} failed, please look at the log/runner_report.log for the failing information. )')             
                                else:
                                    print(f'{len(ps)} files running finish, {failed_num} failed.( please look at the log/runner_report.log for the failing information. )')             
                            
                            sys.exit(-1) 
            else:
                # dispatcher
                if total_num != 0:
                    if os.path.exists('log/runner_report.log'):                           
                        os.remove( 'log/runner_report.log' )

                [progress, task_id] = process_bar_setup( total_num )

                # lsf server
                class QueueManager(BaseManager):
                    pass 
                case_q = Queue()
                done_q = Queue() 

                QueueManager.register( 'get_case_queue', callable=lambda:case_q )
                QueueManager.register( 'get_done_queue', callable=lambda:done_q )

                port = 5000
                while True:
                    try:
                        m = QueueManager( address=('',port), authkey=b'123456' )
                        m.start()
                        break
                    except OSError:
                        # sometimes port has been used, that will raise OSError. So just pick another port.
                        port += 1
                        continue

                # a queue to dispatch cases and a queue to know workers done
                case_q = m.get_case_queue()
                done_q = m.get_done_queue()  

                print(f"{total_num} cases need to dispatch...")

                for i in range(len(run_case_list)):
                    case_q.put( run_case_list[i] )
                
                # create workers process in different host
                cmd = ' '.join(sys.argv)                
                p = Process( target=create_workers, args=(cmd, args.batch, port)  )
                p.start()

                # get result info from workers
                files_dict = dict()
                failed_files_dict = dict()
                tests_dict = dict()
                fails_dict = dict()
                done_workers = 0
                while True:

                    res_str = done_q.get()
                    if res_str == 'done':
                        done_workers += 1
                        if done_workers == args.batch:
                            progress.stop()
                            break
                        else:
                            continue

                    res_strs = res_str.split('--')

                    if len( res_strs ) == 2:
                        # use tests_done info to update progress bar
                        tests_dict[ res_strs[1] ] = int( res_strs[0].replace('tests_done', '') )
                        tests_sum = sum( tests_dict.values() )
                        progress.update( task_id, completed = tests_sum )

                    elif len( res_strs ) == 5:
                        # update more detail info when a runner process finished
                        files_dict[ res_strs[4] ] = int( res_strs[0].replace('files', '') )
                        failed_files_dict[ res_strs[4] ] = int( res_strs[1].replace('failed_files', '') )
                        tests_dict[ res_strs[4] ] = int( res_strs[2].replace('tests', '') )
                        fails_dict[ res_strs[4] ] = int( res_strs[3].replace('fails', '') )
                        if sum(files_dict.values()) == len(run_case_list):
                            progress.stop()
                            break
                        


                if p.is_alive():
                    # terminate worker create process
                    p.terminate()


                files_sum = sum( files_dict.values() )
                failed_files_sum = sum( failed_files_dict.values() )
                tests_sum = sum( tests_dict.values() )
                fails_sum = sum( fails_dict.values() )

                # merge all runner reports by workers to log/runner_report.log 
                merge_runner_report()

                m.shutdown()

                os.system("sleep 1")
                os.system("stty sane")
                
                print('', end='\r') # the stop function of progress bar make there are some null characters in command line, so take the cursor back to start.
                if fails_sum == 0:
                    print(f'{files_sum} files running finish, all pass.( {tests_sum} tests )')
                    sys.exit(0)
                else:
                    if args.failing_info:
                        print(f'{files_sum} files running finish, {failed_files_sum} failed.( {tests_sum} tests, {fails_sum} failed.)')                    
                    else:
                        print(f'{files_sum} files running finish, {failed_files_sum} failed.( {tests_sum} tests, {fails_sum} failed, please look at the log/runner_report.log for the failing information. )')             
                    
                    sys.exit(-1)                 

        else:
            #worker
            class QueueManager(BaseManager):
                pass
            QueueManager.register('get_case_queue')
            QueueManager.register('get_done_queue')    
            m = QueueManager( address=('bjsw-expsrv01', args.worker_port), authkey=b'123456' )
            try:
                m.connect()
            except ConnectionRefusedError:
                return
            case_q = m.get_case_queue()
            done_q = m.get_done_queue()

            my_name = os.popen("hostname").read()
            nproc = subprocess.check_output( 'nproc', encoding='utf-8' )
            cmd = ' '.join(sys.argv)
            cases_str = ''
            cases_num = 0

            # update args values in runner process
            orig_args = args
            orig_args.retry = False
            orig_args.nproc = int( nproc )
            orig_args.batch = 0
            orig_args.worker = False
            orig_args.append = True 

            # queue and variables to get and keep runner info
            tests_done_q = mpQueue()
            tests_done_last = 0
            files_last = 0
            failed_files_last = 0
            tests_last = 0
            fails_last = 0

            while True:
                if case_q.empty():
                    break

                case_str = case_q.get()

                if cases_num == 0:
                    cases_str += case_str
                else:
                    cases_str += ',' + case_str

                cases_num += 1

                if cases_num == int(nproc):
                    orig_args.cases = cases_str
                    # call main function as runner process
                    p = Process( target=main, args=(orig_args, tests_done_q) )
                    p.start()
                                        
                    while True:
                        q_content = tests_done_q.get()
                        if isinstance( q_content, list ):
                            break
                        tests_done = tests_done_last + q_content
                        done_q.put( f'tests_done{tests_done}--{my_name}' )

                    [files, failed_files, tests, fails] = q_content
                    files_last += files
                    failed_files_last += failed_files
                    tests_last += tests
                    tests_done_last += tests
                    fails_last += fails                  
                    done_q.put( f'files{files_last}--failed_files{failed_files_last}--tests{tests_last}--fails{fails_last}--{my_name}' )

                    cases_str = ''
                    cases_num = 0

            if cases_num != 0:
                orig_args.cases = cases_str
                p = Process( target=main, args=(orig_args, tests_done_q) )
                p.start()
                                    
                while True:
                    q_content = tests_done_q.get()
                    if isinstance( q_content, list ):
                        break
                    tests_done = tests_done_last + q_content
                    done_q.put( f'tests_done{tests_done}--{my_name}' )

                [files, failed_files, tests, fails] = q_content
                files_last += files
                failed_files_last += failed_files
                tests_last += tests
                fails_last += fails                  
                done_q.put( f'files{files_last}--failed_files{failed_files_last}--tests{tests_last}--fails{fails_last}--{my_name}' )

            done_q.put("done")
                       
    
    except KeyboardInterrupt:
        
        if 'pool' in locals():
            pool.close()
            pool.join()
        
        if 'progress' in locals():
            progress.stop()
        
        print("Catch KeyboardInterrupt!")
        os.system("stty sane")
        sys.exit(-1)    



if __name__ == "__main__":
    main()
