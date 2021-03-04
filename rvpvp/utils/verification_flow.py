
from .simulator import *
from .check import *
import subprocess

def co_verification( args, case, case_info_list ):
    # file information
    binary = f'build/{case}/test.elf'
    run_mem = f'build/{case}/run.mem'
    run_log = f'build/{case}/spike.log'
    res_file = f'build/{case}/spike.sig'

    # run elf in spike to check if the elf is right
    ret = spike_run(args, run_mem, binary, run_log, res_file)
    if ret != 0:
        # if failed, set the result of every case as spike-run, means failed when run in spike
        # then return, stop testing this case
        result = "spike_run"
        result_detail = f'\nspike-run failed!!!\nPlease check the spike log in {run_log} '
        return [ result, result_detail, len(case_info_list) ]
    
    # use these two variables to keep test info for this case
    test_result = ''
    test_detail = ''

    # use this to count failed subcases in this case
    failed_case_list = []


    # check the golden result computed by python with the spike result
    spike_result = dict()
    start_dict = dict()
    read_elf_cmd = args.config['compile']['readelf'] + ' -s ' + binary
    try:
        addr_begin_sig_str =  str( subprocess.check_output( read_elf_cmd + ' | grep begin_signature ', shell=True ), encoding = 'utf-8' )
        addr_begin_sig = int( addr_begin_sig_str.split()[1], 16 )
        flag_begin_sig = True
    except:
        flag_begin_sig = False


        # when no check configurations, no need to get result
    for test_case in case_info_list:
        if test_case["rule_params"] != '':
            check_str = test_case["rule_params"]
            if flag_begin_sig:
                try:
                    addr_testdata_str =  str( subprocess.check_output( read_elf_cmd + f' | grep test_{test_case["no"]}_data ', shell=True ), encoding = 'utf-8' )
                    addr_testdata = int( addr_testdata_str.split()[1], 16 )
                except:
                    test_result += test_case["name"]+f'_faild_find_{test_case["no"]}_test_data-'
                    test_detail += f"Can't find symbol test_{test_case['no']}_data, please check build/{case}/test.map.\n"
                    failed_case_list.append(test_case["name"])
                    continue

                golden = test_case["golden"]

                #because many subcases in one signature file, so we need the spike_start to know where to find the result
                result = from_txt( res_file, golden,  addr_testdata - addr_begin_sig )
                start_dict[test_case["name"]] = addr_testdata - addr_begin_sig
                spike_result[test_case["name"]] = result

                #save the python golden result and spike result into check.data file of each case        
                os.makedirs(f'build/{test_case["name"]}', exist_ok=True)

                check_result = check_to_txt( golden, result, f'build/{test_case["name"]}/check.data', check_str )
                if not check_result:
                    # if check failed, set result as "check failed", because the elf can be run in more sims, so don't use result_dict and notify result_condition
                    test_result += test_case["name"]+"_check failed-"
                    test_detail += f'The python golden data and spike results of test case {test_case["no"]} in build/{case}/test.S check failed. You can find the data in build/{test_case["name"]}/check.data\n'
                    failed_case_list.append(test_case["name"])                         

            else:
                test_result += test_case["name"]+"_faild_find_begin_signature"
                test_detail += f"Can't find symbol begin_signature, please check build/{case}/test.map.\n"
                failed_case_list.append(test_case["name"])                    

    # run case in more simulators and compare simulator results with spike results, which need to be same
    sims_result = sims_run( args, f'build/{case}', binary )
    for sim in [ "vcs", "verilator", "gem5" ]:
        if args.config[sim]['path'] == None:
            # don't set the path of sim, so dont't run it and needn't judge the result
            continue

        if sims_result[sim] != 0:
            # sim run failed                       
            # because the elf maybe can be run in more sims, so don't use result_dict and notify result_condition                                                            
            test_result += sim + "_failed-"
            test_detail += f'{binary} runned unsuccessfully in {sim}, please check build/{case}/{sim}.log\n'
            failed_case_list = case_info_list

        else:
            if test_case["rule_params"]!= '':
                # sim run successfully, so we compare the sim results with spike results               
                for test_case in case_info_list:
                
                    golden = test_case["golden"]
                    # get sim result, because many cases in one signature file, so we need the start to know where to find the result
                    if test_case["name"] in start_dict.keys():
                        result = from_txt( f'build/{case}/{sim}.sig', golden,  start_dict[test_case["name"]] )
                    else:
                        test_result += test_case["name"] + '_' + sim + f"_failed_find_{test_case['no']}_start-"
                        test_detail += f"Can't find test case {test_case['no']} start addr computed when check golden and spike result in build/{case}/test.S, please verify that.\n"
                        # maybe check failed or other sim failed either so we have this judge s                            
                        if test_case["name"] not in failed_case_list:
                            failed_case_list.append(test_case["name"]) 
                        continue                           

                    # save the spike result and sim result into diff-sim.data
                    os.makedirs(f'build/{test_case["name"]}', exist_ok=True)  
                    diff_result = diff_to_txt( spike_result[test_case["name"]], result, f'build/{test_case["name"]}/diff-{sim}.data', "spike", sim )

                    if not diff_result:
                        # if spike result don't equal with sim result, diff failed, write 'sim_diff failed' to test_result
                        test_result += test_case["name"] + '_' + sim + "_diff failed-"
                        test_detail += f'The results of spike and {sim} of test case {test_case["no"]}in build/{case}/test.S check failed. You can find the data in build/{test_case["name"]}/diff-{sim}.data\n'
                        # maybe check failed or other sim failed either so we have this judge                             
                        if test_case["name"] not in failed_case_list:
                            failed_case_list.append(test_case["name"])

    if test_result == '':
        return [ "ok", '', 0 ]
    else:
        return [ test_result, test_detail, len( failed_case_list )]

def riscv_dv_simulation( args, case, case_info_list ):
    # file information
    case_name = re.search('\/(\w+)$', case)
    case_name = case_name.group(1)
    binary = f'build/{case}/{case_name}.o'
    run_mem = f'build/{case}/run.mem'
    run_log = f'build/{case}/spike.log'
    res_file = f'build/{case}/spike.sig'

    # run elf in spike to check if the elf is right
    ret = spike_run(args, run_mem, binary, run_log, res_file)
    if ret != 0:
        # if failed, set the result of every case as spike-run, means failed when run in spike
        # then return, stop testing this case
        result = "spike_run"
        result_detail = f'\nspike-run failed!!!\nPlease check the spike log in {run_log} '
        return [ result, result_detail, len(case_info_list) ]
    
    # use these two variables to keep test info for this case
    test_result = ''
    test_detail = ''

    # use this to count failed subcases in this case
    failed_case_list = []


    # check the golden result computed by python with the spike result
    #spike_result = dict()
    #start_dict = dict()
    #read_elf_cmd = args.config['compile']['readelf'] + ' -s ' + binary
    #try:
    #    addr_begin_sig_str =  str( subprocess.check_output( read_elf_cmd + ' | grep begin_signature ', shell=True ), encoding = 'utf-8' )
    #    addr_begin_sig = int( addr_begin_sig_str.split()[1], 16 )
    #    flag_begin_sig = True
    #except:
    #    flag_begin_sig = False


    #    # when no check configurations, no need to get result
    #for test_case in case_info_list:
    #    if test_case["rule_params"] != '':
    #        check_str = test_case["rule_params"]
    #        if flag_begin_sig:
    #            try:
    #                addr_testdata_str =  str( subprocess.check_output( read_elf_cmd + f' | grep test_{test_case["no"]}_data ', shell=True ), encoding = 'utf-8' )
    #                addr_testdata = int( addr_testdata_str.split()[1], 16 )
    #            except:
    #                test_result += test_case["name"]+f'_faild_find_{test_case["no"]}_test_data-'
    #                test_detail += f"Can't find symbol test_{test_case['no']}_data, please check build/{case}/test.map.\n"
    #                failed_case_list.append(test_case["name"])
    #                continue

    #            #golden = test_case["golden"]

    #            #because many subcases in one signature file, so we need the spike_start to know where to find the result
    #            result = from_txt( res_file, golden,  addr_testdata - addr_begin_sig )
    #            start_dict[test_case["name"]] = addr_testdata - addr_begin_sig
    #            spike_result[test_case["name"]] = result

    #            #save the python golden result and spike result into check.data file of each case        
    #            os.makedirs(f'build/{test_case["name"]}', exist_ok=True)

    #            check_result = check_to_txt( golden, result, f'build/{test_case["name"]}/check.data', check_str )
    #            if not check_result:
    #                # if check failed, set result as "check failed", because the elf can be run in more sims, so don't use result_dict and notify result_condition
    #                test_result += test_case["name"]+"_check failed-"
    #                test_detail += f'The python golden data and spike results of test case {test_case["no"]} in build/{case}/test.S check failed. You can find the data in build/{test_case["name"]}/check.data\n'
    #                failed_case_list.append(test_case["name"])                         

    #        else:
    #            test_result += test_case["name"]+"_faild_find_begin_signature"
    #            test_detail += f"Can't find symbol begin_signature, please check build/{case}/test.map.\n"
    #            failed_case_list.append(test_case["name"])                    

    ## run case in more simulators and compare simulator results with spike results, which need to be same
    sims_result = sims_run( args, f'build/{case}', binary )
    for sim in [ "vcs", "verilator", "gem5" ]:
        if args.config[sim]['path'] == None:
            # don't set the path of sim, so dont't run it and needn't judge the result
            continue

        if sims_result[sim] != 0:
            # sim run failed                       
            # because the elf maybe can be run in more sims, so don't use result_dict and notify result_condition                                                            
            test_result += sim + "_failed-"
            test_detail += f'{binary} runned unsuccessfully in {sim}, please check build/{case}/{sim}.log\n'
            failed_case_list = case_info_list

        else:
            stem = os.getenv('STEM')
            os.system("perl " + stem + "/scripts/diff_log.pl" + " -r " + sim + " -p " + "build/" + case + " > build/" + case + "/check.log")
        #else:
        #    if test_case["rule_params"]!= '':
        #        # sim run successfully, so we compare the sim results with spike results               
        #        for test_case in case_info_list:
        #        
        #            golden = test_case["golden"]
        #            # get sim result, because many cases in one signature file, so we need the start to know where to find the result
        #            if test_case["name"] in start_dict.keys():
        #                result = from_txt( f'build/{case}/{sim}.sig', golden,  start_dict[test_case["name"]] )
        #            else:
        #                test_result += test_case["name"] + '_' + sim + f"_failed_find_{test_case['no']}_start-"
        #                test_detail += f"Can't find test case {test_case['no']} start addr computed when check golden and spike result in build/{case}/test.S, please verify that.\n"
        #                # maybe check failed or other sim failed either so we have this judge s                            
        #                if test_case["name"] not in failed_case_list:
        #                    failed_case_list.append(test_case["name"]) 
        #                continue                           

        #            # save the spike result and sim result into diff-sim.data
        #            os.makedirs(f'build/{test_case["name"]}', exist_ok=True)  
        #            diff_result = diff_to_txt( spike_result[test_case["name"]], result, f'build/{test_case["name"]}/diff-{sim}.data', "spike", sim )

        #            if not diff_result:
        #                # if spike result don't equal with sim result, diff failed, write 'sim_diff failed' to test_result
        #                test_result += test_case["name"] + '_' + sim + "_diff failed-"
        #                test_detail += f'The results of spike and {sim} of test case {test_case["no"]}in build/{case}/test.S check failed. You can find the data in build/{test_case["name"]}/diff-{sim}.data\n'
        #                # maybe check failed or other sim failed either so we have this judge                             
        #                if test_case["name"] not in failed_case_list:
        #                    failed_case_list.append(test_case["name"])

    if test_result == '':
        return [ "ok", '', 0 ]
    else:
        return [ test_result, test_detail, len( failed_case_list )]
