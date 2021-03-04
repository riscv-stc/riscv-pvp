#!/usr/bin/env python3

import click
import textwrap
import shutil

from rvpvp import generator, runner
from rvpvp.common import Args, parse_config_files, get_package_root

@click.group()
def cli():
    """RISC-V PVP tool to manage co-verification flow."""
    pass


@cli.command()
@click.argument('dir')
def new(dir):
    """Create a new target for verification."""
    click.echo(f'New target is created in {dir}.')
    shutil.copytree(get_package_root() + '/new', dir)


@cli.command()
@click.option('--config', help='config yaml file', default='target.yml')
@click.option('--nproc', '-n', help='generate elf files on n processes', type=int, default=1)
@click.option('--level', '-l', help='level of cases to compile into one elf file',
                type=click.Choice(['group', 'type', 'case'], case_sensitive=False), default="case")
@click.option('--specs', '-s', help='test case specs')
@click.option('--cases', '-c', help=textwrap.dedent('''\
                             test case list string or file, for example:
                             - vsub_vv,addi/test_imm_op/
                             - cases.list
                             you can find more examples with option --collect'''), default='')
@click.option('--collect', help='just collect the test case to know what cases we can test', is_flag=True)
@click.option('--little', help='only run at most 4 test cases for each test type of each instruction', is_flag=True)
@click.option('--basic', '-b', help='run basic tests of basic_cases test data in yml for regression.', is_flag=True)
@click.option('--random', '-r', help='run random tests of random_cases test data in yml', is_flag=True)
@click.option('--seed', help="set random seed for random functions of each spec yaml", type=int, default=3428)
@click.option('--rtimes', help="set random cases generation times", type=int, default=1)
@click.option('--retry', help='retry last failed cases', is_flag=True)
@click.option('--failing-info', '-fi', help="print the failing info into the screen, rather than into the log/generator_report.log.", is_flag=True)
@click.option('--param-info', '-pi', help="print params information into log/params.yaml of cases collected.", is_flag=True)
@click.option('--timeout', '-to', help='if a case compiles longer than timeout seconds, we think it fails.', default=None, type=float)
@click.option('--riscv_dv', '-dv', help='riscv dv only', is_flag=True)
@click.option('--target_dv', '-t', help='riscv dv only, target name', default="rv32imc")
@click.option('--test_dv', '-tn', help='riscv dv only, test name', default="")
@click.option('--exclude-groups', '-eg', help='key word list for excluding groups', default="")
@click.option('--exclude-cases', '-ec', help='key word list for excluding cases', default="")

def gen(**kwargs):
    """Generate verification cases for current target."""
    kwargs['config'] = parse_config_files(kwargs['config'])
    kwargs['pkgroot'] = get_package_root()
    generator.main(Args(**kwargs))


@ cli.command()
@click.option('--config', help='config yaml file, default config/prod.yml', default='target.yml')
@click.option('--retry', '-r', help='retry last failed cases', is_flag=True)
@click.option('--nproc', '-n', help='runner process number for run cases, default 1', type=int, default=1)
@click.option('--cases', '-c', help=textwrap.dedent('''\
                                    test case list string or file, for example:
                                    - vsub_vv,addi/test_imm_op/
                                    - cases.list'''), default='')
@click.option('--failing-info', '-fi', help="print the failing info into the screen, rather than  in the log/runner_report.log.", is_flag=True)
@click.option('--timeout', '-to', help='if simulator runs longer than timeout seconds, we think it fails.', default=None, type=float)
# options to configure the simulator
@click.option('--batch', '-b', help="lsf batch number, the number of servers in lsf clusters to run cases, default 0, don't use lsf.", type=int, default=0)
@click.option('--worker', '-w', help="use this runner as a lsf worker, set by server, needn't set by users.", is_flag=True)
@click.option('--worker_port', help="the port which worker use to connect server, set by server, needn't set by users.", type=int, default=50000)
@click.option('--worker_job_name', '-wjn', help="bsub job name for this worker, set by server, needn't set by users.", default='')
@click.option('--append', '-a', help="append cases results to log/runner_report.log, set by server, needn't set by users..", is_flag=True)
@click.option('--lsf', help='run tests on with lsf clusters, if not set, depend on lsf:is_flag in the file set by --config', is_flag=True)
@click.option('--fsdb', '-f', help='generate fsdb waveform file when running vcs simulator, if not set, depend on vcs:fsdb in the file set by --config', is_flag=True)
@click.option('--tsiloadmem', '-tlm', help='Load binary through TSI instead of backdoor, if not set, depend on vcs:tsiloadmem in the file set by --config', is_flag=True)
@click.option('--vcstimeout', '-vto', help='Number of cycles after which VCS stops, if not set, depend on vcs:vcstimeout in the file set by --config', default=-3333, type=int)
@click.option('--riscv_dv', '-dv', help='riscv dv only', is_flag=True)
def run(**kwargs):
    """Run verification cases for current target."""
    kwargs['config'] = parse_config_files(kwargs['config'])
    kwargs['pkgroot'] = get_package_root()
    runner.main(Args(**kwargs))
                              


if __name__ == '__main__':
    cli()
