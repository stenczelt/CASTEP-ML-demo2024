#  Hybrid MD decision making package
#
#  Copyright (c) Tamas K. Stenczel 2021-2023.
"""
CLI of hybrid MD implementation.

The QM calculators should call this with the subcommands at correct points in the
calculation.

"""
import sys
from datetime import datetime

import click

from hybrid_md.decision_making import PreStepReturnNumber, get_decision_maker
from hybrid_md.refit import refit
from hybrid_md.state_objects import HybridMD

VERBOSE = True


@click.group("hybrid-md")
def main():
    """
    Hybrid MD main CLI. See the subcommands for more details.
    """


@main.command("initialise")
@click.argument("mode", type=click.STRING)
@click.argument("seed", type=click.STRING)
@click.argument("md-iteration", type=click.INT, default=0)
def initialise(seed, md_iteration, mode):
    """Initialisation of the Hybrid MD run.

    Answer: integer exit code (three bits encoded)
    - i_exit &  1: read log and put in code's log
    - i_exit &  2: start with ab-initio (on True), else GAP
    """

    # create the initial state object
    state = HybridMD(seed, md_iteration)
    state.carry.current_check_interval = state.settings.check_interval
    state.carry.next_is_pre_step = True

    # decide if we are continuing a calculation
    continuation = md_iteration > 0
    if continuation:
        state.handle_continuation()

    # write state to disc, only `next_is_pre_step` relevant though
    state.carry.dump()

    # write the log for the .castep file
    state.io_initial_step_banner()

    # exit status -- log reading always ON
    if state.settings.num_initial_steps == 0:
        exit_code = 1
    else:
        exit_code = 3

    if VERBOSE:
        print(
            f"Hybrid-MD: INIT Step, exit: {exit_code}x"
            f" -- num_initial_steps {state.settings.num_initial_steps}",
            f" -- continuation {continuation}",
        )
    sys.exit(exit_code)


@main.command("pre-step")
@click.argument("mode", type=click.STRING)
@click.argument("seed", type=click.STRING)
@click.argument("md-iteration", type=click.INT)
def pre_step(seed, md_iteration, mode):
    """Start of the MD-Hybrid step, called by md step

    Answer: integer exit code (three bits encoded)
    - i_exit &  1: read log and put in code's log
    - i_exit &  2: do ab-initio now
    - i_exit &  4: do ab-initio in next iteration (rest of loop)
    - i_exit &  8: do update cell
    - i_exit & 16: use ab-initio forces in MD
    """

    # state of object
    state = HybridMD(seed, md_iteration)
    state.carry.load()
    state.carry.reset()
    if not state.carry.next_is_pre_step:
        raise RuntimeError(
            "Hybrid MD steps called in the wrong order, expected post-step"
        )

    # decision-making & update of state object
    decision = get_decision_maker(state).get_step_kind(md_iteration)
    converter = PreStepReturnNumber(state)
    return_value = converter.push_state(decision)

    # dump state
    state.carry.dump()

    if VERBOSE:
        print(
            f"{datetime.now().isoformat()} - "
            f"Hybrid-MD:  PRE Step, exit:{return_value:4}, "
            f"md_iteration:{md_iteration:3}  -->",
            converter.do_ab_initio,
            converter.next_ab_initio,
            converter.do_update_cell,
            converter.use_qm_forces,
        )

    # pass the result back
    sys.exit(return_value)


@main.command("post-step")
@click.argument("mode", type=click.STRING)
@click.argument("seed", type=click.STRING)
@click.argument("md-iteration", type=click.INT)
def post_step(seed, md_iteration, mode):
    """End of the MD-Hybrid step, called by md.f90

    Answer: integer exit code (three bits encoded)
    - i_exit &  1: read log and put in code's log
    - i_exit &  2: read the GAP model again
    """

    # state of object
    state = HybridMD(seed, md_iteration)
    state.carry.load()
    if state.carry.next_is_pre_step:
        raise RuntimeError(
            "Hybrid MD steps called in the wrong order, expected pre-step"
        )

    # main logic
    if state.carry.do_comparison:
        # 1. read output 2. calculate E, F, S errors for this frame and cumulatively,
        # force by species
        state.read_xyz()

        # 3. decide if we are fitting or not
        if not state.check_tolerances() and state.settings.can_update:
            state.carry.do_update_model = True

        # 4. IO: errors of this step and cumulative ones as well
        state.error_table()
        state.cumulative_error_table()

    # 5. update model -- triggered by either pre- or post-step
    if state.carry.do_update_model:
        refit(state)

    # 6. post-step call to decision maker
    get_decision_maker(state).post_step_action(md_iteration)
    read_num_steps = False
    if state.carry.do_comparison or (
        # re-using the same case, this could be nicely refactored
        md_iteration == 1
        and state.settings.num_initial_steps == 1
    ):
        # write next num steps
        num_steps = state.carry.current_check_interval
        print("Saving skip N step in file:", num_steps)
        with open(f"{seed}.ml-energy-num.txt", "w") as file:
            file.write(str(num_steps) + "\n")
        read_num_steps = True

    # save state
    state.carry.next_is_pre_step = True
    state.carry.dump()

    if VERBOSE:
        print(
            f"{datetime.now().isoformat()} - "
            f"Hybrid-MD: POST Step, exit:"
            f"{int(state.carry.do_update_model):4}, "
            f"md_iteration:{md_iteration:3}, "
            f"read_num_steps:{read_num_steps:3}"
        )

    # exit status -- log reading always ON
    exit_code = (
        int(state.carry.do_comparison)
        + 2 * int(state.carry.do_update_model)
        + 4 * int(read_num_steps)
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
    # in case main() does not exit
    sys.exit(-999)
