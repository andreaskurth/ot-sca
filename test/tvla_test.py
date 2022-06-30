# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .cmd import Args, Cmd
from .repo import REPO_PATH, RepoCmd


class TvlaCmd(RepoCmd):
    def __init__(self, args: Args):
        # Insert (relative) path to TVLA before the given arguments.
        args = Args('cw/cw305/tvla.py') + args
        super().__init__(args)


def test_help():
    tvla = TvlaCmd(Args('--help')).run()
    # Assert that a message is printed on stdout or stderr.
    assert(len(tvla.stdout()) != 0 or len(tvla.stderr()) != 0)


def ttest_significant(ttest_trace) -> bool:
    """Determine if a t-test trace contains a significant deviation from the mean."""
    mean = np.mean(ttest_trace, axis=3)
    stddev = np.std(ttest_trace, axis=3)
    threshold = mean + 4.5 * stddev
    abs_max = np.max(np.abs(ttest_trace), axis=3)
    return np.any(abs_max > threshold)


def test_general_leaking_histogram():
    # TODO: Remove the following line once defaults are taken from the Typer definitions.
    Cmd(Args(['cp', REPO_PATH / 'cw/cw305/tvla_aes.yaml', '.'])).run()
    tvla = TvlaCmd(Args(['run-tvla', '--no-plot-figures',
                         '--input-file', 'test/data/tvla_general/sha3_hist_leaking.npz'])
                   ).run()
    assert ttest_significant(np.load('tmp/ttest.npy')), (
           f"{tvla} did not find significant leakage, which is unexpected")


def test_general_nonleaking_histogram():
    # TODO: Remove the following line once defaults are taken from the Typer definitions.
    Cmd(Args(['cp', REPO_PATH / 'cw/cw305/tvla_aes.yaml', '.'])).run()
    tvla = TvlaCmd(Args(['run-tvla', '--no-plot-figures',
                         '--input-file', 'test/data/tvla_general/sha3_hist_nonleaking.npz'])
                   ).run()
    assert not ttest_significant(np.load('tmp/ttest.npy')), (
           f"{tvla} did find significant leakage, which is unexepcted")
