#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

cd "$(dirname "$0")/../tests/integration/py"

run_tests () {
    torchrun --nnodes 1 --nproc_per_node 4 AllReduceTest.py --verbose
}

# XCCL
export TEST_BACKEND=xccl
export TEST_DEVICE="xpu"
run_tests
