#!/bin/bash
which gap_fit
which castep.mpi
which hybrid-md

orterun -n 16 --oversubscribe  castep.mpi my_system
