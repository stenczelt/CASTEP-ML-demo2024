#!/bin/bash
which gap_fit
which castep.mpi
which hybrid-md

orterun -n 4 --oversubscribe  castep.mpi si
