# MiraNetworkMapper
Scripts to generate mpi maps for Mira [and cetus]

## Usage
```
./make_map APP_TOPOLOGY NETWORK_TOPOLOGY MODE
```
where `APP_TOPOLOGY` is a comma separated list `NX,NY,NZ` (no spaces),
`NETWORK_TOPOLOGY` is another list `A,B,C,D,E`, and mode is an integer.

## Partitions

The partitions on Mira and Cetus are listed [here](https://www.alcf.anl.gov/user-guides/running-jobs#mapping-of-mpi-tasks-to-cores).
They support power-of-two modes from 1 to 64.

