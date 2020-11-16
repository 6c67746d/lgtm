# NSG
## Configuration
Edit `config.json`.

### Fields
- `data_path`: dataset path (csv)
- `query_path`: query path (csv)
- `groundtruth_path`: ground truth path (csv).
See README.md of scan-knn-search project to make it.
- `graph_path`: AKNNG path (csv).
See README.md of AKNNG project to make it.
- `save_dir`: output directory
- `n`: number of data
- `n_query`: number of query
- `k`: number of result (corresponds to k of kNN search)
- `m`: out-degree
- `l`: number of candidates while greedy search

## Build
```
cmake .
make
```

## Run
```
./nsg
```
