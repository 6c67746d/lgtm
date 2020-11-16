# HNSW
## Configuration
Edit `config.json`

### Fields
- `data_path`: dataset path (csv)
- `query_path`: query path (csv)
- `groundtruth_path`: ground truth path (csv)
- `save_dir`: output directory
- `n`: number of data
- `n_query`: number of query
- `k`: number of result (corresponds to k of kNN search)
- `m`: minimum out-degree of index 
(actual out-degree is `2 * degree` because it is bidirectional graph)
- `ef`: number of candidates while greedy search

## Build
```
cmake .
make
```

## Run
```
./hnsw
```
