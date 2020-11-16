# Scan
## Configuration
Edit `config.json`.

### Fields
- `data_path`: dataset path (csv)
- `query_path`: query path (csv)
- `save_dir`: output directory
- `n`: number of data
- `n_query`: number of query
- `k`: number of result (corresponds to k of kNN search)

## Build
```
cmake .
make
```

## Run
```
./scan-knn-search
```
