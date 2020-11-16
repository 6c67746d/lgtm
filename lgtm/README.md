# LGTM
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
- `t`: number of thread (= the number of hash table of LSH)
- `m`: number of hash function of LSH family 
(hash vector H(o) = (h1(o), h2(o), ..., hm(o)))
- `w`: denominator of hash function (h(o) = a･o + b / w)
- `degree`: minimum out-degree of LGTM graph index 
(actual out-degree is `2 * degree` because it is bidirectional graph)
- `ef`: number of candidates while greedy search
- `n_start_node`: number of start node (= number of samples from hash table)

## Build
```
cmake .
make
```

## Run
```
./lgtm
```
