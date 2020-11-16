#include <mylib.hpp>
#include <hnsw.hpp>

using namespace mylib;
using namespace hnsw;

int main() {
    const auto config = read_config();
    const string data_path = config["data_path"],
            query_path = config["query_path"],
            groundtruth_path = config["groundtruth_path"];
    const int n = config["n"], n_query = config["n_query"];

    const auto dataset = load_data(data_path, n);
    const auto queries = load_data(query_path, n_query);
    const auto ground_truth = load_neighbors(groundtruth_path, n_query, true);

    const auto start = get_now();

    int m = config["m"];
    auto index = HNSW(m);
    index.build(dataset);

    const auto end = get_now();
    const auto build_time = get_duration(start, end);

    cout << "complete: build index" << endl;
    cout << "indexing time: " << build_time / 60000000 << " [min]" << endl;

    int k = config["k"];
    int ef = config["ef"];

    SearchResults results(n_query);
    for (int i = 0; i < n_query; i++) {
        const auto& query = queries[i];
        auto result = index.knn_search(query, k, ef);
        result.recall = calc_recall(result.result, ground_truth[query.id], k);
        results[i] = result;
    }

    const string save_name = "k" + to_string(k) +
                             "-m" + to_string(m) +
                             "-ef" + to_string(ef) + ".csv";
    const string result_base_dir = config["save_dir"];
    const string log_path = result_base_dir + "log-" + save_name;
    const string result_path = result_base_dir + "result-" + save_name;
    results.save(log_path, result_path);
}
