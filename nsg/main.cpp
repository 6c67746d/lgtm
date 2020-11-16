#include <iostream>
#include <mylib.hpp>
#include <nsg.hpp>

using namespace std;
using namespace mylib;

int main() {
    const auto config = read_config();
    const int n = config["n"], n_query = config["n_query"];
    const string data_path = config["data_path"];
    const string query_path = config["query_path"];
    const string graph_path = config["graph_path"];
    const string ground_truth_path = config["groundtruth_path"];

    const auto queries = load_data(query_path, n_query);
    const auto ground_truth = load_neighbors(ground_truth_path, n_query, true);

    int m = config["m"];

    int k = config["k"];
    int l = config["l"];

    auto index = NSG(m);
    index.build(data_path, graph_path, n);

    cout << "complete: build index" << endl;

    SearchResults results;
    for (const auto& query : queries) {
        auto result = index.knn_search(query, k, l);
        result.recall = calc_recall(result.result, ground_truth[query.id], k);
        results.push_back(move(result));
    }

    const string save_postfix = "k" + to_string(k) +
                                "-m" + to_string(m) +
                                "-l" + to_string(l) + ".csv";
    const string save_dir = config["save_dir"];
    const string log_path = save_dir + "log-" + save_postfix;
    const string result_path = save_dir + "result-" + save_postfix;
    results.save(log_path, result_path);
}
