#include <mylib.hpp>

using namespace mylib;

struct SearchResult {
    time_t time = 0;
    multimap<double, int> result;
    unsigned long n_dist_calc = 0;
};

struct SearchResults {
    vector<SearchResult> results;
    void push_back(const SearchResult& result) { results.push_back(result); }

    void save(const string& log_path, const string& result_path) {
        ofstream log_ofs(log_path);
        string line = "time,n_dist_calc";
        log_ofs << line << endl;

        ofstream result_ofs(result_path);
        line = "query_id,data_id,dist";
        result_ofs << line << endl;

        int query_id = 0;
        for (const auto& result : results) {
            line = to_string(result.time) + "," +
                   to_string(result.n_dist_calc);
            log_ofs << line << endl;

            for (const auto& result_pair : result.result) {
                line = to_string(query_id) + "," +
                        to_string(result_pair.second) + "," +
                        to_string(result_pair.first);
                result_ofs << line << endl;
            }

            query_id++;
        }
    }
};

SearchResult knn_search(const Data<>& query, const int k, const Dataset<>& series) {
    auto result = SearchResult();
    const auto start_time = get_now();

    for (const auto& data : series) {
        const auto dist = euclidean_distance(query, data);
        result.result.emplace(dist, data.id);
        if (result.result.size() > k)
            result.result.erase(--result.result.cend());
    }

    const auto end_time = get_now();
    result.time = get_duration(start_time, end_time);

    return result;
}

int main() {
    const auto config = read_config();
    const int n = config["n"], n_query = config["n_query"], k = config["k"];
    const string data_path = config["data_path"];
    const auto dataset = load_data(data_path, n);

    const string query_path = config["query_path"];
    const auto queryset = load_data(query_path, n_query);

    const string save_dir = config["save_dir"];
    const string log_path = save_dir + "log.csv", result_path = save_dir + "result.csv";

    SearchResults results;
    results.results.resize(queryset.size());

#pragma omp parallel for
    for (int i = 0; i < queryset.size(); i++) {
        const auto& query = queryset[i];
        results.results[i] = knn_search(query, k, dataset);
    }

    results.save(log_path, result_path);
}
