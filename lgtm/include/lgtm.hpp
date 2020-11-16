//
//

#ifndef LGTM_LGTM_HPP
#define LGTM_LGTM_HPP

#include <lsh.hpp>
#include <graph.hpp>

namespace lgtm {
    struct SearchResult {
        time_t time = 0;
        time_t lsh_time = 0;
        time_t graph_time = 0;
        time_t merge_time = 0;
        vector<mylib::Neighbor> result;
        unsigned long n_bucket_content = 0;
        unsigned long n_node_access = 0;
        unsigned long n_dist_calc = 0;
        unsigned long n_hop = 0;
        double dist_from_start = 0;
        double recall = 0;
    };

    struct SearchResults {
        vector<SearchResult> results;
        void push_back(const SearchResult& result) { results.push_back(result); }

        void save(const string& log_path, const string& result_path) {
            ofstream log_ofs(log_path);
            string line = "time,lsh_time,graph_time,merge_time,"
                          "n_bucket_content,n_node_access,n_dist_calc,"
                          "n_hop,dist_from_start,recall";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id,dist";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                line = to_string(result.time) + "," +
                        to_string(result.lsh_time) + "," +
                        to_string(result.graph_time) + "," +
                        to_string(result.merge_time) + "," +
                        to_string(result.n_bucket_content) + "," +
                        to_string(result.n_node_access) + "," +
                        to_string(result.n_dist_calc) + "," +
                        to_string(result.n_hop) + "," +
                        to_string(result.dist_from_start) + "," +
                        to_string(result.recall);
                log_ofs << line << endl;

                for (const auto& neighbor : result.result) {
                    line = to_string(query_id) + "," +
                           to_string(neighbor.id) + "," +
                           to_string(neighbor.dist);
                    result_ofs << line << endl;
                }

                query_id++;
            }
        }
    };

    struct LGTMIndex {
        int n_thread;
        lsh::LSHIndex lsh;
        graph::GraphIndex graph;

        LGTMIndex(int m, int r, int L, int degree) : n_thread(L), lsh(m, r, L), graph(degree) {}

        void build_lsh(const Dataset<>& dataset) {
            // build ordinal lsh index
            const auto dim = dataset[0].size();
            lsh.build(dataset);

            // collect keys and buckets
            vector<vector<int>> keys;
            vector<Dataset<>> buckets;

            for (const auto& bucket_pair : lsh.hash_tables[0]) {
                keys.emplace_back(bucket_pair.first);

                Dataset<> bucket;
                for (const auto id : bucket_pair.second)
                    bucket.emplace_back(dataset[id]);
                buckets.emplace_back(bucket);
            }

            // calc medoid of each bucket
            vector<int> medoids(buckets.size());
#pragma omp parallel for
            for (int i = 0; i < buckets.size(); ++i) {
                medoids[i] = calc_medoid(buckets[i]);
            }

            // replace bucket with medoid
            for (int i = 0; i < medoids.size(); ++i) {
                lsh.hash_tables[0][keys[i]] = vector<int>{medoids[i]};
            }
        }

        void build(const string& data_path, const string& graph_path, int n) {
            auto dataset_1 = load_data(data_path, n);
            auto dataset_2 = dataset_1;
            cout << "complete: load data" << endl;

            lsh.build(dataset_1);
            cout << "complete: build lsh" << endl;

            graph.load(dataset_2, graph_path, n);
            graph.make_bidirectional();
            graph.optimize_edge();
            cout << "complete: build graph" << endl;
        }

        auto knn_search(const Data<>& query, int k, int n_start_node, int ef) {
            auto result = SearchResult();
            const auto start_time = get_now();

            // lsh
            auto start_ids = lsh.find(query, n_start_node);
            if (start_ids.empty()) start_ids.emplace_back(0);

            result.n_bucket_content = start_ids.size();
            const auto lsh_end_time = get_now();
            result.lsh_time = get_duration(start_time, lsh_end_time);

            // graph
            const auto graph_start_time = get_now();

            auto graph_result = graph.knn_search(query, k, ef, start_ids, n_start_node);

            result.result = graph_result.result;
            result.n_node_access = graph_result.n_node_access;
            result.n_dist_calc = graph_result.n_dist_calc;
            result.n_hop = graph_result.n_hop;
            result.dist_from_start = graph_result.dist_from_start;

            const auto end_time = get_now();
            result.graph_time = get_duration(graph_start_time, end_time);
            result.time = get_duration(start_time, end_time);

            return result;
        }

        auto knn_search_para(const Data<>& query, int k, int n_start_node, int ef) {
            auto result = SearchResult();
            const auto start_time = get_now();

            vector<graph::SearchResult> graph_results(n_thread);
#pragma omp parallel for num_threads(n_thread) schedule(dynamic, 1)
            for (int i = 0; i < n_thread; ++i) {
                // lsh
                const auto& hash_table = lsh.hash_tables[i];
                const auto key = lsh.G[i](query);

                try {
                    const auto& start_ids = hash_table.at(key);
                    graph_results[i] = graph.knn_search(query, k, ef, start_ids, n_start_node);
                } catch (out_of_range) {
                    const auto start_ids = vector<int>{0};
                    graph_results[i] = graph.knn_search(query, k, ef, start_ids, n_start_node);
                }
            }

            // merge
            const auto merge_start_time = get_now();

            vector<bool> added(lsh.dataset.size());
            for (const auto& graph_result : graph_results) {
                for (const auto& neighbor : graph_result.result) {
                    if (added[neighbor.id]) continue;
                    added[neighbor.id] = true;
                    result.result.emplace_back(neighbor);
                }

                result.lsh_time = max(result.lsh_time, graph_result.lsh_time);
                result.graph_time = max(result.graph_time, graph_result.time);
                result.n_node_access = max(result.n_node_access, graph_result.n_node_access);
                result.n_dist_calc = max(result.n_dist_calc, graph_result.n_dist_calc);
                result.n_hop = max(result.n_hop, graph_result.n_hop);
                result.dist_from_start = max(result.dist_from_start, graph_result.dist_from_start);
            }

            sort_neighbors(result.result);
            result.result.resize(k);

            const auto end_time = get_now();
            result.time = get_duration(start_time, end_time);
            result.merge_time = get_duration(merge_start_time, end_time);

            return result;
        }
    };
}

#endif //LGTM_LGTM_HPP
