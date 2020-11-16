//
//

#ifndef mylib_GRAPH_HPP
#define mylib_GRAPH_HPP

#include <queue>
#include <mylib.hpp>

using namespace std;
using namespace mylib;

namespace graph {
    struct Node {
        int id;
        Data<> data;
        Neighbors neighbors;
        unordered_map<size_t, bool> added;

        void init() { added[id] = true; }
        Node() : data(Data<>(0, {0})) { init(); }
        Node(const Data<>& data) : id(data.id), data(data) { init(); }

        void add_neighbor(double dist, int neighbor_id) {
            if (added.find(neighbor_id) != added.end()) return;
            added[neighbor_id] = true;
            neighbors.emplace_back(dist, neighbor_id);
        }

        void clear_neighbor() {
            added.clear();
            neighbors.clear();
            added[id] = true;
        }
    };

    struct SearchResult {
        time_t time = 0;
        time_t lsh_time = 0;
        vector<Neighbor> result;
        unsigned long n_node_access = 0;
        unsigned long n_dist_calc = 0;
        unsigned long n_hop = 0;
        double dist_from_start = 0;
    };

    struct GraphIndex {
        vector<Node> nodes;
        int degree, max_degree;
        DistanceFunction<> calc_dist;

        GraphIndex(int degree) : degree(degree), max_degree(degree * 2),
                                 calc_dist(select_distance()) {}

        auto size() const { return nodes.size(); }
        auto begin() const { return nodes.begin(); }
        auto end() const { return nodes.end(); }
        auto& operator [] (size_t i) { return nodes[i]; }
        auto& operator [] (const Node& n) { return nodes[n.data.id]; }
        const auto& operator [] (size_t i) const { return nodes[i]; }
        const auto& operator [] (const Node& n) const { return nodes[n.data.id]; }

        void init_data(const Dataset<>& dataset) {
            for (const auto& data : dataset) {
                nodes.emplace_back(data);
            }
        }

        void load(const Dataset<>& series, const string& graph_path, int n) {
            init_data(series);

            // csv file
            if (is_csv(graph_path)) {
                ifstream ifs(graph_path);
                if (!ifs) {
                    const string message = "Can't open file!: " + graph_path;
                    throw runtime_error(message);
                }

                string line;
                while (getline(ifs, line)) {
                    const auto row = split<double>(line);
                    auto& node = nodes[row[0]];
                    if (degree == -1 || node.neighbors.size() < degree) {
                        node.add_neighbor(row[2], row[1]);
                    }
                }
                return;
            }

            // dir
#pragma omp parallel for
            for (int i = 0; i < n; i++) {
                const string path = graph_path + "/" + to_string(i) + ".csv";
                ifstream ifs(path);

                if (!ifs) {
                    const string message = "Can't open file!: " + path;
                    throw runtime_error(message);
                }

                string line;
                while (getline(ifs, line)) {
                    const auto row = split<double>(line);
                    auto& node = nodes[row[0]];
                    if (degree == -1 || node.neighbors.size() < degree) {
                        node.add_neighbor(row[2], row[1]);
                    }
                }
            }
        }

        void load(const string& data_path, const string& graph_path, int n) {
            auto series = load_data(data_path, n);
            load(series, graph_path, n);
        }

        void save(const string& save_path) {
            // csv
            if (is_csv(save_path)) {
                ofstream ofs(save_path);
                string line;
                for (const auto& node : nodes) {
                    for (const auto& neighbor : node.neighbors) {
                        line = to_string(node.data.id) + "," +
                               to_string(neighbor.id) + "," +
                               to_string(neighbor.dist) + "\n";
                    }
                    ofs << line;
                }
                return;
            }

            // dir
            vector<string> lines(static_cast<unsigned long>(ceil(nodes.size() / 1000.0)));
            for (const auto& node : nodes) {
                const size_t line_i = node.data.id / 1000;
                for (const auto& neighbor : node.neighbors) {
                    lines[line_i] += to_string(node.data.id) + "," +
                                     to_string(neighbor.id) + "," +
                                     to_string(neighbor.dist) + "\n";
                }
            }

            for (int i = 0; i < lines.size(); i++) {
                const string path = save_path + "/" + to_string(i) + ".csv";
                ofstream ofs(path);
                ofs << lines[i];
            }
        }

        auto knn_search(const Data<>& query, int k, int ef,
                const vector<int>& start_ids, int n_start_id) {
            auto result = SearchResult();
//            const auto start_time = get_now();

            priority_queue<Neighbor, vector<Neighbor>, CompGreater> candidates;
            priority_queue<Neighbor, vector<Neighbor>, CompLess> top_candidates;

            vector<bool> visited(nodes.size());

            Neighbors initial_candidates;

            // calculate distance to start nodes
            n_start_id = min(n_start_id, (int)start_ids.size());
            for (int i = 0; i < n_start_id; ++i) {
                const auto start_id = start_ids[i];
                const auto& start_node = nodes[start_id];
                const auto dist = calc_dist(query, start_node.data);

                initial_candidates.emplace_back(dist, start_id);
            }

            // decide nearest node as start node
            sort_neighbors(initial_candidates);
            const auto nearest_start_candidate = initial_candidates[0];
            visited[nearest_start_candidate.id] = true;
            candidates.emplace(nearest_start_candidate);
            top_candidates.emplace(nearest_start_candidate);

            result.dist_from_start = nearest_start_candidate.dist;

            while (!candidates.empty()) {
                const auto nearest_candidate = candidates.top();
                const auto& nearest_candidate_node = nodes[nearest_candidate.id];
                candidates.pop();

                if (nearest_candidate.dist > top_candidates.top().dist) break;

                ++result.n_hop;

                for (const auto neighbor : nearest_candidate_node.neighbors) {
                    if (visited[neighbor.id]) continue;
                    visited[neighbor.id] = true;

                    const auto& neighbor_node = nodes[neighbor.id];
                    const auto dist_from_neighbor =
                            calc_dist(query, neighbor_node.data);
                    ++result.n_dist_calc;

                    if (dist_from_neighbor < top_candidates.top().dist ||
                        top_candidates.size() < ef) {
                        candidates.emplace(dist_from_neighbor, neighbor.id);
                        top_candidates.emplace(dist_from_neighbor, neighbor.id);

                        if (top_candidates.size() > ef) top_candidates.pop();
                    }
                }
            }

            while (!top_candidates.empty()) {
                result.result.emplace_back(top_candidates.top());
                top_candidates.pop();
            }

            reverse(result.result.begin(), result.result.end());
            if (result.result.size() > k) result.result.resize(k);

//            const auto end_time = get_now();
//            result.time = get_duration(start_time, end_time);

            return result;
        }

        auto knn_search_nsg(const Data<>& query, int k, const vector<int>& start_ids, int l) {
            auto result = SearchResult();
            const auto start_time = get_now();

            vector<bool> checked(nodes.size()), added(nodes.size());
            vector<Neighbor> candidates;
            candidates.reserve(l + start_ids.size() + max_degree);

            for (const auto data_id : start_ids) {
                added[data_id] = true;
                const auto& start_node = nodes[data_id];

                const auto dist_to_start_node = calc_dist(query, start_node.data);
                ++result.n_dist_calc;
                candidates.emplace_back(dist_to_start_node, data_id);
            }

            sort(candidates.begin(), candidates.end(),
                 [](const auto& n1, const auto& n2) { return n1.dist < n2.dist; });

            result.dist_from_start = candidates.front().dist;

            while (true) {
                // find the first unchecked node
                int first_unchecked_index = 0;
                for (const auto candidate : candidates) {
                    if (!checked[candidate.id]) break;
                    ++first_unchecked_index;
                }

                // checked all candidates
                if (first_unchecked_index >= l) break;

                ++result.n_hop;

                const auto first_unchecked_node_id = candidates[first_unchecked_index].id;
                checked[first_unchecked_node_id] = true;
                const auto& first_unchecked_node = nodes[first_unchecked_node_id];

                for (const auto& neighbor : first_unchecked_node.neighbors) {
                    result.n_node_access++;

                    if (added[neighbor.id]) continue;
                    added[neighbor.id] = true;

                    result.n_dist_calc++;

                    const auto& neighbor_node = nodes[neighbor.id];
                    const auto dist = calc_dist(query, neighbor_node.data);
                    candidates.emplace_back(dist, neighbor.id);
                }

                // sort and resize candidates l
                sort(candidates.begin(), candidates.end(),
                     [](const auto& n1, const auto& n2) { return n1.dist < n2.dist; });
                candidates.resize(l);
            }

            for (const auto& c : candidates) {
                result.result.emplace_back(c);
                if (result.result.size() >= k) break;
            }

            const auto end_time = get_now();
            result.time = get_duration(start_time, end_time);

            return result;
        }

        auto knn_search(const Data<>& query, int k, int start_id, int l) {
            const auto start_series = vector<int>{start_id};
            return knn_search_nsg(query, k, start_series, l);
        }

        auto tolerant_knn_search(const Data<>& query, int k,
                                 const vector<int>& start_ids, int tol) {
            auto result = SearchResult();
            const auto start_time = get_now();

            vector<bool> checked(nodes.size()), added(nodes.size());

            priority_queue<Neighbor, vector<Neighbor>, CompGreater> candidates;
            priority_queue<Neighbor, vector<Neighbor>, CompLess> top_candidates;

            // init candidates
            for (const auto& start_id : start_ids) {
                if (added[start_id]) continue;
                added[start_id] = true;
                const auto& start_node = nodes[start_id];

                const auto dist_to_start_node = calc_dist(query, start_node.data);
                candidates.emplace(dist_to_start_node, start_id);
            }

            result.dist_from_start = candidates.top().dist;

            int n_result_unchanged = 0;
            while (true) {
                const auto nearest_candidate_id = candidates.top().id;
                const auto& nearest_candidate = nodes[nearest_candidate_id];
                checked[nearest_candidate_id] = true;
                candidates.pop();

                result.n_hop++;

                bool result_changed = false;
                for (const auto neighbor : nearest_candidate.neighbors) {
                    result.n_node_access++;

                    if (added[neighbor.id]) continue;
                    added[neighbor.id] = true;

                    result.n_dist_calc++;

                    const auto& neighbor_node = nodes[neighbor.id];
                    const auto dist = calc_dist(query, neighbor_node.data);
                    candidates.emplace(dist, neighbor.id);

                    if (top_candidates.empty()) {
                        top_candidates.emplace(dist, neighbor.id);
                        continue;
                    }

                    if (dist < top_candidates.top().dist || top_candidates.size() < k) {
                        top_candidates.emplace(dist, neighbor.id);
                        if (top_candidates.size() > k) top_candidates.pop();
                        result_changed = true;
                    }
                }

                if (result_changed) n_result_unchanged = 0;
                else ++n_result_unchanged;

                if (n_result_unchanged > tol) break;
            }

            while (!top_candidates.empty()) {
                result.result.emplace_back(top_candidates.top());
                top_candidates.pop();
            }

            reverse(result.result.begin(), result.result.end());

            const auto end_time = get_now();
            result.time = get_duration(start_time, end_time);

            return result;
        }

        auto tolerant_knn_search_nsg(const Data<>& query, int k,
                                 const vector<int>& start_ids, int tol) {
            auto result = SearchResult();
            const auto start_time = get_now();

            vector<bool> checked(nodes.size()), added(nodes.size());
            vector<Neighbor> candidates;

            // init candidates
            for (const auto& start_id : start_ids) {
                if (added[start_id]) continue;
                added[start_id] = true;
                const auto& start_node = nodes[start_id];

                const auto dist_to_start_node = calc_dist(query, start_node.data);
                candidates.emplace_back(dist_to_start_node, start_id);
            }

            result.dist_from_start = candidates.front().dist;

            int n_result_unchanged = 0;
            while (true) {
                sort(candidates.begin(), candidates.end(),
                     [](const Neighbor& n1, const Neighbor& n2) {
                         return n1.dist < n2.dist; });

                // find the first unchecked node
                int first_unchecked_index = 0;
                for (const auto candidate : candidates) {
                    if (!checked[candidate.id]) break;
                    ++first_unchecked_index;
                }

                // checked all candidates
                if (first_unchecked_index >= candidates.size()) break;

                // not update top-k candidates
                if (first_unchecked_index >= k) {
                    ++n_result_unchanged;
                    if (n_result_unchanged > tol) break;
                }

                const auto first_unchecked_node_id = candidates[first_unchecked_index].id;
                checked[first_unchecked_node_id] = true;
                const auto& first_unchecked_node = nodes[first_unchecked_node_id];

                for (const auto neighbor : first_unchecked_node.neighbors) {
                    result.n_node_access++;

                    if (added[neighbor.id]) continue;
                    added[neighbor.id] = true;

                    result.n_dist_calc++;

                    const auto& neighbor_node = nodes[neighbor.id];
                    const auto dist = calc_dist(query, neighbor_node.data);
                    candidates.emplace_back(dist, neighbor.id);
                }
            }

            for (const auto& result_pair : candidates) {
                result.result.emplace_back(result_pair);
                if (result.result.size() >= k) break;
            }

            const auto end_time = get_now();
            result.time = get_duration(start_time, end_time);

            return result;
        }

        void make_reverse() {

        }

        void make_bidirectional() {
            for (auto& node : nodes) {
                for (const auto& neighbor : node.neighbors) {
                    // add reverse edge
                    nodes[neighbor.id].add_neighbor(neighbor.dist, node.id);
                }
            }
        }

        void optimize_edge() {
            for (auto& node : nodes) {
                auto& neighbors = node.neighbors;
                if (neighbors.size() < max_degree) continue;

                // sort edges with its length
                sort(neighbors.begin(), neighbors.end(),
                     [](const Neighbor& n1, const Neighbor& n2) {
                         return n1.dist < n2.dist; });

                // select appropriate edge
                vector<bool> added(nodes.size());
                vector<Neighbor> new_neighbors;
                new_neighbors.emplace_back(neighbors.front());

                for (const auto& candidate : neighbors) {
                    const auto& candidate_node = nodes[candidate.id];

                    bool good = true;
                    for (const auto& new_neighbor : new_neighbors) {
                        const auto& new_neighbor_node = nodes[new_neighbor.id];
                        const auto dist = calc_dist(
                                candidate_node.data, new_neighbor_node.data);

                        if (dist < candidate.dist) {
                            good = false;
                            break;
                        }
                    }

                    if (!good) continue;
                    added[candidate.id] = true;
                    new_neighbors.emplace_back(candidate);

                    if (new_neighbors.size() >= max_degree) break;
                }

                for (const auto& candidate : neighbors) {
                    if (new_neighbors.size() >= max_degree) break;

                    if (added[candidate.id]) continue;
                    added[candidate.id] = true;
                    new_neighbors.emplace_back(candidate);
                }

                node.neighbors = new_neighbors;
            }
        }
    };
}

#endif //mylib_GRAPH_HPP
