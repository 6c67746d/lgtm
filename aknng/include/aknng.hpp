//
//

#ifndef NNDESCENT_AKNNG_HPP
#define NNDESCENT_AKNNG_HPP

#include <mylib.hpp>

using namespace std;
using namespace mylib;

namespace aknng {
    struct Node {
        int id;
        Data<> data;
        int degree;
        multimap<double, int> neighbors;
        unordered_map<size_t, bool> added;

        Node(const Data<>& data, int degree) : data(data), id(data.id), degree(degree) {
            added[data.id] = true;
        }

        int add_neighbor(const Node& node) {
            if (added.find(node.id) != added.end()) return 0;

            const auto dist = euclidean_distance(data, node.data);

            if (neighbors.size() < degree) {
                neighbors.emplace(dist, node.id);
                added[node.id] = true;
                return 1;
            }

            const auto furthest_ptr = --neighbors.cend();
            const auto furthest_dist = furthest_ptr->first;

            if (dist >= furthest_dist) return 0;

            neighbors.emplace(dist, node.id);
            added[node.id] = true;

            // delete neighbor if over degree
            if (neighbors.size() > degree) {
                neighbors.erase(furthest_ptr);
                added.erase(furthest_ptr->second);
            }

            return 1;
        }
    };

    struct AKNNG {
        vector<Node> nodes;
        int degree;
        mt19937 engine;

        AKNNG(int degree) : degree(degree), engine(42) {}
        auto size() const { return nodes.size(); }
        auto begin() const { return nodes.begin(); }
        auto end() const { return nodes.end(); }
        decltype(auto) operator [] (size_t i) { return nodes[i]; }
        decltype(auto) operator [] (const Node& n) { return nodes[n.id]; }

        auto get_neighbors_list() {
            vector<vector<int>> neighbors_list(nodes.size());
            unordered_map<int, bool> added;

            for (const auto& node : nodes) {
                for (const auto& neighbor_pair : node.neighbors) {
                    const auto neighbor_id = neighbor_pair.second;

                    // get neighbors
                    neighbors_list[node.id].emplace_back(neighbor_id);

                    // get reverse neighbors
                    neighbors_list[neighbor_id].emplace_back(node.id);
                }
            }

            return neighbors_list;
        }

        void build(const Dataset<>& dataset) {
            // init nodes
            for (auto& data : dataset) {
                nodes.emplace_back(move(data), degree);
            }

            // init neighbors
            uniform_int_distribution<int> dist(0, nodes.size() - 1);
            for (auto& node : nodes) {
                while (node.neighbors.size() < degree) {
                    const auto& random_node = nodes[dist(engine)];
                    node.add_neighbor(random_node);
                }
            }

            while (true) {
                int n_updated = 0;
                const auto neighbors_list = get_neighbors_list();
#pragma omp parallel
                {
#pragma omp for schedule(dynamic, 1000) nowait reduction(+:n_updated)
                    for (int id = 0; id < nodes.size(); ++id) {
                        auto& node = nodes[id];

                        for (const auto neighbor_id_1 : neighbors_list[id]) {
                            for (const auto neighbor_id_2 : neighbors_list[neighbor_id_1]) {
                                const auto& neighbor = nodes[neighbor_id_2];
                                n_updated += node.add_neighbor(neighbor);
                            }
                        }
                    }
                };
                if (n_updated <= 0) break;
            }
        }

        void build(string data_path, int n = -1) {
            build(load_data(data_path, n));
        }

        void save(const string& save_path) {
            // csv
            if (is_csv(save_path)) {
                ofstream ofs(save_path);
                string line;
                for (const auto& node : nodes) {
                    for (const auto& neighbor_pair : node.neighbors) {
                        line = to_string(node.data.id) + ',' +
                               to_string(neighbor_pair.second) + ',' +
                               to_string(neighbor_pair.first);
                        ofs << line << endl;
                    }
                }
                return;
            }

            // dir
            vector<string> lines(static_cast<unsigned long>(ceil(nodes.size() / 1000.0)));
            for (const auto& node : nodes) {
                const size_t line_i = node.data.id / 1000;
                for (const auto& neighbor_pair : node.neighbors) {
                    lines[line_i] += to_string(node.data.id) + "," +
                                     to_string(neighbor_pair.second) + "," +
                                     to_string(neighbor_pair.first) + "\n";
                }
            }

            for (int i = 0; i < lines.size(); i++) {
                const string path = save_path + "/" + to_string(i) + ".csv";
                ofstream ofs(path);
                ofs << lines[i];
            }
        }
    };
}

#endif //NNDESCENT_AKNNG_HPP
