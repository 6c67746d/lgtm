#include <mylib.hpp>
#include <aknng.hpp>

using namespace std;
using namespace mylib;
using namespace aknng;

int main() {
    auto config = read_config();
    string data_path = config["data_path"];

    int degree = config["degree"], n = config["n"];
    AKNNG aknng(degree);
    aknng.build(data_path, n);

    string save_path = config["save_path"];
    aknng.save(save_path);
    cout << "complete" << endl;
}
