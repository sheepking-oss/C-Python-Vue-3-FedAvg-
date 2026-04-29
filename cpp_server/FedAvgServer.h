#ifndef FED_AVG_SERVER_H
#define FED_AVG_SERVER_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>

using json = nlohmann::json;

struct TensorData {
    std::string name;
    std::vector<double> data;
    std::vector<size_t> shape;
};

struct ClientModel {
    std::string client_id;
    int sample_count;
    int round;
    std::vector<TensorData> weights;
};

class FedAvgServer {
public:
    FedAvgServer(int global_rounds = 100);
    ~FedAvgServer() = default;

    bool submit_weights(const std::string& client_id, 
                       int sample_count, 
                       int round,
                       const std::vector<TensorData>& weights);

    json get_aggregated_weights() const;
    json get_status() const;
    bool is_round_complete() const;
    int get_current_round() const { return current_round_; }
    
    void set_expected_clients(int count) { expected_clients_ = count; }
    int get_expected_clients() const { return expected_clients_; }
    
    void reset();
    
    bool aggregate();

private:
    TensorData average_tensors(const std::vector<const TensorData*>& tensors,
                               const std::vector<int>& sample_counts) const;

    int current_round_;
    int max_rounds_;
    int expected_clients_;
    std::unordered_map<std::string, ClientModel> client_models_;
    std::vector<TensorData> global_weights_;
    mutable std::mutex mutex_;
};

#endif
