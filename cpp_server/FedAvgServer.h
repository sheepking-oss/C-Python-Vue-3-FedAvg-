#ifndef FED_AVG_SERVER_H
#define FED_AVG_SERVER_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_map>
#include <cmath>
#include <limits>

using json = nlohmann::json;

struct TensorData {
    std::string name;
    std::vector<double> data;
    std::vector<size_t> shape;
    
    bool operator==(const TensorData& other) const {
        if (name != other.name) return false;
        if (shape != other.shape) return false;
        if (data.size() != other.data.size()) return false;
        return true;
    }
    
    bool shapeMatches(const TensorData& other) const {
        return shape == other.shape && data.size() == other.data.size();
    }
    
    size_t totalElements() const {
        if (shape.empty()) return data.size();
        size_t total = 1;
        for (size_t s : shape) total *= s;
        return total;
    }
};

struct ClientModel {
    std::string client_id;
    int sample_count;
    int round;
    std::vector<TensorData> weights;
};

struct AggregationResult {
    bool success;
    std::string message;
    int num_clients_aggregated;
    int current_round;
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
    void reset_round();
    
    AggregationResult aggregate();
    
    bool validate_weights(const std::vector<TensorData>& weights, 
                          std::string* error_message = nullptr) const;

private:
    TensorData weighted_average_tensors(
        const std::vector<const TensorData*>& tensors,
        const std::vector<double>& weights) const;
    
    TensorData safe_average_tensors(
        const std::vector<const TensorData*>& tensors,
        const std::vector<int>& sample_counts) const;
    
    double compute_epsilon() const {
        return std::numeric_limits<double>::epsilon() * 100.0;
    }
    
    bool are_all_tensors_compatible(
        const std::vector<const TensorData*>& tensors,
        std::string* error_message = nullptr) const;

    int current_round_;
    int max_rounds_;
    int expected_clients_;
    std::unordered_map<std::string, ClientModel> client_models_;
    std::vector<TensorData> global_weights_;
    std::vector<TensorData> reference_weights_template_;
    mutable std::mutex mutex_;
};

#endif
