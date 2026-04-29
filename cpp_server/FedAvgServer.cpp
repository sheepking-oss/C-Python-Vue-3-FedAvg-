#include "FedAvgServer.h"
#include <numeric>
#include <iostream>

FedAvgServer::FedAvgServer(int global_rounds) 
    : current_round_(1), max_rounds_(global_rounds), expected_clients_(0) {
}

bool FedAvgServer::submit_weights(const std::string& client_id,
                                    int sample_count,
                                    int round,
                                    const std::vector<TensorData>& weights) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (round != current_round_) {
        std::cerr << "Round mismatch: expected " << current_round_ << ", got " << round << std::endl;
        return false;
    }
    
    ClientModel model;
    model.client_id = client_id;
    model.sample_count = sample_count;
    model.round = round;
    model.weights = weights;
    
    client_models_[client_id] = model;
    std::cout << "Received weights from client: " << client_id 
              << ", samples: " << sample_count 
              << ", round: " << round << std::endl;
    
    return true;
}

TensorData FedAvgServer::average_tensors(
    const std::vector<const TensorData*>& tensors,
    const std::vector<int>& sample_counts) const {
    
    size_t total_samples = std::accumulate(sample_counts.begin(), sample_counts.end(), 0);
    
    TensorData result;
    result.name = tensors[0]->name;
    result.shape = tensors[0]->shape;
    result.data.resize(tensors[0]->data.size(), 0.0);
    
    for (size_t i = 0; i < tensors.size(); ++i) {
        double weight = static_cast<double>(sample_counts[i]) / total_samples;
        for (size_t j = 0; j < result.data.size(); ++j) {
            result.data[j] += tensors[i]->data[j] * weight;
        }
    }
    
    return result;
}

bool FedAvgServer::aggregate() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (client_models_.empty()) {
        std::cerr << "No client models to aggregate" << std::endl;
        return false;
    }
    
    std::cout << "Aggregating weights from " << client_models_.size() << " clients" << std::endl;
    
    std::vector<std::string> client_ids;
    std::vector<int> sample_counts;
    
    for (const auto& pair : client_models_) {
        client_ids.push_back(pair.first);
        sample_counts.push_back(pair.second.sample_count);
    }
    
    size_t num_weights = client_models_[client_ids[0]].weights.size();
    global_weights_.clear();
    
    for (size_t w = 0; w < num_weights; ++w) {
        std::vector<const TensorData*> tensors;
        for (const auto& id : client_ids) {
            tensors.push_back(&client_models_[id].weights[w]);
        }
        
        TensorData avg = average_tensors(tensors, sample_counts);
        global_weights_.push_back(avg);
    }
    
    current_round_++;
    std::cout << "Aggregation complete. Moving to round " << current_round_ << std::endl;
    
    return true;
}

json FedAvgServer::get_aggregated_weights() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    json result;
    result["round"] = current_round_;
    result["weights"] = json::array();
    
    for (const auto& tensor : global_weights_) {
        json t;
        t["name"] = tensor.name;
        t["shape"] = tensor.shape;
        t["data"] = tensor.data;
        result["weights"].push_back(t);
    }
    
    return result;
}

json FedAvgServer::get_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    json result;
    result["current_round"] = current_round_;
    result["max_rounds"] = max_rounds_;
    result["expected_clients"] = expected_clients_;
    result["submitted_clients"] = static_cast<int>(client_models_.size());
    result["is_complete"] = is_round_complete();
    
    json client_list = json::array();
    for (const auto& pair : client_models_) {
        client_list.push_back({
            {"client_id", pair.first},
            {"sample_count", pair.second.sample_count},
            {"round", pair.second.round}
        });
    }
    result["clients"] = client_list;
    
    return result;
}

bool FedAvgServer::is_round_complete() const {
    if (expected_clients_ <= 0) {
        return !client_models_.empty();
    }
    return client_models_.size() >= static_cast<size_t>(expected_clients_);
}

void FedAvgServer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_round_ = 1;
    client_models_.clear();
    global_weights_.clear();
}
