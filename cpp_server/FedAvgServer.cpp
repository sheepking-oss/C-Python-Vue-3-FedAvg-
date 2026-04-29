#include "FedAvgServer.h"
#include <numeric>
#include <iostream>
#include <iomanip>
#include <sstream>

FedAvgServer::FedAvgServer(int global_rounds) 
    : current_round_(1), max_rounds_(global_rounds), expected_clients_(0) {
}

bool FedAvgServer::validate_weights(const std::vector<TensorData>& weights, 
                                      std::string* error_message) const {
    if (weights.empty()) {
        if (error_message) *error_message = "Weights array is empty";
        return false;
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        const TensorData& w = weights[i];
        
        if (w.name.empty()) {
            if (error_message) *error_message = "Tensor name is empty at index " + std::to_string(i);
            return false;
        }
        
        if (w.data.empty()) {
            if (error_message) *error_message = "Tensor data is empty for: " + w.name;
            return false;
        }
        
        size_t expected_size = w.totalElements();
        if (w.data.size() != expected_size) {
            std::stringstream ss;
            ss << "Tensor " << w.name << " size mismatch: expected " << expected_size 
               << " elements based on shape, but got " << w.data.size();
            if (error_message) *error_message = ss.str();
            return false;
        }
        
        for (double val : w.data) {
            if (!std::isfinite(val)) {
                std::stringstream ss;
                ss << "Tensor " << w.name << " contains invalid value (NaN or Inf)";
                if (error_message) *error_message = ss.str();
                return false;
            }
        }
    }
    
    if (!reference_weights_template_.empty()) {
        if (weights.size() != reference_weights_template_.size()) {
            std::stringstream ss;
            ss << "Number of tensors mismatch: expected " << reference_weights_template_.size()
               << " but got " << weights.size();
            if (error_message) *error_message = ss.str();
            return false;
        }
        
        for (size_t i = 0; i < weights.size(); ++i) {
            if (!weights[i].shapeMatches(reference_weights_template_[i])) {
                std::stringstream ss;
                ss << "Tensor " << weights[i].name << " shape mismatch with reference";
                if (error_message) *error_message = ss.str();
                return false;
            }
            if (weights[i].name != reference_weights_template_[i].name) {
                std::stringstream ss;
                ss << "Tensor name mismatch at position " << i << ": expected " 
                   << reference_weights_template_[i].name << " but got " << weights[i].name;
                if (error_message) *error_message = ss.str();
                return false;
            }
        }
    }
    
    return true;
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
    
    if (sample_count <= 0) {
        std::cerr << "Invalid sample count: " << sample_count << " for client " << client_id << std::endl;
        return false;
    }
    
    std::string error_msg;
    if (!validate_weights(weights, &error_msg)) {
        std::cerr << "Weight validation failed for client " << client_id << ": " << error_msg << std::endl;
        return false;
    }
    
    if (reference_weights_template_.empty()) {
        reference_weights_template_ = weights;
        std::cout << "Set reference weight template from client: " << client_id << std::endl;
        std::cout << "Number of tensors: " << weights.size() << std::endl;
        for (const auto& w : weights) {
            std::cout << "  - " << w.name << ": shape=[";
            for (size_t i = 0; i < w.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << w.shape[i];
            }
            std::cout << "], elements=" << w.data.size() << std::endl;
        }
    }
    
    if (client_models_.find(client_id) != client_models_.end()) {
        std::cout << "Warning: Client " << client_id << " re-submitting weights for round " << round << std::endl;
    }
    
    ClientModel model;
    model.client_id = client_id;
    model.sample_count = sample_count;
    model.round = round;
    model.weights = weights;
    
    client_models_[client_id] = model;
    std::cout << "Received weights from client: " << client_id 
              << ", samples: " << sample_count 
              << ", round: " << round 
              << ", tensors: " << weights.size() << std::endl;
    
    return true;
}

bool FedAvgServer::are_all_tensors_compatible(
    const std::vector<const TensorData*>& tensors,
    std::string* error_message) const {
    
    if (tensors.empty()) {
        if (error_message) *error_message = "Empty tensor list";
        return false;
    }
    
    const TensorData* first = tensors[0];
    for (size_t i = 1; i < tensors.size(); ++i) {
        const TensorData* current = tensors[i];
        
        if (first->name != current->name) {
            std::stringstream ss;
            ss << "Tensor name mismatch: expected '" << first->name 
               << "' but got '" << current->name << "'";
            if (error_message) *error_message = ss.str();
            return false;
        }
        
        if (!first->shapeMatches(*current)) {
            std::stringstream ss;
            ss << "Tensor shape mismatch for '" << first->name << "'";
            if (error_message) *error_message = ss.str();
            return false;
        }
    }
    
    return true;
}

TensorData FedAvgServer::weighted_average_tensors(
    const std::vector<const TensorData*>& tensors,
    const std::vector<double>& weights) const {
    
    std::string error_msg;
    if (!are_all_tensors_compatible(tensors, &error_msg)) {
        throw std::runtime_error("Tensor compatibility check failed: " + error_msg);
    }
    
    const TensorData* first = tensors[0];
    size_t data_size = first->data.size();
    
    double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weight_sum <= compute_epsilon()) {
        throw std::runtime_error("Sum of weights is zero or negative");
    }
    
    TensorData result;
    result.name = first->name;
    result.shape = first->shape;
    result.data.resize(data_size, 0.0);
    
    for (size_t j = 0; j < data_size; ++j) {
        double sum = 0.0;
        double compensation = 0.0;
        
        for (size_t i = 0; i < tensors.size(); ++i) {
            double normalized_weight = weights[i] / weight_sum;
            double val = tensors[i]->data[j] * normalized_weight;
            
            double y = val - compensation;
            double t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        
        result.data[j] = sum;
        
        if (!std::isfinite(result.data[j])) {
            std::cerr << "Warning: Non-finite value detected in tensor " << result.name 
                      << " at index " << j << ", resetting to 0" << std::endl;
            result.data[j] = 0.0;
        }
    }
    
    return result;
}

TensorData FedAvgServer::safe_average_tensors(
    const std::vector<const TensorData*>& tensors,
    const std::vector<int>& sample_counts) const {
    
    if (tensors.size() != sample_counts.size()) {
        throw std::runtime_error("Mismatch between number of tensors and sample counts");
    }
    
    std::vector<double> weights;
    for (int count : sample_counts) {
        if (count <= 0) {
            throw std::runtime_error("Sample count must be positive");
        }
        weights.push_back(static_cast<double>(count));
    }
    
    return weighted_average_tensors(tensors, weights);
}

AggregationResult FedAvgServer::aggregate() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    AggregationResult result;
    result.success = false;
    result.num_clients_aggregated = 0;
    result.current_round = current_round_;
    
    if (client_models_.empty()) {
        result.message = "No client models to aggregate";
        std::cerr << "Aggregation failed: " << result.message << std::endl;
        return result;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Starting FedAvg aggregation for round " << current_round_ << std::endl;
    std::cout << "Number of clients: " << client_models_.size() << std::endl;
    
    std::vector<std::string> client_ids;
    std::vector<int> sample_counts;
    int total_samples = 0;
    
    for (const auto& pair : client_models_) {
        client_ids.push_back(pair.first);
        sample_counts.push_back(pair.second.sample_count);
        total_samples += pair.second.sample_count;
        std::cout << "  - Client " << pair.first << ": " 
                  << pair.second.sample_count << " samples" << std::endl;
    }
    std::cout << "Total samples across clients: " << total_samples << std::endl;
    
    if (total_samples <= 0) {
        result.message = "Total sample count is zero or negative";
        std::cerr << "Aggregation failed: " << result.message << std::endl;
        return result;
    }
    
    size_t num_weights = client_models_[client_ids[0]].weights.size();
    std::cout << "Number of weight tensors per client: " << num_weights << std::endl;
    
    global_weights_.clear();
    
    try {
        for (size_t w = 0; w < num_weights; ++w) {
            std::vector<const TensorData*> tensors;
            std::vector<int> client_sample_counts;
            
            for (size_t i = 0; i < client_ids.size(); ++i) {
                const auto& client_weights = client_models_[client_ids[i]].weights;
                
                if (w >= client_weights.size()) {
                    throw std::runtime_error(
                        "Client " + client_ids[i] + " has insufficient weights: " +
                        std::to_string(client_weights.size()) + " vs expected " + 
                        std::to_string(num_weights)
                    );
                }
                
                tensors.push_back(&client_weights[w]);
                client_sample_counts.push_back(sample_counts[i]);
            }
            
            TensorData avg = safe_average_tensors(tensors, client_sample_counts);
            global_weights_.push_back(avg);
            
            std::cout << "  Aggregated: " << avg.name 
                      << " (" << avg.data.size() << " elements)" << std::endl;
        }
    } catch (const std::exception& e) {
        result.message = std::string("Aggregation error: ") + e.what();
        std::cerr << "Aggregation failed: " << result.message << std::endl;
        return result;
    }
    
    client_models_.clear();
    
    result.success = true;
    result.message = "Aggregation completed successfully";
    result.num_clients_aggregated = static_cast<int>(client_ids.size());
    
    std::cout << "Aggregation completed successfully for round " << current_round_ << std::endl;
    std::cout << "========================================" << std::endl;
    
    current_round_++;
    result.current_round = current_round_;
    
    return result;
}

void FedAvgServer::reset_round() {
    std::lock_guard<std::mutex> lock(mutex_);
    client_models_.clear();
    std::cout << "Reset round state for round " << current_round_ << std::endl;
}

void FedAvgServer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_round_ = 1;
    client_models_.clear();
    global_weights_.clear();
    reference_weights_template_.clear();
    std::cout << "Server completely reset. Starting new session at round 1." << std::endl;
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
    result["has_global_weights"] = !global_weights_.empty();
    result["reference_template_set"] = !reference_weights_template_.empty();
    
    json client_list = json::array();
    int total_submitted_samples = 0;
    for (const auto& pair : client_models_) {
        total_submitted_samples += pair.second.sample_count;
        client_list.push_back({
            {"client_id", pair.first},
            {"sample_count", pair.second.sample_count},
            {"round", pair.second.round},
            {"num_weights", static_cast<int>(pair.second.weights.size())}
        });
    }
    result["clients"] = client_list;
    result["total_submitted_samples"] = total_submitted_samples;
    
    if (!reference_weights_template_.empty()) {
        json template_info = json::array();
        for (const auto& w : reference_weights_template_) {
            template_info.push_back({
                {"name", w.name},
                {"shape", w.shape},
                {"num_elements", w.data.size()}
            });
        }
        result["weight_template"] = template_info;
    }
    
    return result;
}

bool FedAvgServer::is_round_complete() const {
    if (expected_clients_ <= 0) {
        return !client_models_.empty();
    }
    return client_models_.size() >= static_cast<size_t>(expected_clients_);
}
