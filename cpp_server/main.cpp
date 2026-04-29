#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include "FedAvgServer.h"

using json = nlohmann::json;

FedAvgServer g_server;

std::vector<TensorData> json_to_tensors(const json& weights_json) {
    if (!weights_json.is_array()) {
        throw std::runtime_error("weights must be an array");
    }
    
    std::vector<TensorData> tensors;
    
    for (const auto& w : weights_json) {
        TensorData t;
        
        if (!w.contains("name")) {
            throw std::runtime_error("Tensor missing 'name' field");
        }
        t.name = w["name"].get<std::string>();
        
        if (w.contains("shape")) {
            if (!w["shape"].is_array()) {
                throw std::runtime_error("Tensor 'shape' must be an array for: " + t.name);
            }
            t.shape = w["shape"].get<std::vector<size_t>>();
        }
        
        if (!w.contains("data")) {
            throw std::runtime_error("Tensor missing 'data' field for: " + t.name);
        }
        
        if (w["data"].is_array()) {
            t.data = w["data"].get<std::vector<double>>();
        } else if (w["data"].is_number()) {
            t.data.push_back(w["data"].get<double>());
        } else {
            throw std::runtime_error("Tensor 'data' must be array or number for: " + t.name);
        }
        
        if (t.data.empty()) {
            throw std::runtime_error("Tensor 'data' cannot be empty for: " + t.name);
        }
        
        if (!t.shape.empty()) {
            size_t expected_size = 1;
            for (size_t s : t.shape) expected_size *= s;
            if (t.data.size() != expected_size) {
                throw std::runtime_error(
                    "Tensor " + t.name + " size mismatch: shape indicates " + 
                    std::to_string(expected_size) + " elements but data has " + 
                    std::to_string(t.data.size())
                );
            }
        }
        
        tensors.push_back(t);
    }
    
    return tensors;
}

int main() {
    httplib::Server svr;
    
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type"}
    });
    
    svr.Options("/.*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });
    
    svr.Post("/api/submit", [](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                throw std::runtime_error("Empty request body");
            }
            
            json body = json::parse(req.body);
            
            if (!body.contains("client_id")) {
                throw std::runtime_error("Missing 'client_id' field");
            }
            if (!body.contains("sample_count")) {
                throw std::runtime_error("Missing 'sample_count' field");
            }
            if (!body.contains("round")) {
                throw std::runtime_error("Missing 'round' field");
            }
            if (!body.contains("weights")) {
                throw std::runtime_error("Missing 'weights' field");
            }
            
            std::string client_id = body["client_id"].get<std::string>();
            int sample_count = body["sample_count"].get<int>();
            int round = body["round"].get<int>();
            
            if (client_id.empty()) {
                throw std::runtime_error("'client_id' cannot be empty");
            }
            if (sample_count <= 0) {
                throw std::runtime_error("'sample_count' must be positive");
            }
            if (round <= 0) {
                throw std::runtime_error("'round' must be positive");
            }
            
            std::vector<TensorData> weights = json_to_tensors(body["weights"]);
            
            bool success = g_server.submit_weights(client_id, sample_count, round, weights);
            
            if (success) {
                json response;
                response["status"] = "success";
                response["message"] = "Weights submitted successfully";
                response["client_id"] = client_id;
                response["sample_count"] = sample_count;
                response["round"] = round;
                response["num_weights"] = static_cast<int>(weights.size());
                
                res.set_content(response.dump(), "application/json");
            } else {
                json response;
                response["status"] = "error";
                response["message"] = "Failed to submit weights. Round mismatch or validation failed.";
                response["expected_round"] = g_server.get_current_round();
                response["received_round"] = round;
                res.status = 400;
                res.set_content(response.dump(), "application/json");
            }
        } catch (const std::exception& e) {
            std::cerr << "Submit error: " << e.what() << std::endl;
            json response;
            response["status"] = "error";
            response["message"] = std::string("Invalid request: ") + e.what();
            res.status = 400;
            res.set_content(response.dump(), "application/json");
        }
    });
    
    svr.Get("/api/weights", [](const httplib::Request&, httplib::Response& res) {
        try {
            json weights = g_server.get_aggregated_weights();
            weights["status"] = "success";
            res.set_content(weights.dump(), "application/json");
        } catch (const std::exception& e) {
            json response;
            response["status"] = "error";
            response["message"] = e.what();
            res.status = 500;
            res.set_content(response.dump(), "application/json");
        }
    });
    
    svr.Get("/api/status", [](const httplib::Request&, httplib::Response& res) {
        try {
            json status = g_server.get_status();
            res.set_content(status.dump(), "application/json");
        } catch (const std::exception& e) {
            json response;
            response["status"] = "error";
            response["message"] = e.what();
            res.status = 500;
            res.set_content(response.dump(), "application/json");
        }
    });
    
    svr.Post("/api/aggregate", [](const httplib::Request&, httplib::Response& res) {
        try {
            AggregationResult result = g_server.aggregate();
            
            json response;
            if (result.success) {
                response["status"] = "success";
                response["message"] = result.message;
                response["num_clients_aggregated"] = result.num_clients_aggregated;
                response["current_round"] = result.current_round;
                res.set_content(response.dump(), "application/json");
            } else {
                response["status"] = "error";
                response["message"] = result.message;
                response["current_round"] = result.current_round;
                res.status = 400;
                res.set_content(response.dump(), "application/json");
            }
        } catch (const std::exception& e) {
            json response;
            response["status"] = "error";
            response["message"] = e.what();
            res.status = 500;
            res.set_content(response.dump(), "application/json");
        }
    });
    
    svr.Post("/api/reset_round", [](const httplib::Request&, httplib::Response& res) {
        try {
            g_server.reset_round();
            
            json response;
            response["status"] = "success";
            response["message"] = "Round state reset successfully";
            response["current_round"] = g_server.get_current_round();
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json response;
            response["status"] = "error";
            response["message"] = e.what();
            res.status = 500;
            res.set_content(response.dump(), "application/json");
        }
    });
    
    svr.Post("/api/config", [](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                throw std::runtime_error("Empty request body");
            }
            
            json body = json::parse(req.body);
            
            if (body.contains("expected_clients")) {
                int expected = body["expected_clients"].get<int>();
                if (expected < 0) {
                    throw std::runtime_error("expected_clients cannot be negative");
                }
                g_server.set_expected_clients(expected);
            }
            
            json response;
            response["status"] = "success";
            response["message"] = "Configuration updated";
            response["expected_clients"] = g_server.get_expected_clients();
            response["current_round"] = g_server.get_current_round();
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json response;
            response["status"] = "error";
            response["message"] = e.what();
            res.status = 400;
            res.set_content(response.dump(), "application/json");
        }
    });
    
    svr.Post("/api/validate_weights", [](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.body.empty()) {
                throw std::runtime_error("Empty request body");
            }
            
            json body = json::parse(req.body);
            
            if (!body.contains("weights")) {
                throw std::runtime_error("Missing 'weights' field");
            }
            
            std::vector<TensorData> weights = json_to_tensors(body["weights"]);
            
            std::string error_msg;
            bool valid = g_server.validate_weights(weights, &error_msg);
            
            json response;
            if (valid) {
                response["status"] = "success";
                response["message"] = "Weights are valid";
                response["num_weights"] = static_cast<int>(weights.size());
            } else {
                response["status"] = "error";
                response["message"] = error_msg;
            }
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json response;
            response["status"] = "error";
            response["message"] = e.what();
            res.status = 400;
            res.set_content(response.dump(), "application/json");
        }
    });
    
    svr.Post("/api/reset", [](const httplib::Request&, httplib::Response& res) {
        try {
            g_server.reset();
            
            json response;
            response["status"] = "success";
            response["message"] = "Server reset successfully";
            response["current_round"] = g_server.get_current_round();
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            json response;
            response["status"] = "error";
            response["message"] = e.what();
            res.status = 500;
            res.set_content(response.dump(), "application/json");
        }
    });
    
    std::cout << "========================================" << std::endl;
    std::cout << "  FedAvg Aggregation Server" << std::endl;
    std::cout << "  (Enhanced with validation & precision)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Server starting on http://localhost:8080" << std::endl;
    std::cout << "Available endpoints:" << std::endl;
    std::cout << "  POST /api/submit          - Submit client weights" << std::endl;
    std::cout << "  GET  /api/weights         - Get aggregated weights" << std::endl;
    std::cout << "  GET  /api/status          - Get server status" << std::endl;
    std::cout << "  POST /api/aggregate       - Trigger aggregation" << std::endl;
    std::cout << "  POST /api/reset_round     - Reset current round state" << std::endl;
    std::cout << "  POST /api/config          - Configure server" << std::endl;
    std::cout << "  POST /api/validate_weights- Validate weights format" << std::endl;
    std::cout << "  POST /api/reset           - Reset server state completely" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Enhancements:" << std::endl;
    std::cout << "  - Kahan summation for numerical stability" << std::endl;
    std::cout << "  - Full tensor validation (shape, NaN, Inf)" << std::endl;
    std::cout << "  - Reference template for shape consistency" << std::endl;
    std::cout << "  - client_models cleared after aggregation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    svr.listen("0.0.0.0", 8080);
    
    return 0;
}
