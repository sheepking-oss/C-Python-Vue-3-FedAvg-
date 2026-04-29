#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include "FedAvgServer.h"

using json = nlohmann::json;

FedAvgServer g_server;

std::vector<TensorData> json_to_tensors(const json& weights_json) {
    std::vector<TensorData> tensors;
    
    for (const auto& w : weights_json) {
        TensorData t;
        t.name = w["name"].get<std::string>();
        
        if (w.contains("shape")) {
            t.shape = w["shape"].get<std::vector<size_t>>();
        }
        
        if (w["data"].is_array()) {
            t.data = w["data"].get<std::vector<double>>();
        } else if (w["data"].is_number()) {
            t.data.push_back(w["data"].get<double>());
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
            json body = json::parse(req.body);
            
            std::string client_id = body["client_id"].get<std::string>();
            int sample_count = body["sample_count"].get<int>();
            int round = body["round"].get<int>();
            
            std::vector<TensorData> weights = json_to_tensors(body["weights"]);
            
            bool success = g_server.submit_weights(client_id, sample_count, round, weights);
            
            if (success) {
                json response;
                response["status"] = "success";
                response["message"] = "Weights submitted successfully";
                response["round"] = round;
                
                res.set_content(response.dump(), "application/json");
            } else {
                json response;
                response["status"] = "error";
                response["message"] = "Failed to submit weights. Round mismatch or invalid data.";
                res.status = 400;
                res.set_content(response.dump(), "application/json");
            }
        } catch (const std::exception& e) {
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
            bool success = g_server.aggregate();
            
            if (success) {
                json response;
                response["status"] = "success";
                response["message"] = "Aggregation completed successfully";
                response["current_round"] = g_server.get_current_round();
                res.set_content(response.dump(), "application/json");
            } else {
                json response;
                response["status"] = "error";
                response["message"] = "Failed to aggregate weights. No client models available.";
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
    
    svr.Post("/api/config", [](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::parse(req.body);
            
            if (body.contains("expected_clients")) {
                g_server.set_expected_clients(body["expected_clients"].get<int>());
            }
            
            json response;
            response["status"] = "success";
            response["message"] = "Configuration updated";
            response["expected_clients"] = g_server.get_expected_clients();
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
    std::cout << "========================================" << std::endl;
    std::cout << "Server starting on http://localhost:8080" << std::endl;
    std::cout << "Available endpoints:" << std::endl;
    std::cout << "  POST /api/submit    - Submit client weights" << std::endl;
    std::cout << "  GET  /api/weights   - Get aggregated weights" << std::endl;
    std::cout << "  GET  /api/status    - Get server status" << std::endl;
    std::cout << "  POST /api/aggregate - Trigger aggregation" << std::endl;
    std::cout << "  POST /api/config    - Configure server" << std::endl;
    std::cout << "  POST /api/reset     - Reset server state" << std::endl;
    std::cout << "========================================" << std::endl;
    
    svr.listen("0.0.0.0", 8080);
    
    return 0;
}
