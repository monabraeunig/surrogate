#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "umbridge.h"

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
#include <future>

std::mutex mtx;

template<typename T>
std::string to_string(const std::vector<T>& vector) {
  std::string s;
  for (auto entry : vector)
      s += (s.empty() ? "" : ",") + std::to_string(entry);
  return s;
}


/*std::vector<std::vector<double>> generateRandomValues() {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-6, 6);
    
    std::vector<double> randomValues(2);
	randomValues[0] = dis(gen);
	randomValues[1] = dis(gen);
	
	
	std::vector<std::vector<double>> inputs {{randomValues[0], randomValues[1]}};
	
	return inputs;
}*/

std::vector<std::vector<double>> generateRandomValues() {
	
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-6, 6);
    
    std::vector<std::vector<double>> inputs(1000, std::vector<double>(2));
    for(int i = 0; i<1000; i++){
		inputs[i][0]= dis(gen);
		inputs[i][1] = dis(gen);
	}
	//std::cout << inputs << std::endl;
	return inputs;
}
		

int main(int argc, char** argv) {
  if (argc <= 1) {
    std::cout << "Expecting address to connect to as argument, e.g. http://localhost:4242" << std::endl;
    exit(-1);
  }
  std::string host = argv[1];
  // List supported models
  std::vector<std::string> models = umbridge::SupportedModels(host);
  // Connect to a model
  umbridge::HTTPModel client(host, "surrogate");
	const int numThreads = 100;
    const int requestsPerSecond = 1;
    //per Thread
    const int totalRequests = 10;
    
	std::vector<std::vector<double>> inputs = generateRandomValues();
	
    std::vector<std::thread> threads;
    std::mutex mtx; // Mutex to protect shared counter
    // Function to be executed by each thread
    auto threadFunction = [&mtx, &totalRequests, &requestsPerSecond, &client, &inputs](int threadId) {
		//std::thread::id threadId = std::this_thread::get_id(); // Get the thread ID
        for (int i = 0; i<1; i++){
			//std::cout << std::this_thread::get_id() << threadId << std::endl;
            // Send a request
            //std::vector<std::vector<double>> inputs = generateRandomValues();
            // generate Output
            std::vector<std::vector<double>> in(1,std::vector<double>(2));
            in[0][0]=inputs[threadId-1][0];
            in[0][1]=inputs[threadId-1][1];
            /*std::cout << "in: ";
				for (const auto& inner_vec : in) {
					for (const auto& value : inner_vec) {
						std::cout << value << ' ';
					}
				}*/
            std::vector<std::vector<double>> outputs = client.Evaluate(in);
            /*std::cout << "in: ";
				for (const auto& inner_vec : in) {
					for (const auto& value : inner_vec) {
						std::cout << value << ' ';
					}
				}
            std::cout << "out: ";
				for (const auto& inner_vec : outputs) {
					for (const auto& value : inner_vec) {
						std::cout << value << ' ';
					}
				}*/
            //std::cout << "outputs: " << outputs << std::endl;
        }
    };
	auto startstartTime = std::chrono::steady_clock::now();
    // Create threads
    for (int i = 0; i < numThreads; ++i) {
		//std::cout << "Thread " << i << std::endl;
        threads.emplace_back(threadFunction, i + 1);
    }
    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }
	auto endendTime = std::chrono::steady_clock::now();
	auto totalelapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endendTime - startstartTime);
	std::cout << "Zeit: " << totalelapsedTime.count() << std::endl;
    std::cout << "fertig" << std::endl;
}
