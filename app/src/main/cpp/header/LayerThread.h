//
// Created by drp on 16.03.24.
//

#ifndef ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYERTHREAD_H
#define ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYERTHREAD_H
#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>


namespace CNN {
    //handle circular dependency
    class Network;

    class LayerThread {
    public:
        size_t start, end;
        LayerThread(CNN::Network* network_ptr, const size_t start, const size_t end, const int affinity);
        void set_successor_thread(CNN::LayerThread *successor_thread_arg);
        void set_predecessor_thread(CNN::LayerThread *predecessor_thread_arg);
        void start_thread();
        void join();
        void input_ready_signal();
        void output_ready_signal();
        void request_stop();
        void set_thread_affinity();


    private:
        int core_id;
        CNN::Network *network_ptr;
        CNN::LayerThread *successor_thread = nullptr, *predecessor_thread = nullptr;
        std::thread thread;
        std::mutex mutex;
        std::condition_variable cv;
        std::function<void()> computation;
        std::atomic<bool> input_buffer_ready = false, stop_requested = false, output_buffer_ready=true;
        std::chrono::high_resolution_clock::time_point start_ts, end_ts;
        void run_forward_acl();
    };

} // CNN

#endif //ARMV8A_ARCHITECTURE_OPTIMIZATION_DEEP_LEARNING_LAYERTHREAD_H
