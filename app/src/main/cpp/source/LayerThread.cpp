//
// Created by drp on 16.03.24.
//

#include "../header/LayerThread.h"
#include "../header/Network.h"

namespace CNN {
    LayerThread::LayerThread(CNN::Network* network_ptr, size_t start, size_t end, const int affinity):
    network_ptr(network_ptr), start(start), end(end), core_id(affinity){}


    void LayerThread::set_successor_thread(CNN::LayerThread *successor_thread_arg) {
        this->successor_thread = successor_thread_arg;
    }


    void LayerThread::set_predecessor_thread(CNN::LayerThread *predecessor_thread_arg) {
        this->predecessor_thread = predecessor_thread_arg;
    }


    void LayerThread::start_thread() {
        thread = std::thread([this] {
            set_thread_affinity();
            this->network_ptr->configure_layer_thread(start, end);
            this->run_forward_acl();
        });
    }


    void LayerThread::join() {
        thread.join();
    }


    void LayerThread::input_ready_signal() {
        input_buffer_ready = true;
        cv.notify_one();
    }


    void LayerThread::output_ready_signal() {
        output_buffer_ready = true;
        cv.notify_one();
    }


    // Call this method to stop the thread loop
    void LayerThread::request_stop() {
        std::unique_lock<std::mutex> lock(mutex);
        stop_requested.store(true);
        cv.notify_one();
    }


    // Function to set thread affinity
    void LayerThread::set_thread_affinity() {
        // CPU set to specify the CPUs on which the thread will be eligible to run
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(this->core_id, &cpuset);

        // Setting the affinity of the thread to the specified CPU core
        const int set_result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        if (set_result != 0) {
            LOGI_NETWORK("AffinityError sched_setaffinity failed: %s", strerror(errno));
        }
    }


    void LayerThread::run_forward_acl() {
        while (!stop_requested.load()) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return (input_buffer_ready) || stop_requested.load(); });
                start_ts = std::chrono::high_resolution_clock::now();
                input_buffer_ready = false; // Reset ready state for the next signal
                if (stop_requested.load()) break; // Exit if stop requested
            }

            // Perform the layer's computation
            //auto start_temp = std::chrono::high_resolution_clock::now();
            for (int i = start; i < end; i++) {
                if (i == end-1) {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [this] { return (output_buffer_ready) || stop_requested.load();} );
                    output_buffer_ready = false;
                }
                network_ptr->forward_acl_index(i);
                if (i == start) {
                    // Wake up the successor, if any
                    if (predecessor_thread) {
                        predecessor_thread->output_ready_signal();
                    } else {
                        network_ptr->signal_input_ready();
                        this->input_buffer_ready = true;
                    }
                }
            }

            if (successor_thread) {
                successor_thread->input_ready_signal();
            } else {
                network_ptr->signal_output_ready();
            }

            auto output_ts_new = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(output_ts_new - start_ts).count();
            auto delay = std::chrono::duration_cast<std::chrono::microseconds>(output_ts_new - end_ts).count() - duration;
            end_ts = output_ts_new;
            LOGI_NETWORK("execution time thread %zu -> %zu: %f (ms),    delay = %lld", start, end, ((double)(duration/1000.0)), delay);
        }
    }
} // CNN