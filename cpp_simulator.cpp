#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <cstdlib>

namespace py = pybind11;

class BilliardSimulator {
private:
    float cue_x, cue_y; 
    float obj0_x, obj0_y; // 的球0
    float obj1_x, obj1_y; // 的球1
    float pocket_x = 1.0f, pocket_y = 1.0f;

public:
    BilliardSimulator() { reset(); }

    void reset() {
        cue_x = 0.0f; cue_y = 0.0f;
        obj0_x = 0.5f; obj0_y = 0.5f;
        obj1_x = -0.5f; obj1_y = 0.5f;
    }

    void set_state(std::vector<float> state) {
        cue_x = state[0]; cue_y = state[1];
        obj0_x = state[2]; obj0_y = state[3];
        obj1_x = state[4]; obj1_y = state[5];
    }

    std::vector<float> get_state() {
        return {cue_x, cue_y, obj0_x, obj0_y, obj1_x, obj1_y};
    }

    // 【変更点】引数に target_ball (0 か 1) を追加
    std::vector<float> step(int target_ball, float power, float angle) {
        float target_x = (target_ball == 0) ? obj0_x : obj1_x;
        float target_y = (target_ball == 0) ? obj0_y : obj1_y;
        
        float reward = 0.0f;
        float done = 0.0f; 
        float r = 0.05f;

        float dx_op = pocket_x - target_x;
        float dy_op = pocket_y - target_y;
        float dist_op = std::sqrt(dx_op * dx_op + dy_op * dy_op);

        float ghost_x = target_x - (dx_op / dist_op) * (2.0f * r);
        float ghost_y = target_y - (dy_op / dist_op) * (2.0f * r);

        float ideal_angle = atan2(ghost_y - cue_y, ghost_x - cue_x) * 180.0f / M_PI;
        
        if (std::abs(angle - ideal_angle) < 2.0f && power > 0.1f) {
            // 成功したら狙った球をポケットへ
            if(target_ball == 0) { obj0_x = pocket_x; obj0_y = pocket_y; }
            else                 { obj1_x = pocket_x; obj1_y = pocket_y; }
            reward = 1.0f;    
            done = 1.0f;      
        } else {
            cue_x += power * cos(angle * M_PI / 180.0f) * 0.1f;
            cue_y += power * sin(angle * M_PI / 180.0f) * 0.1f;
            reward = 0.0f;
            done = 1.0f; 
        }

        return {cue_x, cue_y, obj0_x, obj0_y, obj1_x, obj1_y, reward, done};
    }
};

// 【セクション3】Pythonへのバインディング
PYBIND11_MODULE(billiard_env_cpp, m) {
    py::class_<BilliardSimulator>(m, "BilliardSimulator")
        .def(py::init<>())
        .def("reset", &BilliardSimulator::reset)
        .def("set_state", &BilliardSimulator::set_state)
        .def("get_state", &BilliardSimulator::get_state)
        .def("step", &BilliardSimulator::step);
}
