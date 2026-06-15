#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include "FastFiz.h" // FastFizをインクルード

namespace py = pybind11;
using namespace Pool;

class BilliardSimulator {
private:
    TableState ts; // FastFizの盤面クラス

    // 内部用のヘルパー関数：FastFizの座標(Point)から std::vector を作る
    std::vector<float> _get_state_vector() {
        Point cue_p = ts.getBall(Ball::CUE).getPos();
        Point obj0_p = ts.getBall(Ball::ONE).getPos();
        Point obj1_p = ts.getBall(Ball::TWO).getPos();
        return {
            static_cast<float>(cue_p.x), static_cast<float>(cue_p.y),
            static_cast<float>(obj0_p.x), static_cast<float>(obj0_p.y),
            static_cast<float>(obj1_p.x), static_cast<float>(obj1_p.y)
        };
    }

public:
    BilliardSimulator() { reset(); }

    void reset() {
        ts = TableState(); // 盤面をクリア
        
        // 【修正点】一度変数として球を作ってから、セットする
        Ball cue_ball(Ball::CUE, Ball::STATIONARY, 0.5, 0.5);
        Ball obj0_ball(Ball::ONE, Ball::STATIONARY, 0.5, 1.5);
        Ball obj1_ball(Ball::TWO, Ball::STATIONARY, 0.8, 1.5);
        
        ts.setBall(cue_ball);
        ts.setBall(obj0_ball);
        ts.setBall(obj1_ball);
    }

    void set_state(std::vector<float> state) {
        ts = TableState();
        
        // 【修正点】一度変数として球を作ってから、セットする
        Ball cue_ball(Ball::CUE, Ball::STATIONARY, state[0], state[1]);
        Ball obj0_ball(Ball::ONE, Ball::STATIONARY, state[2], state[3]);
        Ball obj1_ball(Ball::TWO, Ball::STATIONARY, state[4], state[5]);
        
        ts.setBall(cue_ball);
        ts.setBall(obj0_ball);
        ts.setBall(obj1_ball);
    }

    std::vector<float> get_state() {
        return _get_state_vector();
    }

    std::vector<float> step(int target_ball, float power, float angle) {
        float reward = 0.0f;
        float done = 1.0f; 

        // 1. Pythonからの入力(angle, power)をFastFizのShotParamsに変換
        ShotParams sp(0.0, 0.0, 10.0, angle, power);

        // 2. ショットが物理的に可能かチェック
        if (ts.isPhysicallyPossible(sp) != TableState::OK_PRECONDITION) {
            auto state = _get_state_vector();
            state.push_back(-1.0f); // ペナルティ
            state.push_back(1.0f);  
            return state;
        }

        // 3. FastFizでシミュレーションを実行！
        Shot* shot = ts.executeShot(sp);

        // 4. イベントリストを解析して報酬(Reward)を計算
        const std::vector<Event*>& events = shot->getEventList();
        Ball::Type target_id = (target_ball == 0) ? Ball::ONE : Ball::TWO;

        for (Event* e : events) {
            if (e->getType() == Event::POCKETED) {
                PocketedEvent* pe = static_cast<PocketedEvent*>(e);
                if (pe->getBall1() == target_id) {
                    reward = 1.0f; // ターゲットの球を落とせたら報酬1.0
                }
            }
        }

        delete shot; // メモリ解放

        // 5. 新しい座標を取得してPythonに返す
        auto next_state = _get_state_vector();
        next_state.push_back(reward);
        next_state.push_back(done);

        return next_state;
    }
};

PYBIND11_MODULE(billiard_env_cpp, m) {
    py::class_<BilliardSimulator>(m, "BilliardSimulator")
        .def(py::init<>())
        .def("reset", &BilliardSimulator::reset)
        .def("set_state", &BilliardSimulator::set_state)
        .def("get_state", &BilliardSimulator::get_state)
        .def("step", &BilliardSimulator::step);
}