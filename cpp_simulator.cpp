#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "FastFiz.h" // FastFizをインクルード

namespace py = pybind11;
using namespace Pool;

class BilliardSimulator {
private:
    TableState ts; // FastFizの盤面クラス
    int last_first_contact_ball = -1;
    int last_precondition_code = 0;

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

    Ball::State _state_from_pocket_index(int pocket_index) {
        switch (pocket_index) {
            case -1:
                return Ball::STATIONARY;

            case 0:
                return Ball::POCKETED_SW;

            case 1:
                return Ball::POCKETED_W;

            case 2:
                return Ball::POCKETED_NW;

            case 3:
                return Ball::POCKETED_NE;

            case 4:
                return Ball::POCKETED_E;

            case 5:
                return Ball::POCKETED_SE;

            default:
                throw std::invalid_argument(
                    "pocket_index must be -1 or 0..5"
                );
        }
    }

public:
    BilliardSimulator() { reset(); }

    void reset() {
        last_first_contact_ball = -1;
        last_precondition_code = 0;
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
        last_first_contact_ball = -1;
        last_precondition_code = 0;
        ts = TableState();
        
        // 【修正点】一度変数として球を作ってから、セットする
        Ball cue_ball(Ball::CUE, Ball::STATIONARY, state[0], state[1]);
        Ball obj0_ball(Ball::ONE, Ball::STATIONARY, state[2], state[3]);
        Ball obj1_ball(Ball::TWO, Ball::STATIONARY, state[4], state[5]);
        
        ts.setBall(cue_ball);
        ts.setBall(obj0_ball);
        ts.setBall(obj1_ball);
    }

    void set_full_state(
        std::vector<float> positions,
        std::vector<int> pocket_indices
    ){
        if (positions.size() != 6) {
            throw std::invalid_argument(
                "positions must contain 6 values"
            );
        }

        if (pocket_indices.size() != 3) {
            throw std::invalid_argument(
                "pocket_indices must contain 3 values"
            );
        }

        last_first_contact_ball = -1;
        last_precondition_code = 0;
        ts = TableState();

        Ball cue_ball(
            Ball::CUE,
            _state_from_pocket_index(pocket_indices[0]),
            positions[0],
            positions[1]
        );

        Ball obj0_ball(
            Ball::ONE,
            _state_from_pocket_index(pocket_indices[1]),
            positions[2],
            positions[3]
        );

        Ball obj1_ball(
            Ball::TWO,
            _state_from_pocket_index(pocket_indices[2]),
            positions[4],
            positions[5]
        );

        ts.setBall(cue_ball);
        ts.setBall(obj0_ball);
        ts.setBall(obj1_ball);
    }

    std::vector<float> get_state() {
        return _get_state_vector();
    }

    int get_pocket_index(int ball_id) {
        Ball::Type id;

        // Python側:
        // -1 = cue ball
        //  0 = object ball 0 (ONE)
        //  1 = object ball 1 (TWO)
        if (ball_id == -1) {
            id = Ball::CUE;
        } else if (ball_id == 0) {
            id = Ball::ONE;
        } else if (ball_id == 1) {
            id = Ball::TWO;
        } else {
            throw std::invalid_argument(
                "ball_id must be -1, 0, or 1"
            );
        }

        Ball::State state = ts.getBall(id).getState();

        switch (state) {
            case Ball::POCKETED_SW:
                return 0;

            case Ball::POCKETED_W:
                return 1;

            case Ball::POCKETED_NW:
                return 2;

            case Ball::POCKETED_NE:
                return 3;

            case Ball::POCKETED_E:
                return 4;

            case Ball::POCKETED_SE:
                return 5;

            default:
                return -1;
        }
    }

    int get_last_first_contact_ball() const { return last_first_contact_ball; }

    // Raw FastFiz bitmask. In particular 128 = CUE_STICK_COLLISION.
    int get_last_precondition_code() const { return last_precondition_code; }
    int get_ball_placement_code() const { return ts.isValidBallPlacement(); }

    std::vector<float> step(int target_ball, float power, float angle, float tip_x_mm = 0.0f, float tip_y_mm = 0.0f, float cue_elevation_deg = 10.0f) {
        last_first_contact_ball = -1;
        last_precondition_code = 0;
        if (target_ball != 0 && target_ball != 1) throw std::invalid_argument("target_ball must be 0 or 1");
        if (!std::isfinite(tip_x_mm) || !std::isfinite(tip_y_mm) || tip_x_mm * tip_x_mm + tip_y_mm * tip_y_mm > 144.0f) throw std::invalid_argument("tip must be within 12 mm radius");
        float reward = 0.0f;
        float done = 1.0f; 

        // 1. Pythonからの入力(angle, power)をFastFizのShotParamsに変換
        ShotParams sp(tip_x_mm, tip_y_mm, cue_elevation_deg, angle, power);

        // 2. ショットが物理的に可能かチェック
        last_precondition_code = ts.isPhysicallyPossible(sp);
        if (last_precondition_code != TableState::OK_PRECONDITION) {
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
            if (last_first_contact_ball == -1 && e->getType() == Event::BALL_COLLISION) {
                Ball::Type b1 = e->getBall1();
                Ball::Type b2 = e->getBall2();
                Ball::Type hit = (b1 == Ball::CUE) ? b2 : ((b2 == Ball::CUE) ? b1 : Ball::UNKNOWN_ID);
                if (hit == Ball::ONE) last_first_contact_ball = 0;
                if (hit == Ball::TWO) last_first_contact_ball = 1;
            }
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
        .def("set_full_state", &BilliardSimulator::set_full_state)
        .def("get_state", &BilliardSimulator::get_state)
        .def("get_pocket_index", &BilliardSimulator::get_pocket_index)
        .def("get_last_first_contact_ball", &BilliardSimulator::get_last_first_contact_ball)
        .def("get_last_precondition_code", &BilliardSimulator::get_last_precondition_code)
        .def("get_ball_placement_code", &BilliardSimulator::get_ball_placement_code)
        .def("step", &BilliardSimulator::step,
             py::arg("target_ball"), py::arg("power"), py::arg("angle"),
             py::arg("tip_x_mm") = 0.0f, py::arg("tip_y_mm") = 0.0f,
             py::arg("cue_elevation_deg") = 10.0f);
}