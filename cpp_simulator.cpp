#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <tuple>
#include <utility>
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


class BilliardSimulatorMulti {
    TableState ts;
    std::vector<int> ids;
    bool ready = false;
    int first_contact = -1, precondition = 0;
    std::vector<std::tuple<double, int, int, int>> events;

    static Ball::Type type_for(int n) {
        if (n < 0 || n > 15) throw std::invalid_argument("ball number must be 0..15");
        return static_cast<Ball::Type>(n);
    }
    static Ball::State state_for(int p) {
        switch (p) {
            case -1: return Ball::STATIONARY;
            case 0: return Ball::POCKETED_SW;
            case 1: return Ball::POCKETED_W;
            case 2: return Ball::POCKETED_NW;
            case 3: return Ball::POCKETED_NE;
            case 4: return Ball::POCKETED_E;
            case 5: return Ball::POCKETED_SE;
            default: throw std::invalid_argument("pocket must be -1 or 0..5");
        }
    }
    bool has(int n) const {
        return std::find(ids.begin(), ids.end(), n) != ids.end();
    }
    void check_ready() const {
        if (!ready) throw std::logic_error("Call set_full_state before shooting");
    }
public:
    explicit BilliardSimulatorMulti(std::vector<int> numbers) : ids(std::move(numbers)) {
        if (ids.size() < 2 || ids.size() > 16 || ids.front() != 0)
            throw std::invalid_argument("Need cue 0 followed by 1..15 object balls");
        for (std::size_t i = 0; i < ids.size(); ++i) {
            type_for(ids[i]);
            for (std::size_t j = 0; j < i; ++j)
                if (ids[i] == ids[j])
                    throw std::invalid_argument("Duplicate ball number");
        }
    }
    std::vector<int> get_ball_numbers() const { return ids; }
    void set_full_state(const std::vector<float>& xy, const std::vector<int>& pockets) {
        if (xy.size() != 2 * ids.size() || pockets.size() != ids.size())
            throw std::invalid_argument("Wrong multi-ball state length");
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (!std::isfinite(xy[2*i]) || !std::isfinite(xy[2*i+1]))
                throw std::invalid_argument("Nonfinite ball position");
            state_for(pockets[i]);
        }
        TableState fresh;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            Ball b(type_for(ids[i]), state_for(pockets[i]), xy[2*i], xy[2*i+1]);
            fresh.setBall(b);
        }
        ts = fresh;
        ready = true;
        first_contact = -1;
        precondition = 0;
        events.clear();
    }
    std::vector<float> get_state() const {
        check_ready();
        std::vector<float> out;
        out.reserve(ids.size() * 2);
        for (int id : ids) {
            const Point& p = ts.getBall(type_for(id)).getPos();
            out.push_back(static_cast<float>(p.x));
            out.push_back(static_cast<float>(p.y));
        }
        return out;
    }
    int get_pocket_index(int number) const {
        check_ready();
        if (!has(number)) throw std::invalid_argument("Unknown ball number");
        switch (ts.getBall(type_for(number)).getState()) {
            case Ball::POCKETED_SW: return 0;
            case Ball::POCKETED_W: return 1;
            case Ball::POCKETED_NW: return 2;
            case Ball::POCKETED_NE: return 3;
            case Ball::POCKETED_E: return 4;
            case Ball::POCKETED_SE: return 5;
            default: return -1;
        }
    }
    int get_last_first_contact_ball() const { return first_contact; }
    int get_last_precondition_code() const { return precondition; }
    int get_ball_placement_code() const {
        check_ready();
        return ts.isValidBallPlacement();
    }
    std::vector<std::tuple<double, int, int, int>> get_last_events() const {
        return events;
    }
    std::vector<float> step(int target_ball, float power, float angle,
                            float tip_x_mm = 0.0f, float tip_y_mm = 0.0f,
                            float cue_elevation_deg = 10.0f) {
        check_ready();
        if (!has(target_ball) || target_ball == 0)
            throw std::invalid_argument("Target must be configured object ball");
        if (!std::isfinite(power) || !std::isfinite(angle)
            || !std::isfinite(tip_x_mm) || !std::isfinite(tip_y_mm)
            || !std::isfinite(cue_elevation_deg)
            || tip_x_mm*tip_x_mm + tip_y_mm*tip_y_mm > 144.0f)
            throw std::invalid_argument("Invalid shot parameters");
        first_contact = -1;
        precondition = 0;
        events.clear();
        ShotParams sp(tip_x_mm, tip_y_mm, cue_elevation_deg, angle, power);
        precondition = ts.isPhysicallyPossible(sp);
        if (precondition != TableState::OK_PRECONDITION) {
            auto out = get_state();
            out.push_back(-1.0f);
            out.push_back(1.0f);
            return out;
        }
        Shot* shot = ts.executeShot(sp);
        float diagnostic_reward = 0.0f;
        for (const Event* e : shot->getEventList()) {
            const int a = static_cast<int>(e->getBall1());
            const int b = static_cast<int>(e->getBall2());
            const int kind = static_cast<int>(e->getType());
            events.emplace_back(e->getTime(), kind, a, b);
            if (first_contact < 0 && e->getType() == Event::BALL_COLLISION) {
                if (a == 0 && b >= 1 && b <= 15) first_contact = b;
                if (b == 0 && a >= 1 && a <= 15) first_contact = a;
            }
            if (e->getType() == Event::POCKETED && a == target_ball)
                diagnostic_reward = 1.0f;
        }
        delete shot;
        auto out = get_state();
        out.push_back(diagnostic_reward);
        out.push_back(1.0f);
        return out;
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
    py::class_<BilliardSimulatorMulti>(m, "BilliardSimulatorMulti")
        .def(py::init<std::vector<int>>())
        .def("get_ball_numbers", &BilliardSimulatorMulti::get_ball_numbers)
        .def("set_full_state", &BilliardSimulatorMulti::set_full_state)
        .def("get_state", &BilliardSimulatorMulti::get_state)
        .def("get_pocket_index", &BilliardSimulatorMulti::get_pocket_index)
        .def("get_last_first_contact_ball", &BilliardSimulatorMulti::get_last_first_contact_ball)
        .def("get_last_precondition_code", &BilliardSimulatorMulti::get_last_precondition_code)
        .def("get_ball_placement_code", &BilliardSimulatorMulti::get_ball_placement_code)
        .def("get_last_events", &BilliardSimulatorMulti::get_last_events)
        .def("step", &BilliardSimulatorMulti::step,
             py::arg("target_ball"), py::arg("power"), py::arg("angle"),
             py::arg("tip_x_mm") = 0.0f, py::arg("tip_y_mm") = 0.0f,
             py::arg("cue_elevation_deg") = 10.0f);
}
