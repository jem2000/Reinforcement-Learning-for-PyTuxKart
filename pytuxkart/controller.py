import pystk


def acc_control(current_vel, aim_point):
    base_acc = 1
    turn_index = abs(aim_point[0])
    acc = (-(0.94 * turn_index ** 2) + 1) * base_acc
    brake = True if (turn_index * current_vel ** 2 > 86) else False
    acc = acc if not brake else 0
    return acc / 2, brake


def steering_control(aim_point):
    turn_index = aim_point[0]
    steering = -(turn_index ** 2) + 2 * turn_index if turn_index > 0 else (turn_index ** 2) + 2 * turn_index
    drift = False if abs(steering) < 0.29 else True
    return steering, drift


def nitro_control(aim_point, current_vel):
    if abs(aim_point[0]) < 0.1 and current_vel > 4:
        return True
    else:
        return False


def rl_control(aim_point, current_vel, override, override_val, override2, override_val2):
    action = pystk.Action()

    action.steer, action.drift = steering_control(aim_point)
    action.acceleration, brake = acc_control(current_vel, aim_point)
    action.nitro = nitro_control(aim_point, current_vel)
    action.brake = brake or action.drift

    setattr(action, override, override_val)
    # setattr(action, override2, (override_val2 + 1) * 0.5)

    return action


def control(aim_point, current_vel):
    action = pystk.Action()

    action.steer, action.drift = steering_control(aim_point)
    action.acceleration, brake = acc_control(current_vel, aim_point)
    action.nitro = nitro_control(aim_point, current_vel)
    action.brake = brake or action.drift

    return action


if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser


    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
