"""
raccoon_env_pitch_real.py — SyncSimRaccoonEnv(로컬)에 pitch(손목 각도)를 살린 서브클래스
============================================================================================

원본 raccoon_env.py 는 그대로 두고 여기서 상속/오버라이드만 한다.

추가 수정:
  - Joint1 베이스 회전 속도만 기본 1/5로 감속한다.
  - lift/place down 구간에서는 chess_pick_place_dataset.py에서 set_motion_speed_scale()을 호출해
    Joint2~3도 함께 느리게 움직일 수 있게 한다.
  - Joint4 손목 lock 속도는 건드리지 않는다.
"""

import math

import numpy as np

from raccoon_env import SyncSimRaccoonEnv


class PitchRealRaccoonEnv(SyncSimRaccoonEnv):

    # 손목 피치 모드 (원본 FREE=0, HORZ=1, VERT=2 에 이어서)
    GRIP_MODE_PITCH = 3

    # Joint4(손목) 물리 범위 ±105°
    J4_LIMIT_DEG = 105.0

    # Joint1 베이스 회전만 기본 감속. 0.2 = 1/5 속도.
    BASE_ROT_SPEED_SCALE = 0.2

    # lift/place down 같은 특정 구간에서 전체 팔 이동 속도를 추가로 낮추기 위한 배율.
    # Joint4 손목 lock 속도에는 적용하지 않는다.
    _motion_speed_scale = 1.0

    # lockp()로 세팅되는 현재 pitch (deg)
    _active_pitch_deg = 0.0

    def set_motion_speed_scale(self, scale):
        """
        chess_pick_place_dataset.py에서 특정 waypoint 구간만 느리게 만들 때 사용한다.

        예:
          scale=1.0  -> 기본
          scale=0.4  -> 약 2.5배 느림
          scale=0.2  -> 약 5배 느림
        """
        self._motion_speed_scale = max(0.05, float(scale))

    def _joint_motion_scale(self, joint_index):
        """
        joint_index:
          0 = Joint1 베이스 회전
          1 = Joint2
          2 = Joint3
          3 = Joint4 손목

        Joint1은 항상 BASE_ROT_SPEED_SCALE을 적용한다.
        Joint2~3은 특정 구간에서 _motion_speed_scale을 적용한다.
        Joint4 손목 lock 속도는 건드리지 않는다.
        """
        if joint_index == 0:
            return self.BASE_ROT_SPEED_SCALE * self._motion_speed_scale

        if joint_index in (1, 2):
            return self._motion_speed_scale

        return 1.0

    # ---------- B. pitch 를 받는 IK ----------
    def _calc_inv_kinematics(self, x, y, z, pitch=0.0):
        if not (
            isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and isinstance(z, (int, float))
        ):
            return None

        if not ((-28.0 <= x <= 28.0) and (-15.0 <= y <= 28.0) and (0.0 <= z <= 36.25)):
            return None

        pr = math.radians(pitch)
        cosp = math.cos(pr)
        sinp = math.sin(pr)

        # 원본 convention
        x, y = y, -x

        th1 = math.atan2(y, x)
        c1 = math.cos(th1)
        s1 = math.sin(th1)

        # 손목 중심: 수평 L4 에 cos(pitch), z 에 +L4*sin(pitch)
        wx = x - self.L4 * cosp * c1
        wy = y - self.L4 * cosp * s1
        z_eff = z + self.L4 * sinp
        wz = z_eff - self.L1

        c3 = (
            wx * wx + wy * wy + wz * wz
            - self.L2 * self.L2
            - self.L3 * self.L3
        ) / (2.0 * self.L2 * self.L3)

        if c3 < -1.0001 or c3 > 1.0001:
            return None

        c3 = float(np.clip(c3, -1.0, 1.0))

        s3_abs = math.sqrt(max(0.0, 1.0 - c3 * c3))
        s3_candidates = [-s3_abs, s3_abs]

        th1_deg = math.degrees(th1)

        for s3 in s3_candidates:
            th3 = math.atan2(s3, c3)

            m1 = c3 * self.L3 + self.L2
            m2 = wz
            m3 = s3 * self.L3
            m4 = c1 * wx + s1 * wy

            c2 = m1 * m2 - m3 * m4
            s2 = -m2 * m3 - m1 * m4
            th2 = math.atan2(s2, c2)

            th2_deg = math.degrees(th2)
            th3_deg = math.degrees(th3)

            # pitch 반영
            th4_deg = -(th2_deg + th3_deg) - 90.0 - pitch

            if th1_deg < -120.0 or th1_deg > 120.0:
                continue
            if th2_deg < -90.0 or th2_deg > 30.0:
                continue
            if th3_deg < -150.0 or th3_deg > 0.0:
                continue
            if th4_deg < -self.J4_LIMIT_DEG or th4_deg > self.J4_LIMIT_DEG:
                continue

            return [th1_deg, th2_deg, th3_deg, th4_deg]

        return None

    # ---------- C. pitch 전달 ----------
    def move_to(self, x_cm, y_cm, z_cm, pitch=0.0, speed=70):
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm, pitch=pitch)
        if angles is None:
            raise ValueError(
                f"도달 불가 (또는 손목범위 초과): "
                f"({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm pitch={pitch:.1f}°"
            )
        self.degree_to([1, 2, 3, 4], angles[:4], speed)

    # ---------- D. pitch lock ----------
    def lockp(self, pitch_deg=0.0):
        """피치 모드 on. 이동 중에도 그리퍼를 이 pitch(지면 기준)로 유지."""
        self._active_pitch_deg = float(pitch_deg)
        self.gripper_mode = self.GRIP_MODE_PITCH

    def _apply_controls_once(self):
        dt = self.model.opt.timestep

        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])

                if self.gripper_mode == self.GRIP_MODE_HORZ:
                    desired = base_angle - np.radians(90)
                elif self.gripper_mode == self.GRIP_MODE_VERT:
                    desired = base_angle - np.radians(180)
                else:
                    desired = (
                        base_angle
                        - np.radians(90)
                        - np.radians(self._active_pitch_deg)
                    )

                error = desired - self.current_setpoints[i]

                # 손목 속도는 기존 그대로 유지한다.
                # Joint1 감속이나 lift/place down 감속을 여기에 적용하지 않는다.
                limit_step = 3.0 * self.MAX_SPEEDS[i] * dt
                step = np.clip(error, -limit_step, limit_step)
                self.current_setpoints[i] += step

            else:
                motion_scale = self._joint_motion_scale(i)

                if self.joint_control_mode[i] == self.MODE_VELOCITY:
                    self.current_setpoints[i] += self.joint_velocities[i] * dt * motion_scale
                else:
                    error = self.target_angles[i] - self.current_setpoints[i]
                    if abs(error) > 1e-4:
                        max_step = abs(self.joint_velocities[i]) * dt * motion_scale
                        step_val = np.clip(error, -max_step, max_step)
                        self.current_setpoints[i] += step_val

            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        # ----- gripper stop-on-contact -----
        try:
            touch_l = self.data.sensor("sensor_L").data[0]
            touch_r = self.data.sensor("sensor_R").data[0]
            is_touched = (touch_l > 0.1) and (touch_r > 0.1)
        except Exception:
            is_touched = False

        if self.gripper_target == self.GRIP_CLOSE and is_touched:
            self.gripper_target = self.data.qpos[4] - 0.028

        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_step = self.GRIPPER_SPEED * dt
            g_move = np.clip(g_err, -g_step, g_step)
            self.current_setpoints[4] += g_move

        self.data.ctrl[4] = self.current_setpoints[4]