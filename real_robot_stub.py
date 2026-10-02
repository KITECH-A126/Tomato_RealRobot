"""NERO·PiPER 관절 및 Orbbec 305·335 카메라의 SDK 인터페이스.

정본 규약: `02_규약_실물_구동_인터페이스 v2_09_26`.

이 파일은 클래스와 check_shapes 함수 정의를 제공한다. import만 하면 연결하거나 검사하지 않는다.
직접 실행하면 RealRobot()을 생성하고 check_shapes를 실행한다(실물 위치 지령 포함).
RealRobot() 생성은 CAN에 연결하고 최종 NERO 드라이버의 속도 비율을 100%로 설정한다.
초기화에서 Enable/위치 지령은 보내지 않는다.
__init__에서 305/335를 각각 자동 감지하고 발견한 카메라만 스트림을 시작한다.
없는 카메라는 건너뛰지만, 발견한 카메라의 연결 실패는 초기화 오류로 알린다.
ROS는 사용하지 않는다. 카메라 감지를 위해 pyorbbecsdk는 설치되어 있어야 한다.
그리퍼는 미구현이다. 시각 통계·RGB 실물 검사는 별도 작업이며 이 층에서 하지 않는다.
"""
from collections import deque
import time

import numpy as np

# 규약의 URDF 관절 순서.
NERO_JOINTS = tuple(f"nero_joint{i}" for i in range(1, 8))    # 7개
PIPER_JOINTS = tuple(f"piper_joint{i}" for i in range(1, 7))  # 6개


class RealRobot:
    """두 팔의 관절 읽기·쓰기와 손목·시선 카메라 프레임을 제공한다."""

    def __init__(self, **conn):
        """연결하고 NERO 속도 비율을 100%로 설정한다. Enable/위치 지령은 보내지 않는다.
        conn: nero_can_port, piper_can_port, wrist_serial, gaze_serial만 허용한다.
        """
        from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile

        self._arms = {}
        self._configs = {}
        self._sample_times = {}
        self._closed = False
        self._camera_sdk = None
        self._camera_context = None
        self._camera_streams = {}
        self._camera_serials = {key: conn.get(key) for key in ("wrist_serial", "gaze_serial")}
        unknown = set(conn) - {"nero_can_port", "piper_can_port", "wrist_serial", "gaze_serial"}
        if unknown:
            raise ValueError(f"지원하지 않는 연결 설정: {sorted(unknown)}")
        for key, serial in self._camera_serials.items():
            if serial is not None and (not isinstance(serial, str) or not serial.strip()):
                raise ValueError(f"{key}: 비어 있지 않은 시리얼 문자열이 필요하다")
        ports = {arm: conn.get(f"{arm}_can_port", f"can_{arm}")
                 for arm in ("nero", "piper")}
        if any(not isinstance(port, str) or not port.strip() for port in ports.values()):
            raise ValueError("CAN 포트는 비어 있지 않은 문자열이어야 한다")
        if ports["nero"] == ports["piper"]:
            raise ValueError("두 팔은 서로 다른 CAN 포트를 사용해야 한다")
        try:
            for arm, names in (("nero", NERO_JOINTS), ("piper", PIPER_JOINTS)):
                config = create_agx_arm_config(robot=arm, comm="can", channel=ports[arm])
                driver = AgxArmFactory.create_arm(config)
                self._arms[arm] = driver
                driver.connect()
                # PiPER SDK는 get_fps() <= 0이면 펌웨어 응답 버퍼를 비운다.
                # 최초 CAN 수신 준비를 확인한 뒤 한 번만 펌웨어를 요청한다.
                # 초기 연결 준비일 뿐, 명령 재시도나 제어 주기 조절이 아니다.
                deadline = time.monotonic() + 2.0
                while True:
                    fps = driver.get_fps()
                    if np.isfinite(fps) and fps > 0:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"{arm} ({ports[arm]}): 2초 안에 CAN 수신이 준비되지 않았다")
                    time.sleep(0.01)
                # 초기 연결에서만 펌웨어 응답 대기. Enable/위치 명령은 보내지 않는다.
                firmware = driver.get_firmware(timeout=1.0)
                if not firmware:
                    raise RuntimeError(
                        f"{arm} ({ports[arm]}): CAN 수신은 준비됐지만 1초 안에 "
                        "완전한 펌웨어 응답을 받지 못했다. CAN 상태와 동시 조회 프로세스를 확인하세요.")
                profile = resolve_firmware_profile(arm, firmware["software_version"])
                if profile != config["firmeware_version"]:
                    driver.disconnect()
                    del self._arms[arm]
                    config = create_agx_arm_config(
                        robot=arm, comm="can", channel=ports[arm], firmeware_version=profile)
                    driver = AgxArmFactory.create_arm(config)
                    self._arms[arm] = driver
                    driver.connect()
                if arm == "nero":
                    driver.set_speed_percent(100)
                expected = [name.split("_", 1)[1] for name in names]
                if list(config["joint_names"]) != expected or driver.joint_nums != len(names):
                    raise RuntimeError(f"{arm}: SDK 관절 순서/개수가 규약과 다르다")
                if list(config["joint_limits"]) != expected:
                    raise RuntimeError(f"{arm}: SDK 관절 한계 배열 순서가 다르다")
                self._configs[arm] = config
                # 펌웨어별 드라이버로 재연결하면 관절 수신 캐시도 새로 시작한다.
                deadline = time.monotonic() + 2.0
                while True:
                    sample = driver.get_joint_angles()
                    if (sample is not None and np.isfinite(sample.hz) and sample.hz > 0
                            and len(sample.msg) == len(names)
                            and np.isfinite(sample.timestamp) and sample.timestamp > 0
                            and np.isfinite(sample.msg).all()):
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"{arm} ({ports[arm]}): 2초 안에 유효한 관절 피드백을 받지 못했다")
                    time.sleep(0.01)

            # 카메라는 생성 시 한 번만 감지한다. 장치가 없을 때만 건너뛴다.
            # SDK 미설치/조회 실패/스트림 시작 실패를 "카메라 없음"으로 숨기지 않는다.
            import pyorbbecsdk as sdk
            self._camera_sdk = sdk
            self._camera_context = sdk.Context()
            self._camera_context.set_timestamp_clock_type(sdk.OBClockType.REALTIME)
            if self._camera_context.get_timestamp_clock_type() != sdk.OBClockType.REALTIME:
                raise RuntimeError("카메라 SDK 시계가 REALTIME이 아니다")
            devices = self._camera_context.query_devices()
            missing = []
            for role, pid, model in (("wrist", 0x0840, "305"), ("gaze", 0x0800, "335")):
                serial = self._camera_serials[f"{role}_serial"]
                model_indices = [i for i in range(devices.get_count())
                                 if devices.get_device_pid_by_index(i) == pid]
                if not model_indices:
                    missing.append(f"{role} ({model})")
                    continue
                matches = [i for i in model_indices
                           if serial is None or devices.get_device_serial_number_by_index(i) == serial]
                if len(matches) != 1:
                    raise RuntimeError(f"{role}: 모델은 있지만 지정한 시리얼과 일치하는 장치가 "
                                       f"{len(matches)}대다. 연결 장치와 시리얼 설정을 확인하세요")
                device = devices.get_device_by_index(matches[0])
                if not device.is_global_timestamp_supported():
                    raise RuntimeError(f"{role}: SDK global timestamp를 지원하지 않는다")
                # SDK가 장치 시각을 호스트 REALTIME 시계에 대응시킨다.
                # 우리 코드의 변환은 반환 시 us -> s (1e-6)뿐이고 지연 보정은 하지 않는다.
                device.enable_global_timestamp(True)
                pipeline = sdk.Pipeline(device)
                pipeline.disable_health_monitor()
                try:
                    color_profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
                except Exception as exc:
                    raise RuntimeError(f"{role}: 대표 Color가 없다. 좌/우 선택이나 preset 변경은 하지 않는다") from exc
                # 두 색상 스트림 모두 명시한 원본 모드만 사용한다. 기본값 대체/resize 없음.
                width, height, color_format, fps = (
                    (1280, 800, sdk.OBFormat.YUYV, 30) if role == "wrist"
                    else (1280, 720, sdk.OBFormat.YUYV, 30))
                requested_mode = f"{width}x{height} {color_format} {fps}fps"
                try:
                    color_profile = color_profiles.get_video_stream_profile(
                        width, height, color_format, fps)
                except Exception as exc:
                    raise RuntimeError(
                        f"{role}: Color {requested_mode}를 사용할 수 없다. 다른 모드로 대체하지 않는다"
                    ) from exc
                if (color_profile.get_width(), color_profile.get_height(),
                        color_profile.get_format(), color_profile.get_fps()) != (
                        width, height, color_format, fps):
                    raise RuntimeError(f"{role}: SDK가 요청한 {requested_mode}와 다른 프로필을 반환했다")
                if color_profile.get_format() not in (
                        sdk.OBFormat.RGB, sdk.OBFormat.BGR, sdk.OBFormat.MJPG,
                        sdk.OBFormat.YUYV, sdk.OBFormat.YUY2, sdk.OBFormat.UYVY):
                    raise RuntimeError(f"{role}: 지원하지 않는 Color 포맷 {color_profile.get_format()}")
                config = sdk.Config()
                config.disable_all_stream()
                config.enable_stream(color_profile)
                align = None
                if role == "gaze":
                    depth_profile = pipeline.get_stream_profile_list(
                        sdk.OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
                    if depth_profile.get_format() != sdk.OBFormat.Y16:
                        raise RuntimeError("gaze: SDK 기본 깊이 포맷이 Y16이 아니다")
                    config.enable_stream(depth_profile)
                    config.set_align_mode(sdk.OBAlignMode.DISABLE)
                    config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                    pipeline.enable_frame_sync()
                    align = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
                    align.set_match_target_resolution(True)
                for name in ("OB_PROP_COLOR_MIRROR_BOOL", "OB_PROP_COLOR_FLIP_BOOL", "OB_PROP_COLOR_ROTATE_INT"):
                    prop = getattr(sdk.OBPropertyID, name)
                    if device.is_property_supported(prop, sdk.OBPermissionType.PERMISSION_READ):
                        value = device.get_int_property(prop) if name.endswith("_INT") else device.get_bool_property(prop)
                        if value:
                            raise RuntimeError(f"{role}: 장치 기하 설정 {name}={value}. 임의 보정하지 않는다")
                latest = deque(maxlen=1)
                self._camera_streams[role] = {
                    "pipeline": pipeline, "device": device, "latest": latest, "align": align}
                # SDK 수신 콜백으로 최신 frameset 한 개만 보관한다.
                pipeline.start(config, latest.append)
                deadline = time.monotonic() + 3.0
                while not latest:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{role}: 3초 안에 최초 프레임을 받지 못했다")
                    time.sleep(0.001)  # 초기 연결 준비만 대기. 제어 주기 조절이 아니다.
            if not self._camera_streams:
                self._camera_context = None
                self._camera_sdk = None
                print("카메라 연결을 사용하지 않도록 생성한 객체입니다 (305·335 미연결, 관절 사용 가능)")
            elif missing:
                print(f"카메라 미연결: {', '.join(missing)} — 해당 grab 함수만 사용할 수 없습니다")
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                raise RuntimeError(f"초기 연결 실패: {exc}; 연결 해제 실패: {cleanup_error}") from exc
            raise

    # ---------------------------------------------------------------- 관절
    def read_joints(self) -> dict:
        """현재 관절 상태. 형식은 아래 그대로.

            {"nero": [q1..q7],      # rad, URDF 순서
             "piper": [q1..q6],     # rad, URDF 순서
             "grip": 0.0,           # 임시값. 실제 그리퍼 피드백이 아니다.
             "t": 1789040210.804}   # 두 팔 SDK 표본 시각 중 오래된 값, 초

        - **단위는 rad다.** 도가 아니다. 영점과 부호는 URDF와 같다.
        - `t`는 호스트 time.time() 기준 초다(SDK 시각과의 차이를 실측 보고한다).
          관절과 이미지의 `t` 차이가 33 ms를 넘으면
          위층에서 그 표본을 버려야 한다.
        - **팔 하나만 읽는 함수를 따로 두지 않는다.** 한 dict에 둘 다 담는다.
        """
        if self._closed:
            raise RuntimeError("RealRobot 연결이 닫혀 있다")
        result = {}
        sample_times = {}
        for arm, names in (("nero", NERO_JOINTS), ("piper", PIPER_JOINTS)):
            sample = self._arms[arm].get_joint_angles()
            if sample is None or not np.isfinite(sample.hz) or sample.hz <= 0:
                raise RuntimeError(f"{arm}: 유효한 관절 피드백이 없다")
            values = list(sample.msg)
            if len(values) != len(names) or any(
                isinstance(q, (bool, np.bool_))
                or not isinstance(q, (int, float, np.integer, np.floating))
                or not np.isfinite(q) for q in values
            ):
                raise RuntimeError(f"{arm}: 잘못된 관절 피드백")
            stamp = float(sample.timestamp)
            if not np.isfinite(stamp) or stamp <= 0:
                raise RuntimeError(f"{arm}: 잘못된 SDK 표본 시각")
            result[arm] = values  # SDK joint1… 순서, rad 그대로 복사
            sample_times[arm] = stamp
        self._sample_times = sample_times
        # 두 팔의 SDK 캐시 시각 중 오래된 시각을 대표값으로 사용한다.
        # 같은 캐시를 다시 읽으면 t도 같다. 현재 시각으로 덮어쓰지 않는다.
        # SDK가 묶은 관절들의 동시 측정이나 카메라와의 동기화를 보장하지 않는다.
        result["t"] = min(sample_times.values())
        result["grip"] = 0.0  # 사용자 승인 임시값: 실제 그리퍼 개방 측정값이 아니다.
        return result

    def command_joints(self, cmd: dict) -> None:
        """위치 지령. `read_joints`와 같은 dict 형식을 받는다.

        - **속도·토크 지령은 이 규약에 없다.**
        - **보내고 바로 돌아온다.** 도달을 기다리는 것은 위층이 한다.
        - 한쪽 팔만 움직이려면 다른 쪽에 방금 읽은 값을 그대로 넣어 보낸다.
        - 입력 검증 후 각 팔의 move_js를 한 번 호출한다. 보간·스무딩·클램프는 없다.
          SDK가 클램프할 입력은 전송 전에 예외로 알린다.
        - 내부 주기 조절이나 도달 대기는 없다. 호출 주기는 위층에서 관리한다.
        - **안전 정지는 위층이 갖는다.** 재생 스크립트 또는 평가 루프가 담당한다.
        """
        from pyAgxArm.utiles.validator import Validator

        if self._closed:
            raise RuntimeError("RealRobot 연결이 닫혀 있다")
        if not isinstance(cmd, dict) or not {"nero", "piper", "grip", "t"} <= cmd.keys():
            raise ValueError("cmd에는 nero, piper, grip, t가 모두 필요하다")
        for key in ("grip", "t"):
            value = cmd[key]
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)):
                raise ValueError(f"{key}: 유한한 숫자가 필요하다")
        # custom gripper가 없는 동안 지원하지 않는 명령을 조용히 버리지 않는다.
        if cmd["grip"] != 0.0:
            raise NotImplementedError("그리퍼 미구현: 임시 grip=0.0만 허용한다")
        prepared = {}
        for arm, names in (("nero", NERO_JOINTS), ("piper", PIPER_JOINTS)):
            try:
                values = list(cmd[arm])
            except TypeError as exc:
                raise ValueError(f"{arm}: 관절 배열이 필요하다") from exc
            if len(values) != len(names):
                raise ValueError(f"{arm}: 관절 {len(names)}개가 필요하다")
            driver = self._arms[arm]
            if not driver.is_connected():
                raise RuntimeError(f"{arm}: SDK 연결이 끊겼다")
            if not driver.get_auto_set_motion_mode_enabled():
                raise RuntimeError(f"{arm}: SDK의 JS 모드 자동 설정이 꺼져 있다")
            if driver.get_joint_limits_enabled():
                limits = list(self._configs[arm]["joint_limits"].values())
            else:
                limits = [(Validator.REF_MIN_ANGLE, Validator.REF_MAX_ANGLE)] * len(names)
            for name, q, (lower, upper) in zip(names, values, limits):
                if (isinstance(q, (bool, np.bool_))
                        or not isinstance(q, (int, float, np.integer, np.floating))
                        or not np.isfinite(q)):
                    raise ValueError(f"{name}: 유한한 rad 값이 필요하다")
                if not lower <= q <= upper:
                    raise ValueError(f"{name}: {q} rad는 SDK가 클램프할 값이다 [{lower}, {upper}]")
            prepared[arm] = values
        # 양쪽 검증 완료 후 각 한 번 전송. CAN 통신 중 실패하면 이미 전송한 팔을
        # 되돌릴 수 없으며, 재시도/보정 없이 예외를 호출자에게 전달한다.
        for arm in ("nero", "piper"):
            self._arms[arm].move_js(prepared[arm])

    def set_gripper(self, grip: float) -> None:
        """미구현. 규약상 0.0은 완전 개방, 1.0은 완전 폐합이다."""
        raise NotImplementedError

    # ---------------------------------------------------------------- 카메라
    def grab_wrist(self) -> tuple:
        """NERO 손목 카메라(305). (rgb, t)를 반환한다.

        rgb: (800, 1280, 3) uint8 RGB (305 Color 1280x800 YUYV 30fps 수신).
        t: 대표 Color 프레임 시각(호스트 time.time() 기준 초).
        SDK global timestamp를 사용한다. 좌/우 스트림으로 자동 대체하지 않는다.
        캐시된 영상을 반환할 때 현재 호출 시각으로 덮어쓰지 않는다.

        수신 영상의 크기를 유지하며 crop/resize/왜곡 보정을 적용하지 않는다.
        SDK 포맷에 맞춰 RGB로 변환한다. 채널 순서는 별도 실물 확인이 필요하다.
        최신 수신 프레임을 반환한다. 호출 순간의 촬영을 보장하지 않으며 로봇을 움직이지 않는다.
        """
        if self._closed:
            raise RuntimeError("RealRobot 연결이 닫혀 있다")
        if "wrist" not in self._camera_streams:
            raise RuntimeError("wrist (305): 카메라 연결을 사용하지 않도록 생성한 객체입니다 "
                               "(초기화 시 미연결). 관절과 연결된 다른 카메라는 사용 가능합니다")
        latest = self._camera_streams["wrist"]["latest"]
        if not latest:
            raise RuntimeError("wrist: 카메라 프레임이 없다")
        frames = latest[-1]
        color = frames.get_color_frame()
        if color is None:
            raise RuntimeError("wrist: 대표 Color 프레임이 없다")
        global_us = int(color.get_global_timestamp_us())
        if global_us <= 0:
            raise RuntimeError("wrist: global timestamp가 없다. 현재 시각으로 대체하지 않는다")
        t = float(global_us * 1e-6)  # SDK 호스트 시계 변환 결과: us -> s. 지연 보정 없음.
        sdk = self._camera_sdk
        height, width = color.get_height(), color.get_width()
        data = np.frombuffer(color.get_data(), dtype=np.uint8)
        fmt = color.get_format()
        if fmt == sdk.OBFormat.RGB:
            rgb = data.reshape(height, width, 3).copy()
        elif fmt == sdk.OBFormat.BGR:
            rgb = data.reshape(height, width, 3)[..., ::-1].copy()
        elif fmt == sdk.OBFormat.MJPG:
            import cv2
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("MJPG 색상 디코딩 실패")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif fmt in (sdk.OBFormat.YUYV, sdk.OBFormat.YUY2, sdk.OBFormat.UYVY):
            import cv2
            conversion = cv2.COLOR_YUV2RGB_UYVY if fmt == sdk.OBFormat.UYVY else cv2.COLOR_YUV2RGB_YUY2
            rgb = cv2.cvtColor(data.reshape(height, width, 2), conversion)
        else:
            raise RuntimeError(f"지원하지 않는 색상 포맷: {fmt}")
        if rgb.shape != (height, width, 3):
            raise RuntimeError("색상 디코딩 전후 크기가 다르다")
        return rgb, t

    def grab_gaze(self) -> tuple:
        """PiPER 끝단 시선 카메라(335). (rgb, depth, t)를 반환한다.

        rgb: (H, W, 3) uint8 RGB. depth: (H, W) float32 m.
        t: 색상 프레임의 시각(호스트 time.time() 기준 초).
        SDK global timestamp를 사용하며 캐시 시각을 현재 호출 시각으로 덮어쓰지 않는다.
        SDK의 같은 frameset 안의 깊이를 색상 격자에 정렬한다. RGB 자체는 변형하지 않는다.

        깊이 0 및 비유한 값은 nan으로 바꾼다. 색상에는 추가 왜곡 보정을 하지 않는다.
        최신 수신 frameset을 사용하며 호출 순간의 촬영이나 RGB-depth 동시 노출은 보장하지 않는다.
        안에서 로봇을 움직이지 않는다.
        """
        if self._closed:
            raise RuntimeError("RealRobot 연결이 닫혀 있다")
        if "gaze" not in self._camera_streams:
            raise RuntimeError("gaze (335): 카메라 연결을 사용하지 않도록 생성한 객체입니다 "
                               "(초기화 시 미연결). 관절과 연결된 다른 카메라는 사용 가능합니다")
        stream = self._camera_streams["gaze"]
        if not stream["latest"]:
            raise RuntimeError("gaze: 카메라 프레임이 없다")
        frames = stream["latest"][-1]
        color = frames.get_color_frame()
        if color is None or frames.get_depth_frame() is None:
            raise RuntimeError("gaze: 같은 frameset에 Color/depth가 모두 있어야 한다")
        global_us = int(color.get_global_timestamp_us())
        if global_us <= 0:
            raise RuntimeError("gaze: global timestamp가 없다. 현재 시각으로 대체하지 않는다")
        t = float(global_us * 1e-6)  # 깊이가 아닌 색상 프레임의 시각.
        # 색상은 원본 프레임에서 디코딩한다. 깊이만 뒤에서 정렬한다.
        sdk = self._camera_sdk
        height, width = color.get_height(), color.get_width()
        data = np.frombuffer(color.get_data(), dtype=np.uint8)
        fmt = color.get_format()
        if fmt == sdk.OBFormat.RGB:
            rgb = data.reshape(height, width, 3).copy()
        elif fmt == sdk.OBFormat.BGR:
            rgb = data.reshape(height, width, 3)[..., ::-1].copy()
        elif fmt == sdk.OBFormat.MJPG:
            import cv2
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("MJPG 색상 디코딩 실패")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif fmt in (sdk.OBFormat.YUYV, sdk.OBFormat.YUY2, sdk.OBFormat.UYVY):
            import cv2
            conversion = cv2.COLOR_YUV2RGB_UYVY if fmt == sdk.OBFormat.UYVY else cv2.COLOR_YUV2RGB_YUY2
            rgb = cv2.cvtColor(data.reshape(height, width, 2), conversion)
        else:
            raise RuntimeError(f"지원하지 않는 색상 포맷: {fmt}")
        if rgb.shape != (height, width, 3):
            raise RuntimeError("색상 디코딩 전후 크기가 다르다")
        aligned = stream["align"].process(frames)
        if aligned is None:
            raise RuntimeError("gaze: SDK 깊이 정렬 결과가 없다")
        depth_frame = aligned.as_frame_set().get_depth_frame()
        if depth_frame is None:
            raise RuntimeError("gaze: 정렬된 깊이 프레임이 없다")
        if depth_frame.get_format() != self._camera_sdk.OBFormat.Y16:
            raise RuntimeError(f"gaze: 지원하지 않는 깊이 포맷 {depth_frame.get_format()}")
        scale_mm = float(depth_frame.get_depth_scale())
        if not np.isfinite(scale_mm) or scale_mm <= 0:
            raise RuntimeError(f"gaze: 잘못된 깊이 단위 {scale_mm}")
        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width())
        depth = raw.astype(np.float32) * np.float32(scale_mm * 0.001)
        depth[(raw == 0) | ~np.isfinite(depth)] = np.nan
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError(f"gaze: 정렬 깊이 {depth.shape}와 색상 {rgb.shape[:2]}가 다르다")
        return rgb, depth, t

    # ---------------------------------------------------------------- 종료
    def close(self) -> None:
        """연결을 닫는다. 두 번 불러도 죽지 않게 만든다."""
        self._closed = True
        errors = []
        for role, stream in list(self._camera_streams.items()):
            try:
                stream["pipeline"].stop()
            except Exception as exc:
                errors.append((f"camera:{role}", exc))
            else:
                stream["latest"].clear()
                del self._camera_streams[role]
        if not self._camera_streams:
            self._camera_context = None
            self._camera_sdk = None
        for arm, driver in list(self._arms.items()):
            try:
                driver.disconnect()
            except Exception as exc:
                errors.append((arm, exc))
            else:
                del self._arms[arm]
        if errors:
            # 실패한 연결은 보존하여 다음 close()에서 다시 해제를 시도할 수 있다.
            raise RuntimeError(f"연결 해제 실패: {[arm for arm, _ in errors]}") from errors[0][1]


# ==================== 실물 관절·카메라 형식 검사 ====================
def check_shapes(bot) -> None:
    """형식·기본 값 범위·시각 증가를 검사하고, 현재 위치 지령을 보낸 뒤 연결을 닫는다."""
    j = bot.read_joints()
    assert set(j) >= {"nero", "piper", "grip", "t"}, f"키가 모자라다: {sorted(j)}"
    assert len(j["nero"]) == 7, f"nero 관절 {len(j['nero'])}개 (7이어야 한다)"
    assert len(j["piper"]) == 6, f"piper 관절 {len(j['piper'])}개 (6이어야 한다)"
    # 실제 그리퍼 피드백과 개폐 지령은 미구현이므로 검사하지 않는다.
    # rad인지 도인지 여기서 단정할 수는 없다. 다만 도라면 거의 확실히 범위를 넘는다.
    q = np.abs(np.r_[j["nero"], j["piper"]])
    if q.max() > 2.0 * np.pi:
        raise AssertionError(f"관절 최대 |q| = {q.max():.1f}. 도로 주고 있지 않은가")

    # 카메라 반환 형식 검사. 시각 차이 실측·RGB 실물 확인은 별도로 한다.
    w, wrist_t = bot.grab_wrist()
    assert np.isfinite(wrist_t), "손목 영상 시각이 유한하지 않다"
    assert w.ndim == 3 and w.shape[2] == 3 and w.dtype == np.uint8, \
        f"손목 색상 {w.shape} {w.dtype} (H,W,3 uint8이어야 한다)"
    g, d, gaze_t = bot.grab_gaze()
    assert np.isfinite(gaze_t), "시선 영상 시각이 유한하지 않다"
    assert g.shape[:2] == d.shape[:2], f"시선 색상 {g.shape[:2]} 대 깊이 {d.shape[:2]} 불일치"
    assert g.ndim == 3 and g.shape[2] == 3 and g.dtype == np.uint8, \
        f"시선 색상 {g.shape} {g.dtype} (H,W,3 uint8이어야 한다)"
    fin = np.isfinite(d)
    assert fin.any(), "깊이가 전부 무효다"
    # 0 무효값을 **먼저** 따로 본다. np.isfinite는 0.0을 유효로 보므로, 아래 nan 검사만
    # 두면 "무효를 0으로 채운" 경우가 그대로 통과하고 범위 검사가 엉뚱한 메시지를 낸다.
    n0 = int(np.count_nonzero(d[fin] == 0.0))
    assert n0 == 0, (f"깊이에 정확히 0인 화소가 {n0}개다. 무효 화소를 0으로 채운 것이 아닌가. "
                     "규약은 nan을 요구한다")
    bad = (~fin) & (d == d)        # 유한하지 않은데 자기와 같다 = inf
    assert not bad.any(), f"무효 화소 {int(bad.sum())}개가 nan이 아니다(inf인 듯). nan이어야 한다"
    lo, hi = float(np.min(d[fin])), float(np.max(d[fin]))
    assert hi < 30.0, f"깊이 최대 {hi:.1f} (m가 아니라 mm로 주고 있지 않은가)"
    assert lo > 0.02, f"깊이 최소 {lo:.4f} m. 센서 하한보다 작다"

    # 실제 위치 지령 전송과 close의 반복 호출을 확인한다.
    bot.command_joints(bot.read_joints())   # 방금 읽은 값을 그대로 = 제자리 지령
    t0 = bot.read_joints()["t"]
    time.sleep(0.05)
    assert bot.read_joints()["t"] > t0, "t가 증가하지 않는다"
    bot.close()
    bot.close()                             # 두 번 불러도 죽지 않아야 한다
    print("관절·카메라 형식 검사 통과 — 그리퍼 검사는 주석 처리된 상태입니다.")
    print("이 검사가 못 잡는 것: 채널 순서(BGR/RGB), 그리퍼 척도, 관절 순서·부호, 보간 주입")


if __name__ == "__main__":
    check_shapes(RealRobot())
