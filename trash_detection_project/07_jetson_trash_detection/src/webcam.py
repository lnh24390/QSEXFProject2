import threading
import time
from pathlib import Path

import cv2


def csi_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=(string)BGRx ! videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
    )


class WebCam:
    """웹캠 / CSI 카메라 / 동영상 파일을 동일한 인터페이스로 읽는다.

    카메라 입력은 별도 스레드에서 계속 읽어 가장 최신 프레임만 남긴다.
    메인 루프가 추론하는 동안 쌓인 프레임을 나중에 처리하면 화면이 밀리기 때문에,
    오래된 프레임은 버리는 편이 체감 FPS와 지연 모두에 유리하다.
    """

    MAX_FAILS = 5

    def __init__(self, source=0, width=1280, height=720, fps=30, csi=False,
                 flip_method=0, mjpg=True, threaded=True, exposure=None):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.csi = csi
        self.flip_method = flip_method
        self.mjpg = mjpg
        # None 이면 자동노출. 역광(밝은 창문 등을 배경으로 찍는 경우) 이면
        # 자동노출이 배경에 맞춰져 전경이 새까맣게 나온다 — 그럴 때 수동 값을 쓴다.
        self.exposure = exposure
        self.cap = None
        self.is_file = isinstance(source, str) and Path(source).exists()
        # 동영상 파일은 프레임을 버리면 안 되므로 스레드를 쓰지 않는다.
        self.threaded = threaded and not self.is_file
        self._fail_count = 0

        self._thread = None
        self._lock = threading.Lock()  # _latest/_latest_id 프레임 버퍼 보호용
        self._cap_lock = threading.Lock()  # self.cap 자체(read/set/재연결) 보호용
        self._latest = None
        self._latest_id = 0
        self._last_taken_id = 0
        self._running = False
        self._dropped = 0

    # ------------------------------------------------------------------ open

    def _open(self):
        if self.csi:
            sensor_id = int(self.source) if str(self.source).isdigit() else 0
            pipeline = csi_pipeline(
                sensor_id=sensor_id,
                width=self.width,
                height=self.height,
                fps=self.fps,
                flip_method=self.flip_method,
            )
            return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if self.is_file:
            return cv2.VideoCapture(self.source)

        # 카메라: 플랫폼별 백엔드를 순서대로 시도한다.
        for backend in (cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY):
            try:
                cap = cv2.VideoCapture(self.source, backend)
            except cv2.error:
                continue
            if cap.isOpened():
                return cap
            cap.release()

        return cv2.VideoCapture(self.source)

    def _configure(self):
        if self.is_file or self.csi:
            return

        # MJPG를 먼저 지정해야 한다. 기본값인 YUYV는 USB 2.0 대역폭 때문에
        # 720p에서 5~10 FPS로 제한되며, 이것만으로 전체 FPS가 묶인다.
        if self.mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._apply_exposure()

    def _apply_exposure(self):
        if self.exposure is None:
            return
        # v4l2 규약: AUTO_EXPOSURE 3=자동, 1=수동. 순서가 중요하다(자동 상태에서
        # EXPOSURE 를 바로 세팅하면 드라이버가 무시하는 경우가 있다).
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    def set_exposure(self, exposure):
        """실행 중 노출을 바꾼다. None 이면 자동노출로 되돌린다.

        백그라운드 스레드가 계속 cap.read() 를 부르고 있어, 여기서 cap.set() 을
        그냥 부르면 두 호출이 겹쳐 조용히 무시될 수 있다(콘솔에는 값이 바뀌었다고
        찍히는데 실제 화면 밝기는 안 바뀌는 증상으로 나타난다). _cap_lock 으로
        read()/set() 이 서로 끼어들지 않게 한다.
        """
        self.exposure = exposure
        if self.cap is None or self.is_file or self.csi:
            return
        with self._cap_lock:
            if exposure is None:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            else:
                self._apply_exposure()

    def describe(self):
        """실제로 적용된 해상도 / FPS / 픽셀 포맷을 사람이 읽을 수 있게."""
        if self.cap is None:
            return "camera not opened"

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip() or "?"
        return f"{w}x{h} @ {fps:.0f}fps  fourcc={fourcc}"

    def start(self):
        self.cap = self._open()
        if self.cap is None or not self.cap.isOpened():
            print(f"error: 입력 소스를 열 수 없습니다 -> {self.source}")
            return False

        self._configure()
        print(f"camera started! {self.describe()}")

        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip()
        if not self.is_file and not self.csi and self.mjpg and fourcc not in ("MJPG", "MJPE"):
            print(">> 경고: MJPG가 적용되지 않았습니다. 캡처가 FPS를 제한할 수 있습니다.")
            print("   `v4l2-ctl -d /dev/video0 --list-formats-ext` 로 지원 포맷을 확인하세요.")

        self._fail_count = 0
        if self.threaded:
            self._running = True
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

        return True

    # -------------------------------------------------------------- reading

    def _read_once(self):
        """카메라에서 한 장 읽는다. 실패 시 재연결을 시도한다."""
        with self._cap_lock:
            ret, frame = self.cap.read()
        if ret and frame is not None:
            self._fail_count = 0
            return frame

        if self.is_file:
            return None  # 동영상 재생 종료

        self._fail_count += 1
        print(f"Failed to read frame ({self._fail_count}/{self.MAX_FAILS})")
        if self._fail_count >= self.MAX_FAILS:
            print(">> 입력을 복구하지 못했습니다. 종료합니다.")
            return None

        if not self._reconnect():
            return None
        return self._read_once()

    def _reconnect(self):
        print(">> 입력이 끊겼습니다. 재연결을 시도합니다...")
        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
            time.sleep(1.0)
            self.cap = self._open()
            ok = self.cap is not None and self.cap.isOpened()
            if ok:
                self._configure()
        if ok:
            print(">> 재연결 성공")
        return ok

    def _reader(self):
        while self._running:
            frame = self._read_once()
            if frame is None:
                break
            with self._lock:
                if self._latest_id != self._last_taken_id:
                    self._dropped += 1  # 메인 루프가 가져가기 전에 덮어쓴 프레임
                self._latest = frame
                self._latest_id += 1
        self._running = False

    def get_frame(self, timeout=5.0):
        """가장 최신 프레임을 반환한다. 더 이상 읽을 수 없으면 None."""
        if self.cap is None:
            return None

        if not self.threaded:
            return self._read_once()

        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._latest is not None and self._latest_id != self._last_taken_id:
                    self._last_taken_id = self._latest_id
                    return self._latest
            if not self._running:
                return None
            if time.time() > deadline:
                print(">> 프레임 대기 시간 초과")
                return None
            time.sleep(0.001)

    @property
    def dropped(self):
        return self._dropped

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if self._dropped:
            print(f">> 추론이 따라가지 못해 버린 프레임: {self._dropped}")
        print("exit camera")
