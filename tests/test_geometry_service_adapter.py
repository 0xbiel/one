from app.config import Settings
from app.geometry import HttpRoomLayoutService, RoomLayoutServiceUnavailable, localization_progress


def test_camera_localization_retries_transient_worker_failures(monkeypatch):
    service = HttpRoomLayoutService(
        Settings(
            geometry_service_url="http://vision-worker:8090",
            geometry_timeout_seconds=1.0,
            geometry_require_gpu=False,
        )
    )
    attempts: list[str] = []
    delays: list[float] = []

    def post(path: str, payload: dict):
        attempts.append(path)
        if len(attempts) < 3:
            raise RoomLayoutServiceUnavailable("http_503")
        return {"status": "needs_rescan"}

    monkeypatch.setattr(service, "_post", post)
    monkeypatch.setattr("app.geometry.time.sleep", delays.append)

    result = service.localize_camera(
        landmarks=[{"point": [0.0, 0.0, 0.0]}] * 6,
        frames=[{"frame_base64": "frame", "width": 640, "height": 480}],
        intrinsics=None,
        fov_degrees=None,
    )

    assert result == {"status": "needs_rescan"}
    assert attempts == ["camera-localization", "camera-localization", "camera-localization"]
    assert delays == [0.5, 1.5]


def test_camera_localization_does_not_retry_unconfigured_worker(monkeypatch):
    service = HttpRoomLayoutService(Settings(geometry_service_url=""))
    sleeps: list[float] = []
    monkeypatch.setattr("app.geometry.time.sleep", sleeps.append)

    try:
        service.localize_camera(
            landmarks=[{"point": [0.0, 0.0, 0.0]}] * 6,
            frames=[{"frame_base64": "frame", "width": 640, "height": 480}],
            intrinsics=None,
            fov_degrees=None,
        )
    except RoomLayoutServiceUnavailable as exc:
        assert exc.code == "not_configured"
    else:  # pragma: no cover - protects the test if the adapter stops failing closed
        raise AssertionError("unconfigured worker did not fail closed")

    assert sleeps == []


def test_camera_localization_progress_job_reports_monotonic_progress(monkeypatch):
    service = HttpRoomLayoutService(
        Settings(
            geometry_service_url="http://vision-worker:8090",
            geometry_timeout_seconds=5.0,
            geometry_localization_stall_timeout_seconds=180.0,
            geometry_require_gpu=False,
        )
    )
    states = iter([
        {"status": "running", "progress": 1, "stage": "Preparing fixed-camera reference frames"},
        {"status": "running", "progress": 42, "stage": "Matching reference frame 2 to RoomPlan landmarks"},
        {"status": "complete", "progress": 100, "stage": "Camera pose solved", "result": {"status": "positioned"}},
    ])
    progress: list[tuple[int, str]] = []

    monkeypatch.setattr(service, "_post", lambda path, payload: {"job_id": "job-1", "status": "running"})
    monkeypatch.setattr(service, "_get", lambda path, timeout=None: next(states))
    monkeypatch.setattr("app.geometry.time.sleep", lambda _delay: None)

    with localization_progress(lambda value, stage: progress.append((value, stage))):
        result = service.localize_camera(
            landmarks=[{"point": [0.0, 0.0, 0.0]}] * 6,
            frames=[{"frame_base64": "frame", "width": 640, "height": 480}],
            intrinsics=None,
            fov_degrees=None,
        )

    assert result == {"status": "positioned"}
    assert [value for value, _stage in progress] == [1, 42, 100]


def test_camera_localization_progress_job_retries_transient_job_creation(monkeypatch):
    service = HttpRoomLayoutService(
        Settings(
            geometry_service_url="http://vision-worker:8090",
            geometry_timeout_seconds=5.0,
            geometry_localization_stall_timeout_seconds=180.0,
            geometry_require_gpu=False,
        )
    )
    attempts: list[str] = []
    delays: list[float] = []

    def post(path: str, payload: dict):
        attempts.append(path)
        if len(attempts) < 3:
            raise RoomLayoutServiceUnavailable("http_503")
        return {"job_id": "job-1", "status": "running"}

    monkeypatch.setattr(service, "_post", post)
    monkeypatch.setattr(service, "_get", lambda path, timeout=None: {"status": "complete", "progress": 100, "result": {"status": "needs_rescan"}})
    monkeypatch.setattr("app.geometry.time.sleep", delays.append)

    with localization_progress(lambda _value, _stage: None):
        result = service.localize_camera(
            landmarks=[{"point": [0.0, 0.0, 0.0]}] * 6,
            frames=[{"frame_base64": "frame", "width": 640, "height": 480}],
            intrinsics=None,
            fov_degrees=None,
        )

    assert result == {"status": "needs_rescan"}
    assert attempts == ["camera-localization/jobs"] * 3
    assert delays == [0.5, 1.5]


def test_camera_localization_progress_job_times_out_only_after_progress_stalls(monkeypatch):
    service = HttpRoomLayoutService(
        Settings(
            geometry_service_url="http://vision-worker:8090",
            geometry_timeout_seconds=5.0,
            geometry_localization_stall_timeout_seconds=180.0,
            geometry_require_gpu=False,
        )
    )
    times = iter([0.0, 0.0, 181.0])
    monkeypatch.setattr(service, "_post", lambda path, payload: {"job_id": "job-1", "status": "running"})
    monkeypatch.setattr(service, "_get", lambda path, timeout=None: {"status": "running", "progress": 1, "stage": "Preparing"})
    monkeypatch.setattr("app.geometry.time.monotonic", lambda: next(times))

    try:
        with localization_progress(lambda _value, _stage: None):
            service.localize_camera(
                landmarks=[{"point": [0.0, 0.0, 0.0]}] * 6,
                frames=[{"frame_base64": "frame", "width": 640, "height": 480}],
                intrinsics=None,
                fov_degrees=None,
            )
    except RoomLayoutServiceUnavailable as exc:
        assert exc.code == "timeout"
    else:  # pragma: no cover
        raise AssertionError("stalled localization did not time out")
