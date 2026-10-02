"""DataSegmentLabeling — 실행 진입점.

    python main.py                 프로젝트 선택 화면으로 시작
    python main.py <프로젝트폴더>   해당 프로젝트를 열고 시작
    python main.py --paths         데이터 폴더 위치를 출력하고 종료
    python main.py --deps          선택 패키지 설치 상태를 출력하고 종료
"""
from __future__ import annotations

import sys
from pathlib import Path


def app_icon():
    """The window/taskbar icon, or None if the assets are missing.

    PyInstaller unpacks bundled data to a temp dir, so the icon is
    looked up under _MEIPASS when frozen rather than beside the exe
    (where only user data lives).
    """
    from PySide6.QtGui import QIcon
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", "") or
                    Path(sys.executable).parent)
    else:
        base = Path(__file__).resolve().parent
    # .ico first: it carries every size Windows asks for.
    for name in ("icon.ico", "icon.svg"):
        q = base / "assets" / name
        if q.exists():
            icon = QIcon(str(q))
            if not icon.isNull():
                return icon
    return None


def _emit(text: str) -> int:
    """Print, and in a windowed build also drop a file beside the exe.

    PyInstaller's windowed bootloader leaves `sys.stdout` as None, so a bare
    print() would raise. The file is what makes these flags usable at all
    when someone double-clicks the exe to find out where its data lives.
    """
    from dsl.core.settings import APP_DIR
    try:
        if sys.stdout is not None:
            print(text)
    except (OSError, ValueError):
        pass
    if getattr(sys, "frozen", False):
        try:
            out = APP_DIR / "diagnostics.txt"
            out.write_text(text, encoding="utf-8")
            if sys.stdout is not None:
                print(f"\n(사본: {out})")
        except OSError:
            pass
    return 0


def _print_paths() -> int:
    """Where this build keeps its data. Works in the frozen exe too.

    Worth having as a flag rather than a docstring: in a PyInstaller build
    the answer depends on where the exe was installed, so "it should be
    beside the executable" is only checkable by asking the app itself.
    """
    from dsl.core.settings import APP_DIR, Settings
    s = Settings.load()
    return _emit("\n".join([
        f"frozen        : {getattr(sys, 'frozen', False)}",
        f"executable    : {sys.executable}",
        f"APP_DIR       : {APP_DIR}",
        f"config        : {s.config_path}",
        f"weights       : {s.weights_path}",
        f"datasets      : {s.export_path}",
        f"runs          : {s.runs_path}",
    ]))


def _print_deps() -> int:
    from dsl.core.deps import missing, report
    n = len(missing())
    return _emit(report() + (f"\n\n미설치 {n}개" if n else "\n\n모두 설치됨"))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    if "--paths" in sys.argv:
        return _print_paths()
    if "--deps" in sys.argv:
        return _print_deps()

    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 가 필요합니다.\n\n    uv pip install PySide6\n")
        return 1

    from dsl.core.settings import Settings
    from dsl.ui.main_window import MainWindow
    from dsl.ui.theme import STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName("DataSegmentLabeling")
    app.setOrganizationName("DataSegmentLabeling")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    icon = app_icon()
    if icon is not None:
        app.setWindowIcon(icon)

    settings = Settings.load()
    win = MainWindow(settings)

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        p = Path(args[0])
        if (p / "project.json").exists():
            win.load_project(p)
        else:
            print(f"프로젝트를 찾지 못했습니다: {p}")

    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
