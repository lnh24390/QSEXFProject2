"""Dark theme + colour helpers."""
from __future__ import annotations

from PySide6.QtGui import QColor

BG = "#1e1f22"
BG_ALT = "#26282c"
BG_DARK = "#141517"
FG = "#e6e6e6"
FG_DIM = "#9aa0a6"
ACCENT = "#4c8dff"
BORDER = "#3a3d42"
OK = "#3ddc84"
WARN = "#ffb300"
ERR = "#ff5252"

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {FG}; font-size: 12px; }}
QMainWindow::separator {{ background: {BORDER}; width: 3px; height: 3px; }}

QToolBar {{ background: {BG_ALT}; border: 0; padding: 3px; spacing: 2px; }}
QToolButton {{ background: transparent; border: 1px solid transparent;
               border-radius: 4px; padding: 4px 8px; }}
QToolButton:hover {{ background: {BORDER}; }}
QToolButton:checked {{ background: {ACCENT}; color: #fff; }}

QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {BG_ALT}; padding: 6px 8px;
                      border-bottom: 1px solid {BORDER}; font-weight: 600; }}

QPushButton {{ background: {BG_ALT}; border: 1px solid {BORDER};
               border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {FG_DIM}; border-color: {BG_ALT}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff;
                       font-weight: 600; }}
QPushButton#danger {{ border-color: {ERR}; color: {ERR}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {BG_DARK}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 4px 6px; selection-background-color: {ACCENT}; }}
QComboBox::drop-down {{ border: 0; width: 18px; }}
QComboBox QAbstractItemView {{ background: {BG_DARK}; border: 1px solid {BORDER};
                               selection-background-color: {ACCENT}; }}

QListView, QTreeView, QTableView {{ background: {BG_DARK}; border: 1px solid {BORDER};
                                    border-radius: 4px; outline: 0; }}
QListView::item {{ padding: 3px; border-radius: 3px; }}
QListView::item:selected, QTreeView::item:selected {{ background: {ACCENT}; color: #fff; }}
QHeaderView::section {{ background: {BG_ALT}; border: 0;
                        border-right: 1px solid {BORDER}; padding: 4px; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 4px; }}
QTabBar::tab {{ background: {BG_ALT}; padding: 6px 12px; border: 1px solid {BORDER};
                border-bottom: 0; border-top-left-radius: 4px;
                border-top-right-radius: 4px; }}
QTabBar::tab:selected {{ background: {BG}; color: {ACCENT}; }}

QScrollBar:vertical {{ background: {BG}; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {FG_DIM}; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

QProgressBar {{ background: {BG_DARK}; border: 1px solid {BORDER};
                border-radius: 4px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QStatusBar {{ background: {BG_ALT}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {FG_DIM}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; margin-top: 14px;
             padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px;
                    color: {FG_DIM}; }}
QMenu {{ background: {BG_ALT}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
QToolTip {{ background: {BG_DARK}; color: {FG}; border: 1px solid {ACCENT};
            padding: 4px; }}
"""


def qcolor(hex_color: str, alpha: int = 255) -> QColor:
    c = QColor(hex_color)
    if not c.isValid():
        c = QColor("#e6194b")
    c.setAlpha(alpha)
    return c


def contrast_text(hex_color: str) -> str:
    c = QColor(hex_color)
    lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return "#000000" if lum > 150 else "#ffffff"
